#!/usr/bin/env python3
"""Evaluate whether cross-task policy transfer preserves morphology ranking.

Tests if a policy trained on Task B can zero-shot discriminate which
morphologies are better on Task A — by comparing per-agent rankings.

Uses only UNIQUE morphologies per cluster (deduplicated by XML MD5 hash)
to account for diversity collapse in LOKI's elite pools.

Metrics:
  - Spearman rho and Kendall tau (rank correlations with p-values)
  - Top-1 match: is the best morphology the same?
  - Exact order match: are rankings identical?

Full 2x2 design per cluster:
  AA: Policy A on Env A (baseline)     BA: Policy B on Env A (zero-shot)
  AB: Policy A on Env B (zero-shot)    BB: Policy B on Env B (baseline)

Usage:
    cd metamorph

    # Single cluster
    MUJOCO_GL=egl PYTHONPATH=./ python ../scripts/transfer/eval_ranking_correlation.py \
        --task_a ft --task_b incline --cluster 0

    # Multiple clusters
    MUJOCO_GL=egl PYTHONPATH=./ python ../scripts/transfer/eval_ranking_correlation.py \
        --task_a ft --task_b incline --clusters 0 1 2 3 4

    # All available clusters
    MUJOCO_GL=egl PYTHONPATH=./ python ../scripts/transfer/eval_ranking_correlation.py \
        --task_a ft --task_b incline --all_clusters
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats as sp_stats

from metamorph.config import cfg
from metamorph.algos.ppo.envs import make_vec_envs, set_ob_rms
from metamorph.algos.ppo.model import Agent
from metamorph.utils import sample as su


TASK_TO_DIR = {
    "ft": "ft", "locomotion": "ft",
    "obstacle": "obstacle", "many_obstacle": "many_obstacle",
    "bump": "bump", "incline": "incline",
    "push_box_incline": "push_box_incline",
    "manipulation_ball": "manipulation_ball",
    "exploration": "exploration", "patrol": "patrol",
}
TASK_TO_CFG = {"many_obstacle": "obstacle", "locomotion": "ft"}


def get_task_dir(task):
    return TASK_TO_DIR.get(task, task)


def get_task_cfg(task):
    return TASK_TO_CFG.get(task, task)


def get_checkpoint_path(loki_base, task, cluster, num_clusters, seed):
    return os.path.join(
        loki_base, get_task_dir(task),
        f"kmeans_cluster/{num_clusters}/{cluster}",
        f"walker20/freq2/drop2/seed{seed}", "Unimal-v0.pt",
    )


def get_walker_base(loki_base, task, cluster, num_clusters, seed):
    return os.path.join(
        loki_base, get_task_dir(task),
        f"kmeans_cluster/{num_clusters}/{cluster}",
        f"walker20/freq2/drop2/seed{seed}",
    )


def get_extra_cfg_opts(task):
    if task == "many_obstacle":
        return ["OBJECT.NUM_OBSTACLES", "150"]
    return []


def md5_file(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def find_unique_walkers(walker_base):
    """Find unique morphologies by MD5 dedup from latest xml_step iteration."""
    xml_step_dir = os.path.join(walker_base, "xml_step")
    if not os.path.isdir(xml_step_dir):
        return []

    iters = [int(d) for d in os.listdir(xml_step_dir)
             if d.isdigit() and os.path.isdir(os.path.join(xml_step_dir, d))]
    if not iters:
        return []

    latest_dir = os.path.join(xml_step_dir, str(max(iters)))
    xmls = sorted(f for f in os.listdir(latest_dir)
                  if f.endswith(".xml") and not f.startswith("tmp_"))

    seen_hashes = set()
    unique = []
    for xml in xmls:
        h = md5_file(os.path.join(latest_dir, xml))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append((xml, os.path.join(latest_dir, xml)))
    return unique


def prepare_walker_dir(unique_xmls, out_dir):
    """Create walker dir with only the unique XMLs."""
    wdir = os.path.join(out_dir, "walkers")
    os.makedirs(os.path.join(wdir, "xml"), exist_ok=True)
    os.makedirs(os.path.join(wdir, "xml_step", "0"), exist_ok=True)
    for xml_name, xml_path in unique_xmls:
        shutil.copy2(xml_path, os.path.join(wdir, "xml", xml_name))
        shutil.copy2(xml_path, os.path.join(wdir, "xml_step", "0", xml_name))
    return wdir


def setup_cfg_for_eval(task, walker_dir, out_dir, num_envs=8):
    """Configure cfg for evaluation."""
    cfg.defrost()
    cfg.merge_from_file(f"./configs/{get_task_cfg(task)}.yaml")
    extra = get_extra_cfg_opts(task)
    if extra:
        cfg.merge_from_list(extra)

    cfg.ENV.WALKER_DIR = walker_dir
    cfg.LOKI.TRAIN = True
    cfg.OUT_DIR = out_dir
    cfg.PPO.NUM_ENVS = num_envs

    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS)
        // cfg.PPO.TIMESTEPS
        // cfg.PPO.NUM_ENVS
    )
    if cfg.ENV_NAME == "Unimal-v0":
        cfg.MODEL.MAX_JOINTS = 20
        cfg.MODEL.MAX_LIMBS = 12

    # Infer walkers
    xml_dir = os.path.join(walker_dir, "xml_step", "0")
    cfg.ENV.WALKERS = sorted(
        f.split(".")[0] for f in os.listdir(xml_dir)
        if f.endswith(".xml") and not f.startswith("tmp_")
    )
    walkers = list(cfg.ENV.WALKERS)
    cfg.freeze()
    return walkers


def evaluate_per_agent(checkpoint_path, task, walker_dir, out_dir,
                       num_episodes=50, num_envs=8, device="cuda:0"):
    """Evaluate a checkpoint, return per-agent mean rewards."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    walkers = setup_cfg_for_eval(task, walker_dir, out_dir, num_envs)

    os.makedirs(out_dir, exist_ok=True)
    num_walkers = len(walkers)
    if num_walkers > 1:
        eps_per = max(num_episodes // num_walkers + 1, 5)
        seq = list(range(num_walkers)) * eps_per * num_envs
        with open(os.path.join(out_dir, "sampling.json"), "w") as f:
            json.dump(seq, f)

    su.set_seed(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    envs = make_vec_envs(training=False, norm_rew=False, num_env=num_envs,
                         force_dummy=True)
    ac, ob_rms = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    ac.to(device)
    ac.eval()
    set_ob_rms(envs, ob_rms)
    agent = Agent(ac)

    agent_rewards = defaultdict(list)
    total_target = num_episodes * num_walkers
    collected = 0

    obs = envs.reset()
    while collected < total_target:
        val, act, logp = agent.act(obs)
        obs, reward, done, infos = envs.step(act)
        for info in infos:
            if "episode" in info:
                agent_rewards[info["name"]].append(info["episode"]["r"])
                collected += 1
        if all(len(v) >= num_episodes for v in agent_rewards.values()):
            break

    envs.close()

    return {
        name: float(np.mean(agent_rewards[name][:num_episodes]))
        if name in agent_rewards else float("nan")
        for name in walkers
    }


def compute_ranking_metrics(rewards_1, rewards_2, agents):
    """Compute rank correlation and order-match metrics."""
    vals1 = [rewards_1[a] for a in agents]
    vals2 = [rewards_2[a] for a in agents]

    # Filter out NaN
    valid = [(a, v1, v2) for a, v1, v2 in zip(agents, vals1, vals2)
             if not (np.isnan(v1) or np.isnan(v2))]
    if len(valid) < 3:
        return None

    a_valid, v1_valid, v2_valid = zip(*valid)

    rank1 = sorted(a_valid, key=lambda a: rewards_1[a], reverse=True)
    rank2 = sorted(a_valid, key=lambda a: rewards_2[a], reverse=True)

    spearman_r, spearman_p = sp_stats.spearmanr(v1_valid, v2_valid)
    kendall_t, kendall_p = sp_stats.kendalltau(v1_valid, v2_valid)

    top1_match = rank1[0] == rank2[0]
    exact_match = rank1 == rank2

    # Top-k overlap (how many of top-k are shared)
    n = len(rank1)
    top_k = min(3, n)
    top_k_overlap = len(set(rank1[:top_k]) & set(rank2[:top_k])) / top_k

    return {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_t),
        "kendall_p": float(kendall_p),
        "top1_match": bool(top1_match),
        "exact_order_match": bool(exact_match),
        "top_k_overlap": float(top_k_overlap),
        "top_k": top_k,
        "n_agents": len(a_valid),
        "ranking_1": list(rank1),
        "ranking_2": list(rank2),
    }


def run_cluster(args, cluster):
    """Run 2x2 evaluation for one cluster."""
    print(f"\n{'='*70}")
    print(f"  Cluster {cluster}: {args.task_a} <-> {args.task_b}")
    print(f"{'='*70}")

    ckpt_a = get_checkpoint_path(
        args.loki_base, args.task_a, cluster, args.num_clusters, args.seed)
    ckpt_b = get_checkpoint_path(
        args.loki_base, args.task_b, cluster, args.num_clusters, args.seed)

    if not os.path.isfile(ckpt_a):
        print(f"  [SKIP] No checkpoint: {args.task_a} c{cluster}")
        return None
    if not os.path.isfile(ckpt_b):
        print(f"  [SKIP] No checkpoint: {args.task_b} c{cluster}")
        return None

    # Find unique morphologies (from task_a's elite pool)
    walker_base = get_walker_base(
        args.loki_base, args.task_a, cluster, args.num_clusters, args.seed)
    unique_xmls = find_unique_walkers(walker_base)
    if len(unique_xmls) < 3:
        print(f"  [SKIP] Only {len(unique_xmls)} unique morphologies")
        return None

    out_base = os.path.join(
        args.out_dir, f"{args.task_a}_x_{args.task_b}", f"c{cluster}")
    walker_dir = prepare_walker_dir(unique_xmls, out_base)
    agents = [x[0].split(".")[0] for x in unique_xmls]

    print(f"  Unique morphologies: {len(unique_xmls)} (from {args.task_a} elite pool)")
    for xml_name, _ in unique_xmls:
        print(f"    {xml_name}")

    # 2x2 evaluations
    conditions = {
        "AA": (ckpt_a, args.task_a, f"Policy({args.task_a}) x Env({args.task_a})"),
        "BA": (ckpt_b, args.task_a, f"Policy({args.task_b}) x Env({args.task_a})"),
        "AB": (ckpt_a, args.task_b, f"Policy({args.task_a}) x Env({args.task_b})"),
        "BB": (ckpt_b, args.task_b, f"Policy({args.task_b}) x Env({args.task_b})"),
    }

    per_agent = {}
    for key, (ckpt, task, desc) in conditions.items():
        print(f"\n  [{key}] {desc}")
        eval_out = os.path.join(out_base, f"eval_{key}")
        rewards = evaluate_per_agent(
            ckpt, task, walker_dir, eval_out,
            num_episodes=args.num_episodes, num_envs=args.num_envs,
            device=args.device,
        )
        per_agent[key] = rewards

        ranked = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
        for rank, (name, rew) in enumerate(ranked, 1):
            print(f"    #{rank} {name}: {rew:.1f}")

    # Compare rankings
    print(f"\n  --- Ranking Metrics ---")
    comparisons = [
        ("AA", "BA", f"{args.task_b}'s policy preserves {args.task_a}'s morph ranking?"),
        ("BB", "AB", f"{args.task_a}'s policy preserves {args.task_b}'s morph ranking?"),
        ("AA", "BB", f"Same morphologies excel at both {args.task_a} and {args.task_b}?"),
    ]

    metrics = {}
    for k1, k2, question in comparisons:
        m = compute_ranking_metrics(per_agent[k1], per_agent[k2], agents)
        if m is None:
            print(f"  {k1} vs {k2}: insufficient data")
            continue

        label = f"{k1}_vs_{k2}"
        m["question"] = question
        metrics[label] = m

        sig = "*" if m["spearman_p"] < 0.05 else ""
        print(f"\n  {k1} vs {k2}: {question}")
        print(f"    Spearman rho = {m['spearman_r']:.3f} (p={m['spearman_p']:.3f}{sig})")
        print(f"    Kendall tau  = {m['kendall_tau']:.3f} (p={m['kendall_p']:.3f})")
        print(f"    Top-1 match  = {m['top1_match']}")
        print(f"    Exact order  = {m['exact_order_match']}")
        print(f"    Top-{m['top_k']} overlap = {m['top_k_overlap']:.0%}")
        print(f"    Ranking 1: {m['ranking_1']}")
        print(f"    Ranking 2: {m['ranking_2']}")

    result = {
        "cluster": cluster,
        "agents": agents,
        "n_unique": len(agents),
        "rewards": per_agent,
        "metrics": metrics,
    }

    out_path = os.path.join(out_base, "ranking_results.json")
    os.makedirs(out_base, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate morphology ranking correlation across tasks")
    parser.add_argument("--task_a", required=True)
    parser.add_argument("--task_b", required=True)
    parser.add_argument("--cluster", type=int, default=None)
    parser.add_argument("--clusters", type=int, nargs="+", default=None)
    parser.add_argument("--all_clusters", action="store_true")
    parser.add_argument("--num_clusters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3429)
    parser.add_argument("--loki_base", default="output/loki_500k")
    parser.add_argument("--out_dir", default="output/ranking_correlation")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.cluster is not None:
        cluster_list = [args.cluster]
    elif args.clusters is not None:
        cluster_list = args.clusters
    elif args.all_clusters:
        cluster_list = list(range(args.num_clusters))
    else:
        parser.error("Specify --cluster, --clusters, or --all_clusters")

    print(f"Ranking Experiment: {args.task_a} <-> {args.task_b}")
    print(f"Clusters: {cluster_list}")
    print(f"Episodes per agent: {args.num_episodes}")

    all_results = []
    agg_metrics = defaultdict(lambda: defaultdict(list))

    for c in cluster_list:
        result = run_cluster(args, c)
        if result is None:
            continue
        all_results.append(result)
        for label, m in result["metrics"].items():
            for k, v in m.items():
                if isinstance(v, (int, float, bool)):
                    agg_metrics[label][k].append(v)

    # Cross-cluster summary
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: {len(all_results)} clusters")
        print(f"{'='*70}")

        summary = {}
        for label, vals in agg_metrics.items():
            question = None
            for r in all_results:
                if label in r["metrics"]:
                    question = r["metrics"][label].get("question", "")
                    break

            rs = vals.get("spearman_r", [])
            taus = vals.get("kendall_tau", [])
            top1 = vals.get("top1_match", [])
            exact = vals.get("exact_order_match", [])
            n_sig = sum(1 for p in vals.get("spearman_p", []) if p < 0.05)

            print(f"\n  {label}: {question}")
            if rs:
                print(f"    Spearman rho:  mean={np.mean(rs):.3f} +/- {np.std(rs):.3f}  "
                      f"[{np.min(rs):.3f}, {np.max(rs):.3f}]")
            if taus:
                print(f"    Kendall tau:   mean={np.mean(taus):.3f} +/- {np.std(taus):.3f}")
            if top1:
                print(f"    Top-1 match:   {sum(top1)}/{len(top1)} "
                      f"({sum(top1)/len(top1):.0%})")
            if exact:
                print(f"    Exact order:   {sum(exact)}/{len(exact)} "
                      f"({sum(exact)/len(exact):.0%})")
            print(f"    Significant:   {n_sig}/{len(rs)} (p<0.05)")

            summary[label] = {
                "mean_spearman_r": float(np.mean(rs)) if rs else None,
                "std_spearman_r": float(np.std(rs)) if rs else None,
                "mean_kendall_tau": float(np.mean(taus)) if taus else None,
                "top1_match_rate": float(np.mean(top1)) if top1 else None,
                "exact_order_rate": float(np.mean(exact)) if exact else None,
                "n_significant": n_sig,
                "n_clusters": len(rs),
            }

        # Save aggregate
        agg_path = os.path.join(
            args.out_dir, f"{args.task_a}_x_{args.task_b}", "aggregate.json")
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump({
                "task_a": args.task_a, "task_b": args.task_b,
                "n_clusters": len(all_results),
                "clusters": [r["cluster"] for r in all_results],
                "summary": summary,
                "per_cluster": all_results,
            }, f, indent=2)
        print(f"\nAggregate: {agg_path}")


if __name__ == "__main__":
    main()
