#!/usr/bin/env python3
"""
Randomly sample morphologies from each cluster for qualitative visualization.

Extracts N random XML files from each cluster tar archive.

Usage:
    python tools/sample_cluster_morphologies.py [--num_clusters 20] [--num_samples 5]

Output:
    results/cluster_samples/cluster_{N}/*.xml
"""

import argparse
import os
import random
import tarfile


def main():
    parser = argparse.ArgumentParser(description="Sample morphologies from clusters")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing cluster tar files")
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of morphologies to sample per cluster")
    parser.add_argument("--output_dir", type=str,
                        default="./results/cluster_samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    for cluster_label in range(args.num_clusters):
        tar_path = os.path.join(
            args.data_dir,
            f"latent_cluster{args.num_clusters}_{cluster_label}.tar"
        )
        if not os.path.exists(tar_path):
            print(f"  WARNING: {tar_path} not found, skipping")
            continue

        cluster_out_dir = os.path.join(args.output_dir, f"cluster_{cluster_label}")
        os.makedirs(cluster_out_dir, exist_ok=True)

        # Collect XML members
        xml_members = []
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".xml"):
                    xml_members.append(member)

        n_sample = min(args.num_samples, len(xml_members))
        sampled = random.sample(xml_members, n_sample)

        # Extract sampled XMLs
        with tarfile.open(tar_path, "r") as tar:
            for member in sampled:
                member_name = os.path.basename(member.name)
                out_path = os.path.join(cluster_out_dir, member_name)
                with open(out_path, "wb") as f:
                    f.write(tar.extractfile(member).read())

        print(f"  Cluster {cluster_label:2d}: sampled {n_sample} morphologies "
              f"(from {len(xml_members)} total)")

    print(f"\nSamples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
