import time
from collections import defaultdict
from collections import deque

import gymnasium as gym
import torch


def _patch_gymnasium_for_old_api():
    """Patch gymnasium base classes to support old gym API (4-tuple step, single obs reset).

    This codebase was written for the old gym API. gymnasium 1.x changed step() to
    return 5 values and reset() to return (obs, info). This patch makes the base
    Wrapper/ObservationWrapper/ActionWrapper/RewardWrapper classes compatible with
    the old API so all existing custom wrappers work unchanged.
    """
    def _wrapper_step(self, action):
        return self.env.step(action)

    def _wrapper_reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _obs_wrapper_step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.observation(obs), rew, done, info

    def _obs_wrapper_reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def _action_wrapper_step(self, action):
        return self.env.step(self.action(action))

    def _reward_wrapper_step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, self.reward(rew), done, info

    def _wrapper_getattr(self, name):
        """Delegate attribute access to wrapped env (old gym behavior)."""
        return getattr(self.env, name)

    gym.Wrapper.step = _wrapper_step
    gym.Wrapper.reset = _wrapper_reset
    gym.Wrapper.__getattr__ = _wrapper_getattr
    gym.ObservationWrapper.step = _obs_wrapper_step
    gym.ObservationWrapper.reset = _obs_wrapper_reset
    gym.ActionWrapper.step = _action_wrapper_step
    gym.RewardWrapper.step = _reward_wrapper_step


_patch_gymnasium_for_old_api()

try:
    from metamorph.config import cfg
    from metamorph.envs.tasks.task import make_env as create_task_env
    from metamorph.envs.vec_env.dummy_vec_env import DummyVecEnv
    from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
    from metamorph.envs.vec_env.subproc_vec_env import SubprocVecEnv
    from metamorph.envs.vec_env.vec_normalize import VecNormalize
    from metamorph.envs.wrappers.multi_env_wrapper import MultiEnvWrapper
except:
    import sys
    sys.path.append("..")
    from metamorph.metamorph.config import cfg
    from metamorph.metamorph.envs.tasks.task import make_env as create_task_env
    from metamorph.metamorph.envs.vec_env.dummy_vec_env import DummyVecEnv
    from metamorph.metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
    from metamorph.metamorph.envs.vec_env.subproc_vec_env import SubprocVecEnv
    from metamorph.metamorph.envs.vec_env.vec_normalize import VecNormalize
    from metamorph.metamorph.envs.wrappers.multi_env_wrapper import MultiEnvWrapper

# Map env IDs to max episode steps (matches the gymnasium registration)
_ENV_MAX_STEPS = {
    "Unimal-v0": 1000,
    "Unimal-eval-v0": 200,
}


def _make_raw_env(xml_file=None, tmp_sample=False):
    """Create raw task env without outer wrappers."""
    return create_task_env(agent_name=xml_file, tmp_sample=tmp_sample)


def _wrap_env(env, env_id):
    """Add TimeLimit, TimeLimitMask, and RecordEpisodeStatistics wrappers."""
    max_steps = _ENV_MAX_STEPS.get(env_id, 1000)
    env = OldTimeLimit(env, max_episode_steps=max_steps)
    env = TimeLimitMask(env)
    env = RecordEpisodeStatistics(env)
    return env


def make_env(env_id, seed, rank, xml_file=None, tmp_sample=False):
    def _thunk():
        env = _make_raw_env(xml_file=xml_file, tmp_sample=tmp_sample)
        env = _wrap_env(env, env_id)
        return env

    return _thunk


# Vecenvs take as input list of functions. Dummy wrapper function for multienvs
def env_func_wrapper(env):
    def _thunk():
        return env
    return _thunk


def make_vec_envs(
    xml_file=None,
    training=True,
    norm_rew=True,
    num_env=None,
    save_video=False,
    render_policy=False,
    tmp_sample=False,
    tmp_sample_full=False,
    eval_walker=False,
    seed=None,
    video_idx=0,
    sampled_ids=[],
):
    if not num_env:
        num_env = cfg.PPO.NUM_ENVS

    device = torch.device(cfg.DEVICE)

    if seed is None:
        seed = cfg.RNG_SEED

    if len(cfg.ENV.WALKERS) <= 1 or render_policy or save_video or eval_walker:
        # if len(cfg.ENV.WALKERS) == 1:
        if save_video or eval_walker:
            xml_file = cfg.ENV.WALKERS[video_idx % len(cfg.ENV.WALKERS)]
        else:
            xml_file = cfg.ENV.WALKERS[0]
        envs = [
            make_env(cfg.ENV_NAME, seed, idx, xml_file=xml_file)
            for idx in range(num_env)
        ]
    elif tmp_sample:
        envs = [
            # env_func_wrapper(MultiEnvWrapper(make_env(cfg.ENV_NAME, seed, idx, xml_file=str(sampled_ids[idx]), tmp_sample=True)(), idx))
            make_env("Unimal-eval-v0", seed, idx, xml_file=str(sampled_ids[idx]), tmp_sample=True)
            for idx in range(num_env)
        ]
    elif tmp_sample_full:
        envs = [
            # env_func_wrapper(MultiEnvWrapper(make_env(cfg.ENV_NAME, seed, idx, xml_file=str(sampled_ids[idx]), tmp_sample=True)(), idx))
            make_env(cfg.ENV_NAME, seed, idx, xml_file=str(sampled_ids[idx]), tmp_sample=True)
            for idx in range(num_env)
        ]
    else:
        # Dummy init the actual xml_file will change on each reset
        xml_file = cfg.ENV.WALKERS[0]
        envs = []
        for idx in range(num_env):
            # Create raw env, wrap with MultiEnvWrapper, then add outer wrappers
            raw_env = _make_raw_env(xml_file=xml_file)
            multi_env = MultiEnvWrapper(raw_env, idx)
            wrapped_env = _wrap_env(multi_env, cfg.ENV_NAME)
            envs.append(env_func_wrapper(wrapped_env))

    if save_video or render_policy:
        envs = DummyVecEnv([envs[0]])
    elif cfg.VECENV.TYPE == "DummyVecEnv":
        envs = DummyVecEnv(envs)
    elif cfg.VECENV.TYPE == "SubprocVecEnv":
        envs = SubprocVecEnv(envs, in_series=cfg.VECENV.IN_SERIES, context="fork")
    else:
        raise ValueError("VECENV: {} is not supported.".format(cfg.VECENV.TYPE))

    envs = VecNormalize(
        envs, gamma=cfg.PPO.GAMMA, training=training, ret=norm_rew,
        obs_to_norm=cfg.MODEL.OBS_TO_NORM
    )
    envs = VecPyTorch(envs, device)
    return envs


def make_vec_envs_zs():
    device = torch.device(cfg.DEVICE)
    seed = cfg.RNG_SEED
    norm_rew = False
    training = False

    envs = [
        make_env(cfg.ENV_NAME, seed, idx, xml_file=cfg.ENV.WALKERS[0])
        for idx in range(cfg.PPO.NUM_ENVS)
    ]
    if cfg.VECENV.TYPE == "DummyVecEnv":
        envs = DummyVecEnv(envs)
    elif cfg.VECENV.TYPE == "SubprocVecEnv":
        envs = SubprocVecEnv(envs, in_series=cfg.VECENV.IN_SERIES, context="fork")
    else:
        raise ValueError("VECENV: {} is not supported.".format(cfg.VECENV.TYPE))

    envs = VecNormalize(
        envs, gamma=cfg.PPO.GAMMA, training=training, ret=norm_rew,
        obs_to_norm=cfg.MODEL.OBS_TO_NORM
    )
    envs = VecPyTorch(envs, device)
    return envs


# Get a render function
def get_render_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def get_env_attr_from_venv(venv, attr_name):
    if hasattr(venv, "envs"):
        return getattr(venv.envs[0], attr_name)
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None


def get_ob_rms(venv):
    return getattr(get_vec_normalize(venv), "ob_rms", None)


def set_ob_rms(venv, ob_rms):
    vec_norm = get_vec_normalize(venv)
    vec_norm.ob_rms = ob_rms


class OldTimeLimit(gym.Wrapper):
    """TimeLimit wrapper using old gym API (4-tuple step, single obs reset)."""
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["timeout"] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


# Checks whether done was caused by time limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and "timeout" in info:
            info["timeout"] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.t0 = time.time()
        self.episode_return = 0.0
        # Stores individual components of the return. For e.g. return might
        # have separate reward for speed and standing.
        self.episode_return_components = defaultdict(int)
        self.episode_length = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_return += reward
        self.episode_length += 1
        for key, value in info.items():
            if "__reward__" in key:
                self.episode_return_components[key] += value

        if done:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.episode_length,
                "t": round(time.time() - self.t0, 6),
            }
            for key, value in self.episode_return_components.items():
                info["episode"][key] = value
                self.episode_return_components[key] = 0

            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0
        return observation, reward, done, info
