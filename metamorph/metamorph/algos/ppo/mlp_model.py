"""Simple MLP actor-critic baseline for transfer learning comparison.

This provides a flat MLP policy (no per-limb tokenization, no attention)
as a baseline against the Transformer-based ActorCritic used by LOKI.
It takes the same observation/action interface so it can be used as a
drop-in replacement via MODEL.ACTOR_CRITIC config override.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from metamorph.config import cfg


class MLPActorCritic(nn.Module):
    """MLP-based actor-critic that flattens all limb observations.

    Unlike the Transformer-based ActorCritic, this model treats all
    proprioceptive observations as a single flat vector. It does not
    use per-limb tokenization or attention, making it a simpler
    baseline for measuring the value of the Transformer architecture.
    """

    def __init__(self, obs_space, action_space):
        super().__init__()
        self.seq_len = cfg.MODEL.MAX_LIMBS  # 12

        # Input: flattened proprioceptive observations
        prop_dim = obs_space["proprioceptive"].shape[0]
        self.num_actions = cfg.MODEL.MAX_LIMBS * 2  # 24 (2 joints per limb)

        hidden_dim = 256

        # Value network (critic)
        self.v_net = nn.Sequential(
            nn.Linear(prop_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Policy network (actor)
        self.mu_net = nn.Sequential(
            nn.Linear(prop_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

        # Action log_std — same interface as TransformerModel ActorCritic
        if cfg.MODEL.ACTION_STD_FIXED:
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False,
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.v_net, self.mu_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
        # Smaller init for output layers
        nn.init.orthogonal_(self.v_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.mu_net[-1].weight, gain=0.01)

    def forward(self, obs, act=None, num_samples=-1, return_attention=False):
        if num_samples > 0:
            batch_size = num_samples
        elif act is not None:
            batch_size = cfg.PPO.BATCH_SIZE
        else:
            batch_size = cfg.PPO.NUM_ENVS

        prop_obs = obs["proprioceptive"]
        act_mask = obs["act_padding_mask"].bool()

        if prop_obs.shape[0] == 1:
            batch_size = 1

        # Value prediction
        val = self.v_net(prop_obs)  # (batch, 1)

        # Policy prediction
        mu = self.mu_net(prop_obs)  # (batch, num_actions)

        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            logp = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy
        else:
            if return_attention:
                return val, pi, None, None
            else:
                return val, pi, None, None
