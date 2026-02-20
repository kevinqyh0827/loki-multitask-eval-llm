import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from metamorph.config import cfg
from metamorph.utils import model as tu

from .transformer import TransformerEncoder, TransformerEncoderLayerResidual


class BinaryDifferentiableMasking(nn.Module):
    def __init__(self, num_limbs):
        super().__init__()
        self.mask = nn.Parameter(torch.randn(num_limbs))

    def forward(self, obs_embeds):
        mask_probs = torch.sigmoid(self.mask)
        binary_mask = torch.bernoulli(mask_probs)
        return obs_embeds * binary_mask.unsqueeze(1).unsqueeze(2)

    def backward(self, grad_output):
        return grad_output


# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()

        self.model_args = cfg.MODEL.TRANSFORMER
        self.seq_len = cfg.MODEL.MAX_LIMBS
        
        # Masking limbs
        self.masking = BinaryDifferentiableMasking(self.seq_len)
        
        # Embedding layer for per limb obs
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        self.limb_obs_embed = nn.Linear(limb_obs_size, self.d_model)
        self.ext_feat_fusion = self.model_args.EXT_MIX

        if self.model_args.POS_EMBEDDING == "learnt":
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.model_args.POS_EMBEDDING == "abs":
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)
        elif self.model_args.POS_EMBEDDING in ("topo", "topo_path"):
            topo_mode = "topo_depth" if self.model_args.POS_EMBEDDING == "topo" else "topo_path"
            self.pos_embedding = TopologyPositionalEncoding(
                self.d_model, cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_JOINTS,
                mode=topo_mode, dropout=self.model_args.DROPOUT,
            )

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        # Task based observation encoder
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])

        if self.ext_feat_fusion == "late":
            decoder_input_dim += self.hfield_encoder.obs_feat_dim

        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = tu.make_mlp_default(
            [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
            final_nonlinearity=False,
        )
        
        
        
        self.init_weights()

    def init_weights(self):
        initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
        self.limb_obs_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder[-1].bias.data.zero_()
        initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, return_attention=False, edges=None):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        obs_embed = self.limb_obs_embed(obs) * math.sqrt(self.d_model)
        # obs_embed = self.masking(obs_embed)

        _, batch_size, _ = obs_embed.shape

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["hfield"])

        if self.ext_feat_fusion in ["late"]:
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)

        attention_maps = None

        if self.model_args.POS_EMBEDDING in ["learnt", "abs"]:
            obs_embed = self.pos_embedding(obs_embed)
        elif self.model_args.POS_EMBEDDING in ["topo", "topo_path"]:
            obs_embed = self.pos_embedding(obs_embed, edges)
        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, src_key_padding_mask=obs_mask
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                obs_embed, src_key_padding_mask=obs_mask
            )

        decoder_input = obs_embed_t
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        # (num_limbs, batch_size, J)
        output = self.decoder(decoder_input)
        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        return output, attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class TopologyPositionalEncoding(nn.Module):
    """Topology-aware positional encoding derived from morphology tree structure.

    Instead of encoding limb position by array index, this encodes each limb's
    position in the morphology tree using the edges (parent-child pairs).

    Modes:
    - "topo_depth": learned embedding indexed by tree depth. Limbs at the same
      depth get identical PE regardless of which branch they're on.
    - "topo_path": per-depth-level embeddings summed along the root-to-limb path.
      More expressive -- distinguishes limbs on different branches.
    """

    def __init__(self, d_model, max_limbs, max_joints, mode="topo_depth", dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_limbs = max_limbs
        self.max_joints = max_joints
        self.mode = mode
        self.dropout = nn.Dropout(p=dropout)

        if mode == "topo_depth":
            self.depth_embed = nn.Embedding(max_limbs, d_model)
        elif mode == "topo_path":
            self.path_embed = nn.ModuleList([
                nn.Embedding(max_limbs, d_model) for _ in range(max_limbs)
            ])

    def _parse_edges(self, edges):
        """Parse flat edges tensor into a parent map.

        Args:
            edges: (batch_size, 2 * max_joints) float tensor.
                   Flattened [child_0, parent_0, child_1, parent_1, ...].
                   Padded pairs use max_limbs - 1.
        Returns:
            parent: (batch_size, max_limbs) long tensor where parent[b][i] is
                    the parent of limb i. Root and padded limbs map to themselves.
        """
        batch_size = edges.shape[0]
        edges_int = edges.long()
        edge_pairs = edges_int.view(batch_size, self.max_joints, 2)
        children = edge_pairs[:, :, 0]  # (B, max_joints)
        parents = edge_pairs[:, :, 1]   # (B, max_joints)

        # Default: each limb is its own parent (self-loop for root + padding)
        parent = torch.arange(self.max_limbs, device=edges.device) \
                      .unsqueeze(0).expand(batch_size, -1).clone()

        # Valid edges: not padding (padding value = max_limbs - 1)
        pad_val = self.max_limbs - 1
        valid = children != pad_val  # (B, max_joints)

        for j in range(self.max_joints):
            m = valid[:, j]
            if m.any():
                parent[m, children[m, j]] = parents[m, j]

        return parent

    def _compute_depths(self, parent):
        """Compute tree depth of each limb via iterative parent lookup.

        Args:
            parent: (batch_size, max_limbs) long tensor
        Returns:
            depths: (batch_size, max_limbs) long tensor
        """
        batch_size = parent.shape[0]
        device = parent.device
        depths = torch.zeros(batch_size, self.max_limbs, dtype=torch.long, device=device)
        arange = torch.arange(self.max_limbs, device=device).unsqueeze(0)
        is_root = (parent == arange)  # (B, L) -- True for root and padded (self-loop)

        # Iteratively propagate: depth[i] = depth[parent[i]] + 1, unless root
        for _ in range(self.max_limbs):
            parent_depths = depths.gather(1, parent)
            depths = torch.where(is_root, torch.zeros_like(depths), parent_depths + 1)

        return depths

    def _compute_child_order(self, parent):
        """Compute child order (sibling index) for each limb.

        For each limb, its child_order is its index among its parent's children,
        sorted by limb index. Since limbs are iterated in index order, the first
        child of a parent gets order 0, the second gets order 1, etc.

        Args:
            parent: (batch_size, max_limbs) long tensor
        Returns:
            child_order: (batch_size, max_limbs) long tensor
        """
        batch_size = parent.shape[0]
        device = parent.device
        child_order = torch.zeros(batch_size, self.max_limbs, dtype=torch.long, device=device)

        # For each limb i (in ascending order), count how many earlier limbs
        # share the same parent. This gives the sibling index.
        arange = torch.arange(self.max_limbs, device=device).unsqueeze(0)
        is_root = (parent == arange)  # (B, L)

        for i in range(1, self.max_limbs):
            if is_root[:, i].all():
                continue
            # Count earlier siblings: limbs j < i with same parent
            same_parent = (parent[:, :i] == parent[:, i:i+1])  # (B, i)
            child_order[:, i] = same_parent.sum(dim=1)

        return child_order

    def forward(self, x, edges):
        """Apply topology-aware positional encoding.

        Args:
            x: (seq_len, batch_size, d_model) -- limb embeddings
            edges: (batch_size, 2 * max_joints) -- edge tensor from observations
        Returns:
            x: (seq_len, batch_size, d_model) -- with positional encoding added
        """
        parent = self._parse_edges(edges)  # (B, L)

        if self.mode == "topo_depth":
            depths = self._compute_depths(parent)  # (B, L)
            pe = self.depth_embed(depths)  # (B, L, d_model)
            pe = pe.permute(1, 0, 2)  # (L, B, d_model)
            x = x + pe

        elif self.mode == "topo_path":
            depths = self._compute_depths(parent)  # (B, L)
            child_order = self._compute_child_order(parent)  # (B, L)

            batch_size = x.shape[1]
            device = x.device

            # Build ancestor chain: ancestors_stack[step, b, i] = ancestor of
            # limb i at `step` hops up. Step 0 = self, step 1 = parent, etc.
            ancestors = []
            current = torch.arange(self.max_limbs, device=device) \
                           .unsqueeze(0).expand(batch_size, -1).clone()
            for _ in range(self.max_limbs):
                ancestors.append(current.clone())
                current = parent.gather(1, current)
            ancestors_stack = torch.stack(ancestors, dim=0)  # (max_limbs, B, L)

            # Precompute batch and limb index grids for advanced indexing
            b_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.max_limbs)
            l_idx = torch.arange(self.max_limbs, device=device).unsqueeze(0).expand(batch_size, -1)

            # For each path position p (0=root, ..., d=limb), look up the
            # child_order of the ancestor at that level and embed it.
            pe = torch.zeros_like(x)  # (L, B, d_model)
            for p in range(self.max_limbs):
                # For a limb at depth d, path position p uses ancestor at (d-p) hops
                hops_up = (depths - p).clamp(min=0, max=self.max_limbs - 1)  # (B, L)
                # Vectorized gather: ancestor index at that hop distance
                anc_idx = ancestors_stack[hops_up, b_idx, l_idx]  # (B, L)
                # Get child_order of that ancestor
                anc_child_order = child_order.gather(1, anc_idx)  # (B, L)
                emb = self.path_embed[p](anc_child_order)  # (B, L, d_model)
                # Only add for path positions that exist (p <= depth)
                valid = (p <= depths).unsqueeze(-1).float()  # (B, L, 1)
                pe = pe + (emb * valid).permute(1, 0, 2)

            x = x + pe

        return self.dropout(x)


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        self.seq_len = cfg.MODEL.MAX_LIMBS
        self.v_net = TransformerModel(obs_space, 1)

        if cfg.ENV_NAME == "Unimal-v0":
            self.mu_net = TransformerModel(obs_space, 2)
            self.num_actions = cfg.MODEL.MAX_LIMBS * 2
        else:
            raise ValueError("Unsupported ENV_NAME")

        if cfg.MODEL.ACTION_STD_FIXED:
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False,
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))

    def forward(self, obs, act=None, num_samples=-1, return_attention=False):
        if num_samples > 0:
            batch_size = num_samples
        elif act is not None:
            batch_size = cfg.PPO.BATCH_SIZE
        else:
            batch_size = cfg.PPO.NUM_ENVS

        obs_env = {k: obs[k] for k in cfg.ENV.KEYS_TO_KEEP}
        if "obs_padding_cm_mask" in obs:
            obs_cm_mask = obs["obs_padding_cm_mask"]
        else:
            obs_cm_mask = None
        obs, obs_mask, act_mask, edges = (
            obs["proprioceptive"],
            obs["obs_padding_mask"],
            obs["act_padding_mask"],
            obs["edges"],
        )

        obs_mask = obs_mask.bool()
        act_mask = act_mask.bool()
        
        if obs.shape[0] == 1:
            batch_size = 1

        obs = obs.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
        # Per limb critic values
        limb_vals, v_attention_maps = self.v_net(
            obs, obs_mask, obs_env, obs_cm_mask, return_attention=return_attention, edges=edges
        )
        # Zero out mask values
        limb_vals = limb_vals * (1 - obs_mask.int())
        # Use avg/max to keep the magnitidue same instead of sum
        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)

        mu, mu_attention_maps = self.mu_net(
            obs, obs_mask, obs_env, obs_cm_mask, return_attention=return_attention, edges=edges
        )
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
                return val, pi, v_attention_maps, mu_attention_maps
            else:
                return val, pi, None, None


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs, num_samples=-1):
        val, pi, _, _ = self.ac(obs, num_samples=num_samples)
        act = pi.sample()
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, _, _, _ = self.ac(obs)
        return val
