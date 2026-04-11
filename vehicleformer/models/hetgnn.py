"""
Heterogeneous Graph Neural Network (HetGNN) Encoder — Contribution 1
=====================================================================
Represents the vehicle-road-cloud system as a typed heterogeneous graph:

  Node types : vehicle, rsu (roadside unit), base_station, satellite
  Edge types : v2v (vehicle↔vehicle)
               v2i (vehicle↔rsu)
               v2n (vehicle↔base_station)
               v2s (vehicle↔satellite)

This is the first unified graph representation of all ICV network
entities with physics-informed typed edges. Prior work uses flat
state vectors; our graph captures structural relationships.

Target journal: IEEE Transactions on Intelligent Transportation Systems
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional
import numpy as np


# ─── Multi-Head Attention Message Passing ────────────────────────────────
class HeteroAttentionConv(nn.Module):
    """
    Single heterogeneous graph convolution layer.
    Implements typed message passing with multi-head attention.
    
    For each edge type (src_type, rel, dst_type):
      m_ij = W_rel * [h_i || h_j || e_ij]
      alpha_ij = softmax( a_rel^T * m_ij )
      h_i' = sigma( sum_j alpha_ij * m_ij )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_types: list,   # list of (src_type, rel, dst_type)
        num_heads: int = 8,
        edge_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert out_dim % num_heads == 0
        self.out_dim    = out_dim
        self.num_heads  = num_heads
        self.head_dim   = out_dim // num_heads
        self.edge_types = edge_types

        # Per-relation transformation matrices
        self.W_msg = nn.ModuleDict({
            f"{src}_{rel}_{dst}": nn.Linear(in_dim * 2 + edge_dim, out_dim, bias=False)
            for src, rel, dst in edge_types
        })
        # Per-relation attention vectors
        self.a_att = nn.ParameterDict({
            f"{src}_{rel}_{dst}": nn.Parameter(torch.empty(num_heads, self.head_dim))
            for src, rel, dst in edge_types
        })
        # Output projection per node type (collect from all incoming edge types)
        node_types = set()
        for src, rel, dst in edge_types:
            node_types.add(src)
            node_types.add(dst)
        self.W_out = nn.ModuleDict({
            nt: nn.Linear(out_dim, out_dim)
            for nt in node_types
        })

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.ModuleDict({
            nt: nn.LayerNorm(out_dim)
            for nt in node_types
        })

        # Initialize attention vectors
        for key in self.a_att:
            nn.init.xavier_uniform_(self.a_att[key].unsqueeze(0))

    def forward(
        self,
        x: Dict[str, Tensor],            # node features per type
        edge_index: Dict[str, Tensor],   # edge indices per relation
        edge_attr: Dict[str, Tensor],    # edge features per relation
    ) -> Dict[str, Tensor]:
        """
        Args:
            x:          {node_type: (N_type, in_dim)}
            edge_index: {rel_key:   (2, E_rel)}  [src_idx, dst_idx]
            edge_attr:  {rel_key:   (E_rel, edge_dim)}
            
        Returns:
            {node_type: (N_type, out_dim)}
        """
        # Accumulate messages per destination node type
        aggregated: Dict[str, list] = {nt: [] for nt in x}

        for src_type, rel, dst_type in self.edge_types:
            key = f"{src_type}_{rel}_{dst_type}"

            if key not in edge_index or edge_index[key].shape[1] == 0:
                continue

            ei   = edge_index[key]   # (2, E)
            ea   = edge_attr[key]    # (E, edge_dim)
            src_idx = ei[0]
            dst_idx = ei[1]

            h_src = x[src_type][src_idx]    # (E, in_dim)
            h_dst = x[dst_type][dst_idx]    # (E, in_dim)

            # Message: concatenate source, destination, edge features
            msg_input = torch.cat([h_src, h_dst, ea], dim=-1)  # (E, 2*in+edge_dim)
            msg = self.W_msg[key](msg_input)                    # (E, out_dim)

            # Multi-head attention score
            msg_heads = msg.view(-1, self.num_heads, self.head_dim)  # (E, H, D/H)
            a = self.a_att[key]                                       # (H, D/H)
            score = (msg_heads * a.unsqueeze(0)).sum(-1)              # (E, H)
            score = F.leaky_relu(score, 0.2)

            # Softmax over neighbors (scatter)
            N_dst = x[dst_type].shape[0]
            alpha = self._scatter_softmax(score, dst_idx, N_dst)      # (E, H)
            alpha = self.dropout(alpha)

            # Weighted sum → (E, H, D/H)
            weighted = (msg_heads * alpha.unsqueeze(-1))
            # Scatter sum to dst nodes
            out = torch.zeros(N_dst, self.num_heads, self.head_dim,
                              device=msg.device)
            out.scatter_add_(0,
                dst_idx.unsqueeze(-1).unsqueeze(-1)
                    .expand(-1, self.num_heads, self.head_dim),
                weighted)
            out = out.view(N_dst, self.out_dim)                       # (N_dst, out_dim)
            aggregated[dst_type].append(out)

        # Aggregate messages from all incoming relations + residual
        result = {}
        for nt in x:
            msgs = aggregated[nt]
            if msgs:
                h_new = torch.stack(msgs, dim=0).mean(0)              # mean pooling over relations
            else:
                h_new = torch.zeros_like(x[nt][:, :self.out_dim])

            # Output projection + residual (if dims match)
            h_new = self.W_out[nt](h_new)
            if x[nt].shape[-1] == self.out_dim:
                h_new = h_new + x[nt]                                 # residual
            result[nt] = self.layer_norm[nt](F.gelu(h_new))

        return result

    @staticmethod
    def _scatter_softmax(src: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Softmax over scattered neighborhoods."""
        max_val = torch.zeros(num_nodes, src.shape[-1], device=src.device)
        max_val.scatter_reduce_(0,
            index.unsqueeze(-1).expand_as(src), src, reduce="amax", include_self=True)
        src_exp = torch.exp(src - max_val[index])
        sum_exp = torch.zeros_like(max_val)
        sum_exp.scatter_add_(0, index.unsqueeze(-1).expand_as(src_exp), src_exp)
        return src_exp / (sum_exp[index] + 1e-8)


# ─── Full HetGNN Encoder ─────────────────────────────────────────────────
class HetGNNEncoder(nn.Module):
    """
    Multi-layer Heterogeneous GNN producing embeddings for all node types.
    
    Architecture:
        Input projections (per node type)
        → L × HeteroAttentionConv layers
        → Global graph embedding (mean pool over vehicle nodes)
        → Final projection to embedding_dim
    """

    def __init__(self, cfg: dict):
        super().__init__()
        hcfg    = cfg['hetgnn']
        obscfg  = cfg['observation']

        self.hidden_dim    = hcfg['hidden_dim']
        self.embedding_dim = hcfg['embedding_dim']
        self.num_layers    = hcfg['num_layers']
        self.edge_dim      = hcfg['edge_dim']
        self.use_causal_mask = hcfg.get('use_causal_mask', True)

        edge_types = [(e[0], e[1], e[2]) for e in hcfg['edge_types']]

        # ─── Input projections (raw features → hidden_dim) ───────────
        self.input_proj = nn.ModuleDict({
            "vehicle"      : nn.Linear(obscfg['vehicle_dim'], self.hidden_dim),
            "rsu"          : nn.Linear(obscfg['rsu_dim'],     self.hidden_dim),
            "base_station" : nn.Linear(obscfg['bs_dim'],      self.hidden_dim),
            "satellite"    : nn.Linear(obscfg.get('satellite_dim', 4), self.hidden_dim),
        })

        # ─── GNN layers ───────────────────────────────────────────────
        self.conv_layers = nn.ModuleList([
            HeteroAttentionConv(
                in_dim     = self.hidden_dim,
                out_dim    = self.hidden_dim,
                edge_types = edge_types,
                num_heads  = hcfg['num_heads'],
                edge_dim   = self.edge_dim,
                dropout    = hcfg['dropout'],
            )
            for _ in range(self.num_layers)
        ])

        # ─── Output projection ────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=hcfg['num_heads'],
            dropout=hcfg['dropout'],
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(self.embedding_dim)

        # ─── Edge feature encoder ─────────────────────────────────────
        # Encodes raw edge features (distance, RSSI, etc.) → edge_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, self.edge_dim),
            nn.ReLU(),
            nn.Linear(self.edge_dim, self.edge_dim),
        )

    def forward(
        self,
        node_features: Dict[str, Tensor],
        positions: Dict[str, Tensor],
        timestamps: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            node_features: {
                "vehicle":       (N_v, vehicle_dim),
                "rsu":           (N_r, rsu_dim),
                "base_station":  (N_b, bs_dim),
            }
            positions: {
                "vehicle":       (N_v, 2),
                "rsu":           (N_r, 2),
                "base_station":  (N_b, 2),
            }
            
        Returns:
            graph_embedding:  (embedding_dim,)  — global graph state
            node_embeddings:  {node_type: (N, embedding_dim)}
        """
        # ─── Input projection ─────────────────────────────────────────
        h = {
            nt: F.gelu(self.input_proj[nt](feat))
            for nt, feat in node_features.items()
            if nt in self.input_proj
        }

        # ─── Build edges from positions ───────────────────────────────
        edge_index, edge_attr = self._build_edges(positions)

        # ─── Message passing ──────────────────────────────────────────
        for layer in self.conv_layers:
            h = layer(h, edge_index, edge_attr)

        # ─── Output projections ───────────────────────────────────────
        node_embeddings = {nt: self.output_proj(feat) for nt, feat in h.items()}
        if self.use_causal_mask and timestamps is not None and "vehicle" in node_embeddings and "vehicle" in timestamps:
            node_embeddings["vehicle"] = self._apply_causal_temporal_mask(
                node_embeddings["vehicle"], timestamps["vehicle"]
            )

        # ─── Global graph embedding (mean pool vehicles) ──────────────
        graph_embedding = node_embeddings["vehicle"].mean(dim=0)

        return graph_embedding, node_embeddings

    def _apply_causal_temporal_mask(self, features: Tensor, timestamps: Tensor) -> Tensor:
        """Apply causal temporal attention so node embeddings attend only to past steps."""
        if features.ndim != 2 or timestamps.ndim != 1 or features.shape[0] <= 1:
            return features
        order = torch.argsort(timestamps)
        reverse = torch.argsort(order)
        ordered = features[order].unsqueeze(0)
        steps = ordered.shape[1]
        attn_mask = torch.triu(
            torch.ones(steps, steps, device=features.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.temporal_attention(
            ordered,
            ordered,
            ordered,
            attn_mask=attn_mask,
            need_weights=False,
        )
        attended = self.temporal_norm(attended + ordered)
        return attended.squeeze(0)[reverse]

    def _build_edges(
        self,
        positions: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Dynamically build edges based on proximity and network range.
        
        Edge features: [normalized_distance, rssi_approx, link_type_oh(2), direction]
        """
        device = next(iter(positions.values())).device
        edge_index = {}
        edge_attr  = {}

        v_pos  = positions.get("vehicle",       torch.zeros(0, 2, device=device))
        r_pos  = positions.get("rsu",           torch.zeros(0, 2, device=device))
        bs_pos = positions.get("base_station",  torch.zeros(0, 2, device=device))

        # ─── V2V: within 300m ────────────────────────────────────────
        if v_pos.shape[0] > 1:
            ei, ea = self._proximity_edges(v_pos, v_pos, max_dist=300.0, link_type=0)
            edge_index["vehicle_v2v_vehicle"] = ei
            edge_attr["vehicle_v2v_vehicle"]  = ea

        # ─── V2I: vehicle ↔ RSU within 300m ─────────────────────────
        if v_pos.shape[0] > 0 and r_pos.shape[0] > 0:
            ei, ea = self._proximity_edges(v_pos, r_pos, max_dist=300.0, link_type=1)
            edge_index["vehicle_v2i_rsu"] = ei
            edge_attr["vehicle_v2i_rsu"]  = ea

        # ─── V2N: vehicle ↔ base station within 2km ──────────────────
        if v_pos.shape[0] > 0 and bs_pos.shape[0] > 0:
            ei, ea = self._proximity_edges(v_pos, bs_pos, max_dist=2000.0, link_type=2)
            edge_index["vehicle_v2n_base_station"] = ei
            edge_attr["vehicle_v2n_base_station"]  = ea

        # ─── V2S: all vehicles → satellite (always connected if coverage)
        if v_pos.shape[0] > 0:
            N = v_pos.shape[0]
            src = torch.arange(N, device=device)
            dst = torch.zeros(N, dtype=torch.long, device=device)
            ei = torch.stack([src, dst])
            raw_ea = torch.zeros(N, 4, device=device)
            raw_ea[:, 0] = 550000.0 / 5000.0   # normalized altitude
            raw_ea[:, 3] = 1.0                  # link_type = satellite
            edge_index["vehicle_v2s_satellite"] = ei
            edge_attr["vehicle_v2s_satellite"]  = self.edge_encoder(raw_ea)

        return edge_index, edge_attr

    def _proximity_edges(
        self,
        src_pos: Tensor,
        dst_pos: Tensor,
        max_dist: float,
        link_type: int,
    ) -> Tuple[Tensor, Tensor]:
        """Create edges between all src-dst pairs within max_dist."""
        device  = src_pos.device
        N_src   = src_pos.shape[0]
        N_dst   = dst_pos.shape[0]

        # Pairwise distances
        diff = src_pos.unsqueeze(1) - dst_pos.unsqueeze(0)  # (N_s, N_d, 2)
        dist = diff.norm(dim=-1)                             # (N_s, N_d)

        # Find pairs within range (exclude self-loops for V2V)
        if N_src == N_dst and torch.allclose(src_pos, dst_pos):
            mask = (dist < max_dist) & (dist > 0.1)
        else:
            mask = dist < max_dist

        src_idx, dst_idx = mask.nonzero(as_tuple=True)

        if src_idx.shape[0] == 0:
            empty_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            empty_ea = torch.zeros(0, self.edge_dim, device=device)
            return empty_ei, empty_ea

        # Edge raw features: [dist_norm, rssi_approx, sin_angle, cos_angle]
        d_vals   = dist[src_idx, dst_idx] / max_dist              # [0,1]
        rssi_app = torch.clamp(1.0 - d_vals, 0, 1)               # proxy RSSI
        angles   = diff[src_idx, dst_idx]
        angles   = angles / (angles.norm(dim=-1, keepdim=True) + 1e-8)

        raw_ea = torch.stack([
            d_vals,
            rssi_app,
            angles[:, 0],
            angles[:, 1],
        ], dim=-1)  # (E, 4)

        ea = self.edge_encoder(raw_ea)  # (E, edge_dim)
        ei = torch.stack([src_idx, dst_idx])

        return ei, ea


# ─── Graph Observation Builder ───────────────────────────────────────────
class GraphBuilder:
    """
    Converts flat environment observation into graph-structured tensors
    suitable for HetGNNEncoder input.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.device = device
        obs_cfg     = cfg['observation']
        self.v_dim  = obs_cfg['vehicle_dim']
        self.r_dim  = obs_cfg['rsu_dim']
        self.b_dim  = obs_cfg['bs_dim']
        self.max_v  = obs_cfg['max_vehicles']
        self.max_r  = obs_cfg['max_rsus']
        self.max_b  = obs_cfg['max_base_stations']

        # Fixed infrastructure positions
        self.rsu_pos = torch.tensor(
            cfg['simulation']['rsu_positions'], dtype=torch.float32, device=device
        ) / 800.0

        self.bs_pos = torch.tensor(
            cfg['simulation']['base_stations_5g'], dtype=torch.float32, device=device
        ) / 800.0

    def build(self, obs: np.ndarray) -> Dict[str, Dict[str, Tensor]]:
        """
        Convert flat obs → graph dicts for HetGNNEncoder.
        
        Args:
            obs: flat observation vector (obs_dim,)
            
        Returns:
            {
              "node_features": {node_type: Tensor},
              "positions":     {node_type: Tensor},
            }
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # Slice out vehicle features
        v_end = self.max_v * self.v_dim
        r_end = v_end + self.max_r * self.r_dim
        b_end = r_end + self.max_b * self.b_dim

        v_feat = obs_t[:v_end].view(self.max_v, self.v_dim)
        r_feat = obs_t[v_end:r_end].view(self.max_r, self.r_dim)
        b_feat = obs_t[r_end:b_end].view(self.max_b, self.b_dim)

        # Vehicle positions (first 2 dims are x,y normalized)
        v_pos = v_feat[:, :2].clone()

        # Satellite: single node with fixed features [altitude_norm, latency_norm, coverage, bandwidth]
        sat_feat = torch.tensor([[0.69, 0.7, 1.0, 0.3]], dtype=torch.float32, device=self.device)
        sat_pos  = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=self.device)

        return {
            "node_features": {
                "vehicle"      : v_feat,
                "rsu"          : r_feat,
                "base_station" : b_feat,
                "satellite"    : sat_feat,
            },
            "positions": {
                "vehicle"      : v_pos,
                "rsu"          : self.rsu_pos[:self.max_r],
                "base_station" : self.bs_pos[:self.max_b],
                "satellite"    : sat_pos,
            },
        }

    def build_batch(self, obs_batch: np.ndarray) -> Dict[str, Dict[str, Tensor]]:
        """Build batched graph for training. obs_batch: (B, obs_dim)"""
        B = obs_batch.shape[0]
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

        v_end = self.max_v * self.v_dim
        r_end = v_end + self.max_r * self.r_dim
        b_end = r_end + self.max_b * self.b_dim

        v_feat = obs_t[:, :v_end].view(B * self.max_v, self.v_dim)
        r_feat = obs_t[:, v_end:r_end].view(B * self.max_r, self.r_dim)
        b_feat = obs_t[:, r_end:b_end].view(B * self.max_b, self.b_dim)

        v_pos = v_feat[:, :2].clone()

        return {
            "node_features": {
                "vehicle"      : v_feat,
                "rsu"          : r_feat,
                "base_station" : b_feat,
            },
            "positions": {
                "vehicle"      : v_pos,
                "rsu"          : self.rsu_pos[:self.max_r].repeat(B, 1),
                "base_station" : self.bs_pos[:self.max_b].repeat(B, 1),
            },
        }
