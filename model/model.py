"""
model.py
========
GNN model architectures for oil-market geopolitical embedding.

Three variants, all sharing the same WeightedGCN + fusion backbone:
  - TransformerAwareGCN   : Full model  (WeightedGCN → GRU → Transformer)
  - RNNOnlyGCN            : Ablation #1 (WeightedGCN → GRU, no Transformer)
  - TransformerOnlyGCN    : Ablation #2 (WeightedGCN → Transformer, no GRU)

Usage
-----
from model import build_model

model = build_model(
    variant="full",          # "full" | "rnn_only" | "transformer_only"
    num_entities=...,
    spatial_dim=16,
    temporal_dim=32,
    decoder_dim=96,
    use_layernorm=True,      # Toggle LayerNorm in WeightedGCNLayer
    use_context_fusion=True  # Toggle daily_count fusion
)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared building block
# =============================================================================


class WeightedGCNLayer(nn.Module):
    """
    Weighted Graph Convolutional Layer.
    Processes a single relation type and uses continuous values
    (edge_weight) to determine message-passing strength.
    """

    def __init__(self, in_dim: int, out_dim: int, use_layernorm: bool = False):
        super().__init__()
        self.use_layernorm = use_layernorm
        
        # Only one relation type is modeled here,
        # so only one neighbor transformation matrix is needed.
        self.neighbor_weights = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.self_loop = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        # Initialize LayerNorm only if enabled
        if self.use_layernorm:
            self.norm = nn.LayerNorm(out_dim)
            
        nn.init.xavier_uniform_(self.neighbor_weights)
        nn.init.xavier_uniform_(self.self_loop)

    def forward(self, x, edge_index, edge_weight):
        # 1. Compute transformed self-node features (self-loop)
        out = torch.matmul(x, self.self_loop)
        
        # 2. Prepare transformed neighbor features
        neighbor_feat = torch.matmul(x, self.neighbor_weights)
        
        # 3. Perform weighted aggregation based on edge_weight
        src, dst = edge_index
        if len(src) > 0:
            # Multiply source features by edge intensity (edge_weight)
            # unsqueeze(-1) reshapes [num_edges] into [num_edges, 1]
            # to enable broadcasting in multiplication
            weighted_messages = neighbor_feat[src] * edge_weight.unsqueeze(-1)
            
            # Aggregate weighted messages into destination nodes
            out.index_add_(0, dst, weighted_messages)
            
        # 4. Optional LayerNorm
        if self.use_layernorm:
            out = self.norm(out)
            
        return F.relu(out)



# =============================================================================
# Abstract base
# =============================================================================


class BaseGCN(nn.Module):
    """
    Shared backbone: WeightedGCN → project → (optional) context_fusion →
    (subclass updates memory) → decoder.

    Subclasses must implement `_temporal_update(fused_feat, prev_memory)` which
    takes the daily feature and the previous memory and returns the new memory.
    """

    def __init__(self, num_entities: int,
                 spatial_dim: int, temporal_dim: int, decoder_dim: int,
                 use_layernorm: bool = False, use_context_fusion: bool = True):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.use_context_fusion = use_context_fusion

        self.gcn = WeightedGCNLayer(
            in_dim=temporal_dim,
            out_dim=spatial_dim,
            use_layernorm=use_layernorm
        )

        self.project = nn.Sequential(
            nn.Linear(spatial_dim, temporal_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Enable or disable the context fusion module based on configuration
        if self.use_context_fusion:
            self.context_fusion = nn.Sequential(
                nn.Linear(temporal_dim + 1, temporal_dim),
                nn.Tanh()
            )

        # Decoder takes [h_s, h_o] with total size temporal_dim * 2
        # and predicts a single scalar value
        self.decoder = nn.Sequential(
            nn.Linear(temporal_dim * 2, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 1)
        )

    def _temporal_update(self, fused_feat: torch.Tensor,
                         prev_memory: torch.Tensor) -> torch.Tensor:
        """To be implemented by each variant."""
        raise NotImplementedError

    def forward_memory_update(self, edge_index, edge_weight,
                              prev_memory: torch.Tensor,
                              daily_count: torch.Tensor) -> torch.Tensor:
        num_nodes = prev_memory.size(0)

        # Spatial aggregation
        spatial_feat = self.gcn(prev_memory, edge_index, edge_weight)
        projected_feat = self.project(spatial_feat)

        # Conditional branch: whether to apply context fusion
        if self.use_context_fusion:
            count_feat = daily_count.view(1, 1).expand(num_nodes, 1)
            fusion_input = torch.cat([projected_feat, count_feat], dim=1)
            fused_feat = self.context_fusion(fusion_input)
        else:
            # If fusion is disabled, directly pass the projected features
            fused_feat = projected_feat

        return self._temporal_update(fused_feat, prev_memory)



# =============================================================================
# Variant 1 – Full model  (GRU → Transformer)
# =============================================================================


class TransformerAwareGCN(BaseGCN):
    """
    Full model.
    Temporal path: GRUCell then TransformerEncoder with node identity residual.
    """

    def __init__(self, num_entities: int,
                 spatial_dim: int, temporal_dim: int, decoder_dim: int,
                 nhead: int = 4, num_layers: int = 1,
                 use_layernorm: bool = False, use_context_fusion: bool = True):
        super().__init__(num_entities, spatial_dim, temporal_dim, decoder_dim, 
                         use_layernorm, use_context_fusion)

        self.gru = nn.GRUCell(temporal_dim, temporal_dim)

        self.node_identity = nn.Embedding(num_entities, temporal_dim)
        nn.init.normal_(self.node_identity.weight, std=0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=nhead,
            dim_feedforward=temporal_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _temporal_update(self, fused_feat: torch.Tensor,
                         prev_memory: torch.Tensor) -> torch.Tensor:
        raw_memory = self.gru(fused_feat, prev_memory)
        transformer_input = (raw_memory + self.node_identity.weight).unsqueeze(0)
        return self.transformer(transformer_input).squeeze(0)



# =============================================================================
# Variant 2 – RNN only  (GRU, no Transformer)
# =============================================================================


class RNNOnlyGCN(BaseGCN):
    """
    Ablation: Transformer removed.
    Temporal path: GRUCell only.
    """

    def __init__(self, num_entities: int,
                 spatial_dim: int, temporal_dim: int, decoder_dim: int,
                 use_layernorm: bool = False, use_context_fusion: bool = True):
        super().__init__(num_entities, spatial_dim, temporal_dim, decoder_dim,
                         use_layernorm, use_context_fusion)
        self.gru = nn.GRUCell(temporal_dim, temporal_dim)

    def _temporal_update(self, fused_feat: torch.Tensor,
                         prev_memory: torch.Tensor) -> torch.Tensor:
        return self.gru(fused_feat, prev_memory)



# =============================================================================
# Variant 3 – Transformer only  (no GRU)
# =============================================================================


class TransformerOnlyGCN(BaseGCN):
    """
    Ablation: GRU removed, no naive RNN substitute.
    Temporal path:
      1. memory_mix linearly combines prev_memory and fused_feat → mixed
      2. TransformerEncoder with node identity residual refines mixed
    """

    def __init__(self, num_entities: int,
                 spatial_dim: int, temporal_dim: int, decoder_dim: int,
                 nhead: int = 4, num_layers: int = 1,
                 use_layernorm: bool = False, use_context_fusion: bool = True):
        super().__init__(num_entities, spatial_dim, temporal_dim, decoder_dim,
                         use_layernorm, use_context_fusion)

        self.memory_mix = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.Tanh()
        )

        self.node_identity = nn.Embedding(num_entities, temporal_dim)
        nn.init.normal_(self.node_identity.weight, std=0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=nhead,
            dim_feedforward=temporal_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _temporal_update(self, fused_feat: torch.Tensor,
                         prev_memory: torch.Tensor) -> torch.Tensor:
        mixed = self.memory_mix(torch.cat([prev_memory, fused_feat], dim=1))
        transformer_input = (mixed + self.node_identity.weight).unsqueeze(0)
        return self.transformer(transformer_input).squeeze(0)



# =============================================================================
# Factory
# =============================================================================


VARIANT_MAP = {
    "full":               TransformerAwareGCN,
    "rnn_only":           RNNOnlyGCN,
    "transformer_only":   TransformerOnlyGCN,
}


def build_model(variant: str,
                num_entities: int,
                spatial_dim: int = 16,
                temporal_dim: int = 32,
                decoder_dim: int = 96,
                nhead: int = 4,
                num_layers: int = 1,
                use_layernorm: bool = False,
                use_context_fusion: bool = True) -> BaseGCN:
    """
    Instantiate a GCN variant.

    Parameters
    ----------
    variant : "full" | "rnn_only" | "transformer_only"
    num_entities : from gdelt_mappings.pkl
    spatial_dim, temporal_dim, decoder_dim : architecture dimensions
    nhead, num_layers : Transformer hyper-parameters (ignored for rnn_only)
    use_layernorm : whether to use LayerNorm in WeightedGCNLayer
    use_context_fusion : whether to use count_feat fusion in BaseGCN

    Returns
    -------
    nn.Module (BaseGCN subclass)
    """
    if variant not in VARIANT_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANT_MAP)}")

    cls = VARIANT_MAP[variant]

    kwargs = dict(
        num_entities=num_entities,
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        decoder_dim=decoder_dim,
        use_layernorm=use_layernorm,
        use_context_fusion=use_context_fusion
    )

    if variant != "rnn_only":
        kwargs["nhead"] = nhead
        kwargs["num_layers"] = num_layers

    return cls(**kwargs)