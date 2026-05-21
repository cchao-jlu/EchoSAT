import os

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from omegaconf import DictConfig, OmegaConf

from src.data.transform import AddNodeFeatures
from src.model.modules import GNNLayer, FeatureEncoder, SinusoidalNumericalEncoder


class GNN(nn.Module):

    def __init__(
            self,
            channels: int,
            lit_feat_dim: int,
            cls_feat_dim: int,
            num_layers: int,
            out_dim: int = 2,
            global_state_dim: int = 0,
            var_state_dim: int = 0,
            aggr: str | list[str] = "mean",
            feature_encoder: str = "mlp",
            dropout: float = 0.0,
            var_output: bool = True,
            separate_encoders: bool = False,
            event_adapter_enabled: bool = False,
            event_adapter_hidden_dim: int | None = None,
    ):
        """
        A message-passing Graph Neural Network
        :param channels: Hidden model dimension
        :param lit_feat_dim: Dimension of literal node features
        :param cls_feat_dim: Dimension of clause node features
        :param num_layers: Number of message passing layers
        :param out_dim: node-level output dimension
        :param global_state_dim: Optional graph-level feedback state dimension.
        :param var_state_dim: Optional variable-level solver event state dimension.
        :param aggr: Message aggregation function either mean, max, or sum. If a list is provided, than multiple types of aggregation are performed in parallel.
        :param feature_encoder: Type of node feature encoder. Either "mlp" for a simple perceptron or "sin" for a sinusoidal numerical encoder.
        :param dropout: Dropout probability
        :param var_output: If true, the output will be per variable. If false, the output will be per literal, which is useful for our supervised tasks like backbone prediction.
        """
        super(GNN, self).__init__()
        self.channels = channels
        self.separate_encoders = separate_encoders
        self.global_state_dim = global_state_dim
        self.var_state_dim = var_state_dim
        self.event_adapter_enabled = event_adapter_enabled

        if feature_encoder not in {"mlp", "sin"}:
            raise ValueError(f"Unknown feature encoder type {feature_encoder}")
        encoder_cls = FeatureEncoder if feature_encoder == "mlp" else SinusoidalNumericalEncoder

        if not separate_encoders and lit_feat_dim != cls_feat_dim:
            raise ValueError("Shared literal/clause encoder requires matching feature dimensions")

        self.lit_enc = encoder_cls(
            channels_in=lit_feat_dim,
            channels_out=channels,
            dropout=dropout,
        )
        if separate_encoders:
            self.cls_enc = encoder_cls(
                channels_in=cls_feat_dim,
                channels_out=channels,
                dropout=dropout,
            )
        else:
            self.cls_enc = self.lit_enc

        self.layers = nn.ModuleList([
            GNNLayer(channels=channels, aggr=aggr, dropout=dropout) for _ in range(num_layers)
        ])

        self.var_output = var_output
        if self.var_output:
            # output mlp with last layer initialized with zeros
            self.out_lin1 = nn.Linear(2 * channels + global_state_dim + var_state_dim, 2 * channels)
            self.out_lin2 = nn.Linear(2 * channels, out_dim)
            nn.init.zeros_(self.out_lin2.weight)
            self.out_act = nn.SiLU(inplace=True)
            if self.event_adapter_enabled:
                adapter_hidden_dim = event_adapter_hidden_dim or 2 * channels
                self.event_adapter = nn.Sequential(
                    nn.Linear(2 * channels + var_state_dim + global_state_dim, adapter_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_hidden_dim, out_dim),
                )
                nn.init.zeros_(self.event_adapter[-1].weight)
                nn.init.zeros_(self.event_adapter[-1].bias)
            else:
                self.event_adapter = None
        else:
            # output mlp with last layer initialized with zeros
            self.out_lin1 = nn.Linear(channels + global_state_dim, 2 * channels)
            self.out_lin2 = nn.Linear(2 * channels, out_dim)
            self.out_act = nn.SiLU(inplace=True)
            self.event_adapter = None

    def _get_global_state(
        self,
        data: HeteroData,
        num_graphs: int,
    ) -> Tensor | None:
        if self.global_state_dim <= 0:
            return None
        if hasattr(data, "global_state"):
            global_state = data.global_state
            if global_state.dim() == 1:
                if global_state.numel() == num_graphs * self.global_state_dim:
                    global_state = global_state.view(num_graphs, self.global_state_dim)
                elif global_state.numel() == self.global_state_dim:
                    global_state = global_state.unsqueeze(0)
                else:
                    raise ValueError(
                        f"Invalid global_state shape {tuple(global_state.shape)} for "
                        f"{num_graphs} graphs and global_state_dim={self.global_state_dim}"
                    )
            return global_state.to(dtype=torch.float32, device=data["lit"].x.device)
        return torch.zeros(
            (num_graphs, self.global_state_dim),
            dtype=torch.float32,
            device=data["lit"].x.device,
        )

    def _get_var_state(self, data: HeteroData, num_vars: int) -> Tensor | None:
        if self.var_state_dim <= 0:
            return None
        if "var" in data.node_types and hasattr(data["var"], "event_state"):
            var_state = data["var"].event_state
            if var_state.dim() == 1:
                var_state = var_state.view(num_vars, self.var_state_dim)
            return var_state.to(dtype=torch.float32, device=data["lit"].x.device)
        return torch.zeros(
            (num_vars, self.var_state_dim),
            dtype=torch.float32,
            device=data["lit"].x.device,
        )

    def _get_var_batch(self, data: HeteroData, num_vars: int, device: torch.device) -> Tensor:
        if "var" in data.node_types and hasattr(data["var"], "batch"):
            return data["var"].batch.to(device=device)
        if hasattr(data["lit"], "batch"):
            return data["lit"].batch[0::2].to(device=device)
        return torch.zeros(num_vars, dtype=torch.long, device=device)

    def _has_event_adapter_cache(self, data: HeteroData) -> bool:
        return (
            self.var_output
            and self.event_adapter_enabled
            and "var" in data.node_types
            and hasattr(data["var"], "base_embedding")
            and hasattr(data["var"], "base_y")
            and hasattr(data["var"], "event_state")
        )

    def _forward_event_adapter(self, data: HeteroData) -> Tensor:
        base_embedding = data["var"].base_embedding.to(dtype=torch.float32, device=data["var"].event_state.device)
        base_y = data["var"].base_y.to(dtype=torch.float32, device=base_embedding.device)
        num_vars = int(base_y.shape[0])
        var_state = data["var"].event_state.to(dtype=torch.float32, device=base_embedding.device)
        if var_state.dim() == 1:
            var_state = var_state.view(num_vars, self.var_state_dim)

        adapter_input = [base_embedding]
        if self.var_state_dim > 0:
            adapter_input.append(var_state)
        if self.global_state_dim > 0:
            var_batch = self._get_var_batch(data, num_vars=num_vars, device=base_embedding.device)
            num_graphs = int(var_batch.max().item()) + 1 if var_batch.numel() > 0 else 1
            global_state = self._get_global_state(data, num_graphs=num_graphs)
            adapter_input.append(global_state[var_batch])
        delta = self.event_adapter(torch.cat(adapter_input, dim=1))
        return base_y + delta

    def forward(self, data: HeteroData, return_cache: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if self._has_event_adapter_cache(data):
            y_var = self._forward_event_adapter(data)
            if return_cache:
                return y_var, {
                    "base_embedding": data["var"].base_embedding,
                    "base_y": data["var"].base_y,
                }
            return y_var

        x_lit = data["lit"].x
        h_lit = self.lit_enc(x_lit)

        x_cls = data["cls"].x
        h_cls = self.cls_enc(x_cls)

        for layer in self.layers:
            h_lit, h_cls = layer(h_lit, h_cls, data)

        lit_batch = data["lit"].batch
        num_graphs = int(lit_batch.max().item()) + 1 if lit_batch.numel() > 0 else 1
        global_state = self._get_global_state(data, num_graphs=num_graphs)

        if self.var_output:
            # concatenate interleaved embeddings and apply an mlp
            h_var = torch.cat([h_lit[0::2], h_lit[1::2]], dim=1)
            base_embedding = h_var
            var_state = self._get_var_state(data, num_vars=h_var.shape[0])
            if var_state is not None:
                h_var = torch.cat([h_var, var_state], dim=1)
            if global_state is not None:
                var_batch = lit_batch[0::2]
                h_var = torch.cat([h_var, global_state[var_batch]], dim=1)
            y_var = self.out_lin2(self.out_act(self.out_lin1(h_var)))
            if return_cache:
                return y_var, {
                    "base_embedding": base_embedding,
                    "base_y": y_var,
                }
            return y_var
        else:
            if global_state is not None:
                h_lit = torch.cat([h_lit, global_state[lit_batch]], dim=1)
            y_lit = self.out_lin2(self.out_act(self.out_lin1(h_lit)))
            return y_lit


def init_transform(cfg: DictConfig | None = None) -> AddNodeFeatures:
    feature_set = "legacy"
    if cfg is not None and "model" in cfg and "feature_set" in cfg.model:
        feature_set = cfg.model.feature_set
    return AddNodeFeatures(feature_set=feature_set)


def init_model(cfg: DictConfig, transform: AddNodeFeatures, **model_kwargs) -> GNN:
    var_output = model_kwargs.get("var_output", True)
    if "out_dim" not in model_kwargs:
        if var_output:
            learnable_sigma = bool(cfg.model.learnable_sigma) if "learnable_sigma" in cfg.model else False
            model_kwargs["out_dim"] = 3 if learnable_sigma else 2
        else:
            model_kwargs["out_dim"] = 1

    model = GNN(
        channels=cfg.model.channels,
        lit_feat_dim=transform.lit_dim(),
        cls_feat_dim=transform.cls_dim(),
        num_layers=cfg.model.num_layers,
        global_state_dim=int(cfg.model.global_state_dim) if "global_state_dim" in cfg.model else 0,
        var_state_dim=int(cfg.model.var_state_dim) if "var_state_dim" in cfg.model else 0,
        aggr=OmegaConf.to_container(cfg.model.aggr),
        feature_encoder=cfg.model.feature_encoder,
        dropout=cfg.model.dropout if "dropout" in cfg.model else 0.0,
        separate_encoders=bool(cfg.model.separate_encoders) if "separate_encoders" in cfg.model else False,
        event_adapter_enabled=(
            bool(cfg.model.event_adapter.enabled)
            if "event_adapter" in cfg.model and "enabled" in cfg.model.event_adapter
            else False
        ),
        event_adapter_hidden_dim=(
            int(cfg.model.event_adapter.hidden_dim)
            if "event_adapter" in cfg.model and "hidden_dim" in cfg.model.event_adapter
            else None
        ),
        **model_kwargs,
    )
    return model


def load_checkpoint(ckpt_path: str, **model_kwargs) -> tuple[GNN, AddNodeFeatures, DictConfig]:
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    transform = init_transform(cfg)

    model = init_model(cfg, transform, **model_kwargs)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model, transform, cfg
