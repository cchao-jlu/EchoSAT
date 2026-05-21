import torch
from torch_geometric.data import HeteroData

from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform


class AddNodeFeatures(BaseTransform):

    def __init__(self, feature_set: str = "legacy"):
        super(AddNodeFeatures, self).__init__()
        if feature_set not in {"legacy", "rich"}:
            raise ValueError(f"Unknown feature set {feature_set}")
        self.feature_set = feature_set

    def lit_dim(self) -> int:
        if self.feature_set == "legacy":
            return 1
        return 5

    def cls_dim(self) -> int:
        if self.feature_set == "legacy":
            return 1
        return 3

    def forward(self, data: HeteroData) -> HeteroData:
        edge_index = data["cls", "lit"].edge_index
        num_cls, num_lit = data["cls"].num_nodes, data["lit"].num_nodes

        cls_deg = degree(edge_index[0], num_cls, dtype=torch.float32).unsqueeze(1)
        lit_deg = degree(edge_index[1], num_lit, dtype=torch.float32).unsqueeze(1)

        if ("lit", "to", "lit") in data.edge_types and data["lit", "lit"].edge_index is not None:
            lit_deg += degree(data["lit", "lit"].edge_index[1], num_nodes=num_lit).unsqueeze(1)

        if self.feature_set == "legacy":
            x_cls = torch.log1p(cls_deg)
            x_lit = torch.log1p(lit_deg)
        else:
            pos_deg = lit_deg[1::2]
            neg_deg = lit_deg[0::2]
            var_deg = pos_deg + neg_deg

            var_deg_lit = torch.repeat_interleave(var_deg, 2, dim=0)
            balance = (pos_deg - neg_deg) / var_deg.clamp_min(1.0)
            balance_lit = torch.repeat_interleave(balance, 2, dim=0)

            sign = torch.ones((num_lit, 1), dtype=torch.float32, device=lit_deg.device)
            sign[0::2] = -1.0
            lit_share = lit_deg / var_deg_lit.clamp_min(1.0)

            x_cls = torch.cat(
                [
                    torch.log1p(cls_deg),
                    cls_deg,
                    1.0 / cls_deg.clamp_min(1.0),
                ],
                dim=1,
            )
            x_lit = torch.cat(
                [
                    torch.log1p(lit_deg),
                    torch.log1p(var_deg_lit),
                    lit_share,
                    balance_lit,
                    sign,
                ],
                dim=1,
            )

        data["cls"].x = x_cls
        data["lit"].x = x_lit
        return data
