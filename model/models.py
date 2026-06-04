"""Local PyG model definitions extracted from the EPIC_Geo_Attn Chemprop fork."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Size
from torch_geometric.utils import to_dense_batch

from .config import TrainConfig
from .graph_conversion import ATOM_FDIM, BOND_FDIM


def get_activation_function(activation: str) -> nn.Module:
    if activation == "ReLU":
        return nn.ReLU()
    if activation == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    if activation == "PReLU":
        return nn.PReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "SELU":
        return nn.SELU()
    if activation == "ELU":
        return nn.ELU()
    raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Module) -> None:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.kaiming_normal_(param)


def get_unit_sequence(input_dim: int, output_dim: int, n_hidden: int) -> List[int]:
    reverse = False
    if input_dim > output_dim:
        reverse = True
        input_dim, output_dim = output_dim, input_dim

    diff = abs(output_dim.bit_length() - input_dim.bit_length())
    increment = diff // (n_hidden + 1)
    sequence = [input_dim] + [0] * n_hidden + [output_dim]

    for idx in range(n_hidden // 2):
        sequence[idx + 1] = 2 ** ((sequence[idx]).bit_length() + increment - 1)
        sequence[-2 - idx] = 2 ** ((sequence[-1 - idx] - 1).bit_length() - increment)

    if n_hidden % 2 == 1:
        sequence[n_hidden // 2 + 1] = (
            sequence[n_hidden // 2] + sequence[n_hidden // 2 + 2]
        ) // 2

    if reverse:
        sequence.reverse()
    return sequence


class DenseLayer(nn.Module):
    def __init__(self, size_in: int, size_out: int, activation: Optional[nn.Module]):
        super().__init__()
        self.linear = nn.Linear(size_in, size_out)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DenseFFN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, capacity: int, activation: nn.Module):
        super().__init__()
        unit_sequence = get_unit_sequence(input_dim, output_dim, capacity)
        layers = []
        for idx, n_units in enumerate(unit_sequence[:-1]):
            size_out = unit_sequence[idx + 1]
            layer_activation = None if idx == len(unit_sequence) - 2 else activation
            layers.append(DenseLayer(n_units, size_out, layer_activation))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RBF(nn.Module):
    def __init__(self, centers: Tensor, gamma: Tensor):
        super().__init__()
        self.register_buffer("centers", centers.view(1, -1))
        self.register_buffer("gamma", gamma)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class BondLengthRBF(nn.Module):
    def __init__(self, device: Union[str, torch.device]):
        super().__init__()
        centers = torch.arange(0, 2, 0.1, dtype=torch.float32, device=device)
        gamma = torch.tensor(10.0, dtype=torch.float32, device=device)
        self.rbf = RBF(centers, gamma)

    def forward(self, bond_length: Tensor) -> Tensor:
        return self.rbf(bond_length)


class KANLinear(nn.Module):
    """KAN layer used when ``TrainConfig.use_ffkn`` is enabled."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: Callable = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: Tensor) -> Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - self.grid[:, : -(k + 1)])
                / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (self.grid[:, k + 1 :] - x)
                / (self.grid[:, k + 1 :] - self.grid[:, 1:-k])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: Tensor, y: Tensor) -> Tensor:
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)


class FFKN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, capacity: int, activation: nn.Module):
        super().__init__()
        unit_sequence = get_unit_sequence(input_dim, output_dim, capacity)
        self.layers = nn.ModuleList(
            [KANLinear(unit_sequence[idx], unit_sequence[idx + 1]) for idx in range(len(unit_sequence) - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MultiBondFastAttention(nn.Module):
    """Bond-level Fastformer block from the original model."""

    def __init__(self, args: TrainConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.act_func = get_activation_function(args.activation)
        self.num_heads = args.num_heads
        self.att_size = self.hidden_size // self.num_heads
        self.scale_factor = self.att_size**-0.5

        self.weight_alpha = Parameter(torch.randn(self.att_size))
        self.weight_beta = Parameter(torch.randn(self.att_size))
        self.weight_r = nn.Linear(self.att_size, self.att_size, bias=False)
        self.W_b_q = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_k = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_v = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_o = nn.Linear(self.num_heads * self.att_size, self.hidden_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

    def forward(
        self,
        message: Tensor,
        b_scope_starts: Tensor,
        num_bonds_per_graph: Tensor,
    ) -> Tensor:
        if message.numel() == 0:
            return message.new_empty((0, self.hidden_size))

        num_bonds_per_graph = num_bonds_per_graph.to(message.device)
        expected_num_bonds = int(num_bonds_per_graph.sum().item())
        if expected_num_bonds != message.size(0):
            raise ValueError(
                "num_bonds_per_graph does not match the number of bond messages: "
                f"expected {expected_num_bonds}, got {message.size(0)}."
            )

        graph_index = torch.arange(
            num_bonds_per_graph.numel(),
            device=message.device,
            dtype=torch.long,
        )
        bond_batch = torch.repeat_interleave(graph_index, num_bonds_per_graph.long())

        dense_message, mask = to_dense_batch(message, bond_batch)
        batch_size, max_bonds, _ = dense_message.shape

        b_q = self.W_b_q(dense_message).view(
            batch_size, max_bonds, self.num_heads, self.att_size
        )
        b_k = self.W_b_k(dense_message).view(
            batch_size, max_bonds, self.num_heads, self.att_size
        )
        b_v = self.W_b_v(dense_message).view(
            batch_size, max_bonds, self.num_heads, self.att_size
        )
        b_q = b_q.transpose(1, 2)
        b_k = b_k.transpose(1, 2)
        b_v = b_v.transpose(1, 2)

        alpha_weight = F.softmax(b_q * self.weight_alpha * self.scale_factor, dim=-1)
        alpha_weight = alpha_weight * mask[:, None, :, None]
        global_query = torch.sum(alpha_weight * b_q, dim=2)

        p = global_query[:, :, None, :] * b_k
        beta_weight = F.softmax(p * self.weight_beta * self.scale_factor, dim=-1)
        beta_weight = beta_weight * mask[:, None, :, None]
        global_key = torch.sum(beta_weight * p, dim=2)

        key_value_interaction = global_key[:, :, None, :] * b_v
        att_b_h = self.weight_r(key_value_interaction) + b_q
        att_b_h = self.act_func(att_b_h)
        att_b_h = self.dropout_layer(att_b_h)
        att_b_h = att_b_h.transpose(1, 2).contiguous()
        att_b_h = att_b_h.view(batch_size, max_bonds, self.num_heads * self.att_size)
        att_b_h = self.W_b_o(att_b_h)
        att_b_h = self.norm(att_b_h)

        return att_b_h[mask]


class GraphEncoder(nn.Module):
    def __init__(self, args: TrainConfig):
        super().__init__()
        self.args = args
        self.device = args.device
        self.rbf = BondLengthRBF(args.device)
        self.W_rbf = nn.Linear(20, 32)
        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM
        self.hidden_channels = args.hidden_size
        self.activation = get_activation_function(args.activation)
        mapper_cls = FFKN if args.use_ffkn else DenseFFN

        convs = []
        for idx in range(args.depth):
            in_channels = self.atom_fdim if idx == 0 else self.hidden_channels
            convs.append(
                OursConv(
                    in_channels=in_channels,
                    out_channels=self.hidden_channels,
                    activation=self.activation,
                    edge_hidden=self.bond_fdim + 32,
                    num_heads=args.num_heads,
                    args=args,
                    mapper_cls=mapper_cls,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, data) -> Tensor:
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        pos = data.pos.to(self.device)
        batch = data.batch.to(self.device)
        num_bonds_per_graph = data.num_bonds.to(self.device)
        b_scope_starts = torch.cat(
            [
                torch.tensor([0], device=batch.device),
                torch.cumsum(num_bonds_per_graph, dim=0)[:-1],
            ]
        )

        row, col = edge_index
        dist = (pos[col] - pos[row]).pow(2).sum(dim=-1).sqrt().unsqueeze(1)
        edge_attr = torch.cat((edge_attr, self.W_rbf(self.rbf(dist))), dim=-1)

        for conv in self.convs:
            x = conv(
                x,
                edge_index,
                edge_attr,
                b_scope_starts=b_scope_starts,
                num_bonds_per_graph=num_bonds_per_graph,
            )
        return global_mean_pool(x, batch)


class OursConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_hidden: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        num_heads: Optional[int] = None,
        args: Optional[TrainConfig] = None,
        mapper_cls: Callable = DenseFFN,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.mapper_cls = mapper_cls
        edge_hidden = edge_hidden if edge_hidden is not None else in_channels
        self.edge_mapper = nn.Linear(edge_hidden, out_channels)
        self.msg_mapper = self.mapper_cls(
            in_channels + out_channels, in_channels, activation=activation, capacity=1
        )
        self.mlp = self.mapper_cls(in_channels, out_channels, activation=activation, capacity=1)
        self.use_bond_attention = True if args is None else args.use_bond_attention
        self.edge_attention = MultiBondFastAttention(args) if self.use_bond_attention else None
        self.initial_eps = eps
        if train_eps:
            self.eps = Parameter(torch.empty(1))
        else:
            self.register_buffer("eps", torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Optional[Tensor] = None,
        size: Size = None,
        b_scope_starts: Optional[Tensor] = None,
        num_bonds_per_graph: Optional[Tensor] = None,
    ) -> Tensor:
        pair = (x, x)
        out = self.propagate(
            edge_index,
            x=pair,
            edge_attr=edge_attr,
            b_scope_starts=b_scope_starts,
            num_bonds_per_graph=num_bonds_per_graph,
        )
        out = out + (1 + self.eps) * pair[1]
        return self.mlp(out)

    def message(
        self,
        x_j: Tensor,
        edge_attr: Tensor,
        b_scope_starts: Tensor,
        num_bonds_per_graph: Tensor,
    ) -> Tensor:
        edge_attr = self.edge_mapper(edge_attr)
        if self.edge_attention is not None:
            edge_attr = self.edge_attention(edge_attr, b_scope_starts, num_bonds_per_graph)
        return self.msg_mapper(torch.cat((x_j, edge_attr), dim=-1))


class Exp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.exp()


def build_ffn(
    first_linear_dim: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    dropout: float,
    activation: str,
    dataset_type: Optional[str] = None,
    spectra_activation: Optional[str] = None,
) -> nn.Sequential:
    act = get_activation_function(activation)
    if num_layers == 1:
        layers = [nn.Dropout(dropout), nn.Linear(first_linear_dim, output_size)]
    else:
        layers = [nn.Dropout(dropout), nn.Linear(first_linear_dim, hidden_size)]
        for _ in range(num_layers - 2):
            layers.extend([act, nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size)])
        layers.extend([act, nn.Dropout(dropout), nn.Linear(hidden_size, output_size)])

    if dataset_type == "spectra":
        layers.append(nn.Softplus() if spectra_activation == "softplus" else Exp())
    return nn.Sequential(*layers)


class MoleculeModel(nn.Module):
    def __init__(self, args: TrainConfig):
        super().__init__()
        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function
        self.train_class_sizes = getattr(args, "train_class_sizes", None)

        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in [
                "cross_entropy",
                "binary_cross_entropy",
            ]

        self.relative_output_size = 1
        if self.multiclass:
            self.relative_output_size *= args.multiclass_num_classes
            self.num_classes = args.multiclass_num_classes
            self.multiclass_softmax = nn.Softmax(dim=2)
        if self.loss_function == "mve":
            self.relative_output_size *= 2
        if self.loss_function == "dirichlet" and self.classification:
            self.relative_output_size *= 2
        if self.loss_function == "evidential":
            self.relative_output_size *= 4
        if self.loss_function in ["mve", "evidential", "dirichlet"]:
            self.softplus = nn.Softplus()

        self.encoder = GraphEncoder(args)
        self.readout = self._build_readout(args)
        initialize_weights(self)

    def _build_readout(self, args: TrainConfig) -> nn.Sequential:
        return build_ffn(
            first_linear_dim=args.hidden_size,
            hidden_size=args.ffn_hidden_size,
            num_layers=args.ffn_num_layers,
            output_size=self.relative_output_size * args.num_tasks,
            dropout=args.dropout,
            activation=args.activation,
            dataset_type=args.dataset_type,
            spectra_activation=args.spectra_activation,
        )

    def fingerprint(self, batch, fingerprint_type: str = "encoder", **kwargs) -> Tensor:
        if fingerprint_type in {"encoder", "MPN"}:
            return self.encoder(batch)
        if fingerprint_type == "last_FFN":
            return self.readout[:-1](self.encoder(batch))
        raise ValueError(f"Unsupported fingerprint type {fingerprint_type}.")

    def forward(self, batch, *args, **kwargs) -> Tensor:
        output = self.readout(self.encoder(batch))

        if self.multiclass:
            output = output.reshape((output.shape[0], -1, self.num_classes))
            if (
                not (self.training and self.no_training_normalization)
                and self.loss_function != "dirichlet"
            ):
                output = self.multiclass_softmax(output)

        if self.loss_function == "mve":
            means, variances = torch.split(output, output.shape[1] // 2, dim=1)
            output = torch.cat([means, self.softplus(variances)], dim=1)
        if self.loss_function == "evidential":
            means, lambdas, alphas, betas = torch.split(output, output.shape[1] // 4, dim=1)
            output = torch.cat(
                [
                    means,
                    self.softplus(lambdas),
                    self.softplus(alphas) + 1,
                    self.softplus(betas),
                ],
                dim=1,
            )
        if self.loss_function == "dirichlet" and self.classification:
            output = self.softplus(output) + 1

        return output


def build_model(config: TrainConfig) -> nn.Module:
    return MoleculeModel(config)


def build_legacy_model(config: TrainConfig) -> nn.Module:
    """Backward-compatible alias for old training code."""
    return build_model(config)
