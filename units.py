# %%
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_unit_type(unit_type):
    if unit_type.endswith("_LinearScaler"):
        return unit_type.replace("_LinearScaler", ""), "linear"
    if unit_type.endswith("_SigmoidScaler"):
        return unit_type.replace("_SigmoidScaler", ""), "sigmoid"
    return unit_type, "constant"

def _temperature_scaler_param_count(num_subunits, mode):
    if mode == "linear":
        return num_subunits + 1
    if mode == "sigmoid":
        sub_features = round(math.sqrt(num_subunits)) if num_subunits > 1 else 1
        return num_subunits * sub_features + 2 * sub_features + 1
    if mode == "constant":
        return 0
    raise ValueError(f"Unknown scaling mode: {mode}")


def calculate_sub_units(unit_type, input_dim, output_dim, num_params):
    if unit_type == "UniversalMaxout":
        # Two Maxout branches plus the final linear layer
        param_linear_unit = (2 * output_dim + 1) * output_dim
        param_per_sub_unit = (input_dim + 1) * output_dim
        sub_units = (num_params - param_linear_unit) // (2 * param_per_sub_unit)
    elif unit_type == "MoE":
        # Total parameters: (input_dim + 1) * sub_units * (output_dim + 1)
        sub_units = num_params // ((input_dim + 1) * (output_dim + 1))
    elif unit_type == "Maxout":
        # Total parameters: (input_dim + 1) * output_dim * sub_units
        param_per_sub_unit = (input_dim + 1) * output_dim
        sub_units = num_params // param_per_sub_unit
    elif (
        unit_type.startswith("BDGU")
        or unit_type.startswith("BMGU")
        or unit_type.startswith("BRGU")
    ):
        base_type, scaling_mode = parse_unit_type(unit_type)
        if base_type not in {"BDGU", "BMGU", "BRGU"}:
            raise ValueError(f"Unknown unit type: {unit_type}")

        # Base parameter count excludes the temperature scaler
        param_per_sub_unit = 2 * (input_dim + 1) * output_dim
        max_guess = max(num_params // param_per_sub_unit, 0)

        def total_params(sub_units):
            if base_type == "BDGU":
                scaler_inputs = sub_units + 1
            else:
                scaler_inputs = sub_units * sub_units + 1
            scaler_params = _temperature_scaler_param_count(scaler_inputs, scaling_mode)
            return param_per_sub_unit * sub_units + scaler_params

        lo, hi, best = 0, max_guess, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if total_params(mid) <= num_params:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        sub_units = best

    else:
        raise ValueError(f"Unknown unit type: {unit_type}")
    return sub_units


class TemperatureScaler(nn.Module):
    def __init__(self, num_subunits, out_features, mode, seed=None):
        super().__init__()
        self.mode = mode
        self.temp = math.sqrt(out_features)  # Default attention temperature

        if mode == "linear":
            self.scaler = nn.Linear(num_subunits, 1)
            self.init_parameters(seed)
        elif mode == "sigmoid":
            sub_features = round(math.sqrt(num_subunits)) if num_subunits > 1 else 1
            self.pre_scaler = nn.Linear(num_subunits, sub_features)
            self.scaler = nn.Linear(sub_features, 1)
            self.sigmoid = nn.Sigmoid()
            self.init_parameters(seed)

    def init_parameters(self, seed=None):
        if seed != None:
            g = torch.Generator().manual_seed(seed)
            nn.init.xavier_normal_(self.scaler.weight, generator=g)
        else:
            nn.init.xavier_normal_(self.scaler.weight)

    def get_temperature(self):
        return self.temp

    def forward(self, x):
        if self.mode == "linear":
            temp = self.scaler(x)
            self.temp = torch.clamp(temp, min=0.1, max=10)
            return self.temp
        elif self.mode == "sigmoid":
            temp = self.pre_scaler(x)
            temp = self.sigmoid(temp)
            temp = self.scaler(temp)
            temp = self.sigmoid(temp)
            self.temp = torch.clamp(temp * 10, min=0.1, max=10)
            return self.temp
        elif self.mode == "constant":
            return torch.tensor(self.temp, device=x.device)
        else:
            raise ValueError(f"Unknown scaling mode: {self.mode}")


class BaseUnit(nn.Module):
    def __init__(self, in_features, out_features, sub_units, dropout=0, seed=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_units = sub_units
        self.seed = seed
        self.dropout = dropout

    def init_parameters(self, weights, seed=None):
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            init_fn = lambda w: nn.init.xavier_normal_(w, generator=g)
        else:
            init_fn = nn.init.xavier_normal_

        for w in weights:
            if w.ndim == 3:
                for d in range(self.sub_units):
                    init_fn(w[:, :, d])
            else:
                init_fn(w)


class Maxout(BaseUnit):
    def __init__(
        self,
        in_features,
        out_features,
        sub_units,
        dropout=0,
        seed=None,
        scaling_mode="constant",
    ):
        super().__init__(in_features, out_features, sub_units, dropout, seed)

        self.transforms = nn.Parameter(
            torch.Tensor(in_features + 1, out_features, sub_units)
        )

        self.dropout_layer = nn.Dropout(p=dropout)

        self.init_parameters([self.transforms], self.seed)

    def forward(self, x):
        x = torch.cat([x, x.new_ones(x.size(0), 1)], dim=1)
        trans = torch.einsum("bi,iod->bod", x, self.transforms)
        trans = self.dropout_layer(trans)
        out, _ = torch.max(trans, dim=2)
        return out


class UniversalMaxout(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        sub_units,
        dropout=0,
        seed=None,
        scaling_mode="constant",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_units = sub_units

        self.unit_positive = Maxout(in_features, out_features, sub_units, dropout, seed)
        self.unit_negative = Maxout(in_features, out_features, sub_units, dropout, seed)
        self.linear = nn.Linear(out_features * 2, out_features)

    def forward(self, x):
        out_positive = self.unit_positive(x)
        out_negative = self.unit_negative(x)
        out = torch.cat([out_positive, out_negative], dim=1)
        out = self.linear(out)
        return out


class MoE(BaseUnit):
    def __init__(
        self,
        in_features,
        out_features,
        sub_units,
        dropout=0,
        seed=None,
        scaling_mode="constant",
    ):
        super().__init__(in_features, out_features, sub_units, dropout, seed)
        self.transforms = nn.Parameter(
            torch.empty(in_features + 1, out_features, sub_units)
        )
        self.gate_transforms = nn.Parameter(torch.empty(in_features + 1, sub_units))
        self.dropout = nn.Dropout(p=dropout)

        self.init_parameters([self.transforms, self.gate_transforms], self.seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, x.new_ones(x.size(0), 1)], dim=1)

        y = torch.einsum("bi,iod->bod", x, self.transforms)

        gate_weights = torch.einsum("bi,id->bd", x, self.gate_transforms)
        gate_weights = self.dropout(gate_weights)
        gate_weights = F.softmax(gate_weights, dim=1)

        out = torch.sum(y * gate_weights.unsqueeze(1), dim=2)

        return out


class BRGU(BaseUnit):
    def __init__(
        self,
        in_features,
        out_features,
        sub_units,
        dropout=0,
        seed=None,
        scaling_mode="constant",
    ):
        super().__init__(in_features, out_features, sub_units, dropout, seed)

        self.transforms_e = nn.Parameter(
            torch.Tensor(in_features + 1, out_features, sub_units)
        )
        self.transforms_g = nn.Parameter(
            torch.Tensor(in_features + 1, out_features, sub_units)
        )
        self.scaler = TemperatureScaler(sub_units * sub_units + 1, out_features, scaling_mode, seed)

        self.dropout = nn.Dropout(p=dropout)

        self.init_parameters([self.transforms_e, self.transforms_g], seed)

    def get_temperature(self):
        return self.scaler.get_temperature()

    def forward(self, x):
        x = torch.cat([x, x.new_ones(x.size(0), 1)], dim=1)

        experts = torch.einsum("bi,iod->bod", x, self.transforms_e)
        gates = torch.einsum("bi,iod->bod", x, self.transforms_g)

        attention = torch.einsum("bod,bot->bdt", experts, gates)
        temperature = self.scaler(
            torch.cat(
                [
                    attention.reshape(attention.size(0), -1),
                    attention.new_ones(attention.size(0), 1),
                ],
                dim=1,
            )
        )
        attention = attention / temperature
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("bot,bdt->bod", experts, attention)

        out = out.sum(dim=2)

        return out


class BMGU(BaseUnit):
    def __init__(
        self,
        in_features,
        out_features,
        sub_units,
        dropout=0,
        seed=None,
        scaling_mode="constant",
    ):
        super().__init__(in_features, out_features, sub_units, dropout, seed)

        self.transforms_e = nn.Parameter(
            torch.Tensor(in_features + 1, out_features, sub_units)
        )
        self.transforms_g = nn.Parameter(
            torch.Tensor(in_features + 1, out_features, sub_units)
        )
        self.scaler = TemperatureScaler(sub_units * sub_units + 1, out_features, scaling_mode, seed)

        self.dropout = nn.Dropout(p=dropout)
        self.init_parameters([self.transforms_e, self.transforms_g], seed)

    def get_temperature(self):
        return self.scaler.get_temperature()

    def forward(self, x):
        x = torch.cat([x, x.new_ones(x.size(0), 1)], dim=1)

        experts = torch.einsum("bi,iod->bod", x, self.transforms_e)
        gates = torch.einsum("bi,iod->bod", x, self.transforms_g)

        attention = torch.einsum("bod,bot->bdt", experts, gates)
        attention = attention.reshape(attention.size(0), -1)
        temperature = self.scaler(
            torch.cat(
                [attention, attention.new_ones(attention.size(0), 1)],
                dim=1,
            )
        )
        attention = attention / temperature
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        attention = attention.reshape(attention.size(0), self.sub_units, self.sub_units)

        out = torch.einsum("bot,bdt->bod", experts, attention)

        out = out.sum(dim=2)

        return out


class BDGU(BaseUnit):
    def __init__(
        self,
        in_features,
        out_features,
        sub_units,
        dropout=0,
        seed=None,
        scaling_mode="constant",
    ):
        super().__init__(in_features, out_features, sub_units, dropout, seed)

        self.transforms_e = nn.Parameter(
            torch.Tensor(in_features + 1, out_features, sub_units)
        )
        self.transforms_g = nn.Parameter(
            torch.Tensor(in_features + 1, out_features, sub_units)
        )
        self.scaler = TemperatureScaler(sub_units + 1, out_features, scaling_mode, seed)

        self.dropout = nn.Dropout(p=dropout)
        self.init_parameters([self.transforms_e, self.transforms_g], seed)

    def get_temperature(self):
        return self.scaler.get_temperature()

    def forward(self, x):
        x = torch.cat([x, x.new_ones(x.size(0), 1)], dim=1)

        experts = torch.einsum("bi,iod->bod", x, self.transforms_e)
        gates = torch.einsum("bi,iod->bod", x, self.transforms_g)

        attention = torch.einsum("bod,bod->bd", experts, gates)
        temperature = self.scaler(
            torch.cat(
                [attention, attention.new_ones(attention.size(0), 1)],
                dim=1,
            )
        )
        attention = attention / temperature
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        out = torch.sum(experts * attention.unsqueeze(1), dim=2)

        return out
