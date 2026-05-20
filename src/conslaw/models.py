"""
Neural operators for conservation-law surrogates: FNO, CNN (periodic / outflow),
and hybrid (spectral + multi-resolution local + gated mix) in 1D and 2D.

Boundary conventions
--------------------
* ``periodic``: FFT / ``circular`` convolutions; discrete-mean corrections where noted.
* ``outflow`` (transmissive): zero-padding for spectral paths and ``replicate`` convs;
  optional learned **constant-in-space** flux imbalance added to the conservative residual,
  matching the 1D formulas in ``Hyperbolic/*/euler_dual_latent_q.py`` and
  ``burgers_dual_latent_q.py``.

Tensor layouts
--------------
* 1D: ``(B, N, C)`` — batch, cells, channels.
* 2D: ``(B, Ny, Nx, C)``.

Dt-step modules (``*DtStep*``) follow ``u^{n+1} = u^n + dt * rhs(u)`` with optional
spatial mean removal on ``rhs``, compatible with ``Hyperbolic/Burgers2D/*_dt.py`` trainers.
1D conservative hybrid maps use ``u^{n+1} = u^n - q(u^n)`` with zero-mean or affine ``q``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def state_dict_for_ckpt(model: nn.Module) -> dict[str, Any]:
    return getattr(model, "_orig_mod", model).state_dict()


def maybe_torch_compile(
    model: nn.Module,
    device: torch.device,
    *,
    no_compile: bool = False,
    compile_mode: str = "auto",
    fullgraph: bool = False,
) -> nn.Module:
    if no_compile or not hasattr(torch, "compile"):
        return model
    if compile_mode == "auto":
        mode = "max-autotune" if device.type == "cuda" else "reduce-overhead"
    else:
        mode = compile_mode
    try:
        return torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=False)
    except Exception as exc:
        print(f"[compile] torch.compile skipped ({exc}).")
        return model



def zero_mean_q(tilde_q: torch.Tensor) -> torch.Tensor:
    """``tilde_q`` (B, N, C) → zero spatial mean per batch and channel."""
    return tilde_q - tilde_q.mean(dim=1, keepdim=True)


# This is not used.
def affine_q_outflow(
    tilde_q: torch.Tensor,
    F_L: torch.Tensor,
    F_R: torch.Tensor,
    n_cells: int,
    dx: torch.Tensor,
) -> torch.Tensor:
    """q = tilde_q - mean + (F_R - F_L) / (N * dx). F_* shape (B, C)."""
    mean_q = tilde_q.mean(dim=1, keepdim=True)
    den = float(n_cells) * dx
    den = torch.clamp(den, min=1e-30)
    delta = (F_R - F_L).unsqueeze(1) / den
    return tilde_q - mean_q + delta


def zero_mean_rhs_2d(tilde: torch.Tensor) -> torch.Tensor:
    """Remove spatial mean over Ny, Nx per batch and channel."""
    return tilde - tilde.mean(dim=(1, 2), keepdim=True)

# This is not used.
def affine_rhs_outflow_2d(
    tilde_rhs: torch.Tensor,
    F_w: torch.Tensor,
    F_e: torch.Tensor,
    F_s: torch.Tensor,
    F_n: torch.Tensor,
    nx: int,
    ny: int,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """
    Add separable x/y net flux corrections (constant in space), analogous to 1D affine_q.

    ``F_w, F_e, F_s, F_n`` shape ``(B, C)`` (west/east/south/north boundary flux surrogates).
    """
    mean = tilde_rhs.mean(dim=(1, 2), keepdim=True)
    den_x = float(nx) * dx
    den_y = float(ny) * dy
    den_x = torch.clamp(den_x, min=1e-30)
    den_y = torch.clamp(den_y, min=1e-30)
    dfx = (F_e - F_w).unsqueeze(1).unsqueeze(1) / den_x
    dfy = (F_n - F_s).unsqueeze(1).unsqueeze(1) / den_y
    return tilde_rhs - mean + dfx + dfy


def _broadcast_dt(dt: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    d = dt
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, device=u.device, dtype=u.dtype)
    if d.dim() == 0:
        d = d.view(1, 1, 1, 1).expand(u.size(0), 1, 1, 1)
    else:
        while d.dim() < u.dim():
            d = d.unsqueeze(-1)
    return d


def _broadcast_dt_1d(dt: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    d = dt
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, device=u.device, dtype=u.dtype)
    if d.dim() == 0:
        d = d.view(1, 1, 1).expand(u.size(0), 1, 1)
    else:
        while d.dim() < u.dim():
            d = d.unsqueeze(-1)
    return d


# ---------------------------------------------------------------------------
# Spectral convs (real/imag weights) — shared by FNO and hybrid
# ---------------------------------------------------------------------------
#
# Same mathematics as a complex weight matrix on low Fourier modes, but parameters are
# stored as (W_r, W_i) and combined as in complex multiplication. This matches
# ``Hyperbolic/Burgers/burgers_dual_latent_q.py`` / ``euler_dual_latent_q.py`` and keeps
# ``torch.compile`` from routing complex dtypes through Inductor. ``@dynamo.disable``
# keeps rfft/irfft + spectrum ops eager.


class RealSpectralConv1d(nn.Module):
    """
    1D FNO-style spectral layer: learn low modes of ``R`` with
    ``(R @ u_hat)_k = (W_r + i W_i)(u_r + i u_i)`` on truncated coefficients.

    Input ``x``: ``(B, C_in, N)``. If ``pad > 0`` and ``not periodic``, zero-pad
    before FFT then crop (open boundaries); if ``periodic`` is True, ``pad`` is ignored.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        *,
        periodic: bool = True,
        pad: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.periodic = periodic
        self.pad = int(pad) if not periodic else 0
        scale = 1.0 / (in_channels * out_channels)
        self.weights_r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1))
        self.weights_i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1))

    @dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        pad = self.pad
        if pad > 0:
            x = F.pad(x, (pad, pad))
        N = x.size(-1)
        K = N // 2 + 1
        M = min(self.modes1, K)
        x_ft = torch.fft.rfft(x, dim=-1)
        xr = x_ft.real[:, :, :M]
        xi = x_ft.imag[:, :, :M]
        wr = self.weights_r[:, :, :M].to(dtype=xr.dtype)
        wi = self.weights_i[:, :, :M].to(dtype=xr.dtype)
        out_r = torch.einsum("bix,iox->box", xr, wr) - torch.einsum("bix,iox->box", xi, wi)
        out_i = torch.einsum("bix,iox->box", xr, wi) + torch.einsum("bix,iox->box", xi, wr)
        out_ft = x_ft.new_zeros(B, self.out_channels, K)
        out_ft[:, :, :M] = torch.complex(out_r, out_i)
        y = torch.fft.irfft(out_ft, n=N, dim=-1)
        if pad > 0:
            y = y[..., pad:-pad]
        return y


class RealSpectralConv2d(nn.Module):
    """
    2D FNO-style spectral layer (positive and negative ky blocks); optional zero
    padding on H, W before ``rfftn``. Same low-mode layout as ``clop.SpectralConv2d``.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, *, pad: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.pad = int(pad)
        scale = 1.0 / (in_channels * out_channels)
        shp = (in_channels, out_channels, modes1, modes2)
        self.weights1_r = nn.Parameter(scale * torch.randn(*shp))
        self.weights1_i = nn.Parameter(scale * torch.randn(*shp))
        self.weights2_r = nn.Parameter(scale * torch.randn(*shp))
        self.weights2_i = nn.Parameter(scale * torch.randn(*shp))

    @dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.pad
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad))
        B, _, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=cdtype)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        def _cmpl_block(xsl: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
            xr = xsl.real
            xi = xsl.imag
            m1b, m2b = xr.shape[2], xr.shape[3]
            wr_s = wr[:, :, :m1b, :m2b].to(dtype=xr.dtype)
            wi_s = wi[:, :, :m1b, :m2b].to(dtype=xr.dtype)
            out_r = torch.einsum("bixy,ioxy->boxy", xr, wr_s) - torch.einsum("bixy,ioxy->boxy", xi, wi_s)
            out_i = torch.einsum("bixy,ioxy->boxy", xr, wi_s) + torch.einsum("bixy,ioxy->boxy", xi, wr_s)
            return torch.complex(out_r, out_i)

        sl1 = x_ft[:, :, :m1, :m2]
        out_ft[:, :, :m1, :m2] = _cmpl_block(sl1, self.weights1_r, self.weights1_i)
        sl2 = x_ft[:, :, -m1:, :m2]
        out_ft[:, :, -m1:, :m2] = _cmpl_block(sl2, self.weights2_r, self.weights2_i)
        y = torch.fft.irfftn(out_ft, s=(H, W), dim=(-2, -1))
        if pad > 0:
            y = y[..., pad:-pad, pad:-pad]
        return y



class FNO1d(nn.Module):
    """
    1D FNO. ``bc='periodic'`` uses sin/cos grid features; ``bc='outflow'`` uses one linear
    coordinate and optional zero padding before the spectral stack (fixes periodic
    wrap at open boundaries).
    """

    def __init__(
        self,
        modes: int,
        width: int,
        channel: int = 1,
        layers: int = 4,
        padding: int = 2,
        last_activation: bool = False,
        bc: str = "periodic",
        *,
        in_channel: int | None = None,
        out_channel: int | None = None,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("FNO1d.bc must be 'periodic' or 'outflow'.")
        self.modes1 = modes
        self.width = width
        self.padding = int(padding)
        self.layers = layers
        self.last_activation = last_activation
        self.bc = bc
        self.in_channel = channel if in_channel is None else in_channel
        self.out_channel = channel if out_channel is None else out_channel

        grid_dim = 2 if bc == "periodic" else 1
        self.fc0 = nn.Linear(self.in_channel + grid_dim, width)
        self.convs = nn.ModuleList(
            [RealSpectralConv1d(width, width, self.modes1, periodic=True, pad=0) for _ in range(layers)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, self.out_channel)

    def get_grid(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        B, N = shape[0], shape[1]
        if self.bc == "periodic":
            grid_linear = torch.linspace(0, 1, steps=N + 1, dtype=dtype, device=device)[:-1]
            grid_sin = torch.sin(2 * torch.pi * grid_linear)
            grid_cos = torch.cos(2 * torch.pi * grid_linear)
            grid_2d = torch.stack([grid_sin, grid_cos], dim=-1)
            return grid_2d.reshape(1, N, 2).expand(B, N, 2)
        grid_linear = torch.linspace(0, 1, steps=N, dtype=dtype, device=device) + 1.0 / (2.0 * N)
        return grid_linear.reshape(1, N, 1).expand(B, N, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device, x.dtype)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        pad = 0
        if self.bc == "outflow" and self.padding > 0:
            pad = self.padding
            x = F.pad(x, (pad, pad))
        for i in range(self.layers):
            x = self.convs[i](x) + self.ws[i](x)
            if i != self.layers - 1 or self.last_activation:
                x = F.silu(x)
        if pad > 0:
            x = x[..., pad:-pad]
        x = x.permute(0, 2, 1)
        x = F.silu(self.fc1(x))
        return self.fc2(x)


class FNOFluxBackbone1d(nn.Module):
    """
    FNO spectral trunk with the same contract as :class:`HybridBackbone1d`: ``u -> {"tilde_q", "h"}``.
    The field ``tilde_q`` is the pre-projection flux surrogate; :class:`HybridFixedStepMap1d`
    then applies the same ``q``-projection and ``u^{n+1} = u^n - q`` step as the hybrid map.
    """

    def __init__(
        self,
        modes: int,
        width: int,
        n_layers: int,
        n_cons: int,
        bc: str = "periodic",
        padding: int = 2,
        *,
        last_activation: bool = False,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("FNOFluxBackbone1d.bc must be 'periodic' or 'outflow'.")
        self.modes1 = modes
        self.width = width
        self.n_layers = n_layers
        self.n_cons = n_cons
        self.bc = bc
        self.padding = int(padding)
        self.last_activation = last_activation

        grid_dim = 2 if bc == "periodic" else 1
        self.fc0 = nn.Linear(n_cons + grid_dim, width)
        self.convs = nn.ModuleList(
            [RealSpectralConv1d(width, width, modes, periodic=True, pad=0) for _ in range(n_layers)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.tilde_q_head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, n_cons),
        )

    def get_grid(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        B, N = shape[0], shape[1]
        if self.bc == "periodic":
            grid_linear = torch.linspace(0, 1, steps=N + 1, dtype=dtype, device=device)[:-1]
            grid_sin = torch.sin(2 * torch.pi * grid_linear)
            grid_cos = torch.cos(2 * torch.pi * grid_linear)
            grid_2d = torch.stack([grid_sin, grid_cos], dim=-1)
            return grid_2d.reshape(1, N, 2).expand(B, N, 2)
        grid_linear = torch.linspace(0, 1, steps=N, dtype=dtype, device=device) + 1.0 / (2.0 * N)
        return grid_linear.reshape(1, N, 1).expand(B, N, 1)

    def forward(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        if u.dim() != 3 or u.size(-1) != self.n_cons:
            raise ValueError(f"Expected u (B, N, {self.n_cons}), got {tuple(u.shape)}")
        grid = self.get_grid(u.shape, u.device, u.dtype)
        x = torch.cat((u, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        pad = 0
        if self.bc == "outflow" and self.padding > 0:
            pad = self.padding
            x = F.pad(x, (pad, pad))
        for i in range(self.n_layers):
            x = self.convs[i](x) + self.ws[i](x)
            if i != self.n_layers - 1 or self.last_activation:
                x = F.silu(x)
        if pad > 0:
            x = x[..., pad:-pad]
        x = x.permute(0, 2, 1)
        h = x
        tilde_q = self.tilde_q_head(h)
        return {"tilde_q": tilde_q, "h": h}


class FNO2d(nn.Module):
    """
    2D FNO. ``bc='periodic'`` uses four sin/cos features (clop-style).
    ``bc='outflow'`` uses cell-centered ``(x, y)`` in ``[0, 1]`` and zero-pads the
    lifted field before each spectral block.
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        channel: int = 1,
        layers: int = 4,
        padding: int = 4,
        last_activation: bool = False,
        bc: str = "periodic",
        *,
        in_channel: int | None = None,
        out_channel: int | None = None,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("FNO2d.bc must be 'periodic' or 'outflow'.")
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.layers = layers
        self.last_activation = last_activation
        self.bc = bc
        self.padding = int(padding)
        self.in_channel = channel if in_channel is None else in_channel
        self.out_channel = channel if out_channel is None else out_channel

        grid_dim = 4 if bc == "periodic" else 2
        self.fc0 = nn.Linear(self.in_channel + grid_dim, width)
        self.convs = nn.ModuleList(
            [RealSpectralConv2d(width, width, self.modes1, self.modes2, pad=0) for _ in range(layers)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, self.out_channel)

    @staticmethod
    def get_grid(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, bc: str) -> torch.Tensor:
        B, H, W = shape[0], shape[1], shape[2]
        if bc == "periodic":
            x = torch.linspace(0.0, 1.0, steps=W + 1, dtype=dtype, device=device)[:-1]
            y = torch.linspace(0.0, 1.0, steps=H + 1, dtype=dtype, device=device)[:-1]
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            twopi = torch.tensor(2.0 * np.pi, dtype=dtype, device=device)
            grid = torch.stack(
                [
                    torch.sin(twopi * xx),
                    torch.cos(twopi * xx),
                    torch.sin(twopi * yy),
                    torch.cos(twopi * yy),
                ],
                dim=-1,
            )
            return grid.unsqueeze(0).expand(B, H, W, 4)
        xs = torch.linspace(0, 1, steps=W, dtype=dtype, device=device) + 0.5 / W
        ys = torch.linspace(0, 1, steps=H, dtype=dtype, device=device) + 0.5 / H
        gx = xs.view(1, 1, W).expand(B, H, W)
        gy = ys.view(1, H, 1).expand(B, H, W)
        return torch.stack([gx, gy], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device, x.dtype, self.bc)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        pad = 0
        if self.bc == "outflow" and self.padding > 0:
            pad = self.padding
            x = F.pad(x, (pad, pad, pad, pad))
        for i in range(self.layers):
            x = self.convs[i](x) + self.ws[i](x)
            if i != self.layers - 1 or self.last_activation:
                x = F.silu(x)
        if pad > 0:
            x = x[..., pad:-pad, pad:-pad]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.silu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# CNN backbones (clop-style)
# ---------------------------------------------------------------------------


class PeriodicCNN1d(nn.Module):
    """Stack of padded convolutions; ``bc`` maps to ``padding_mode``."""

    def __init__(
        self,
        width: int,
        channel: int = 1,
        layers: int = 4,
        kernel_size: int = 5,
        padding_mode: str = "circular",
        last_activation: bool = False,
        bc: str | None = None,
        *,
        in_channel: int | None = None,
        out_channel: int | None = None,
    ):
        super().__init__()
        if bc is not None:
            padding_mode = {
                "periodic": "circular",
                "zero": "zeros",
                "reflect": "replicate",
                "outflow": "replicate",
            }.get(bc, "circular")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size should be odd for symmetric padding.")
        self.width = width
        self.layers = layers
        self.last_activation = last_activation
        self.padding_mode = padding_mode
        self.in_channel = channel if in_channel is None else in_channel
        self.out_channel = channel if out_channel is None else out_channel
        p = kernel_size // 2
        self.lift = nn.Conv1d(self.in_channel, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    padding=p,
                    padding_mode=self.padding_mode,
                )
                for _ in range(layers)
            ]
        )
        self.head1 = nn.Conv1d(width, 128, kernel_size=1)
        self.head2 = nn.Conv1d(128, self.out_channel, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        x = self.lift(x)
        for i in range(self.layers):
            x = self.blocks[i](x)
            if i != self.layers - 1 or self.last_activation:
                x = self.act(x)
        x = self.act(self.head1(x))
        x = self.head2(x)
        return x.transpose(1, 2).contiguous()


class PeriodicCNN2d(nn.Module):
    """2D CNN with optional non-periodic boundaries (replicate / zeros)."""

    def __init__(
        self,
        width: int,
        channel: int = 1,
        layers: int = 4,
        kernel_size: int = 5,
        last_activation: bool = False,
        bc: str = "periodic",
        *,
        in_channel: int | None = None,
        out_channel: int | None = None,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow", "zero"):
            raise ValueError("PeriodicCNN2d bc must be 'periodic', 'outflow', or 'zero'.")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size should be odd.")
        mode = {"periodic": "circular", "outflow": "replicate", "zero": "zeros"}[bc]
        self.width = width
        self.layers = layers
        self.last_activation = last_activation
        self.bc = bc
        self.in_channel = channel if in_channel is None else in_channel
        self.out_channel = channel if out_channel is None else out_channel
        p = kernel_size // 2
        self.lift = nn.Conv2d(self.in_channel, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    padding=p,
                    padding_mode=mode,
                )
                for _ in range(layers)
            ]
        )
        self.head1 = nn.Conv2d(width, 128, kernel_size=1)
        self.head2 = nn.Conv2d(128, self.out_channel, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.lift(x)
        for i in range(self.layers):
            x = self.blocks[i](x)
            if i != self.layers - 1 or self.last_activation:
                x = self.act(x)
        x = self.act(self.head1(x))
        x = self.head2(x)
        return x.permute(0, 2, 3, 1).contiguous()


# ---------------------------------------------------------------------------
# Hybrid 1D (dual latent): backbone vs. conservative q-projection vs. step
# ---------------------------------------------------------------------------
#
# ``HybridBackbone1d`` only maps state → features and raw ``tilde_q`` (learned flux
# surrogate before conservation).  **How** ``tilde_q`` becomes a conservative ``q`` and
# how ``u^{n+1}`` is formed from ``u`` is delegated to ``QProjector1d`` subclasses and
# optionally your own wrapper — so you can swap projection (periodic / outflow / custom)
# or replace ``u - q`` with another integrator without rewriting the trunk.


class _SpectralOp1d(nn.Module):
    def __init__(self, width: int, modes: int, *, bc: str = "periodic", spectral_pad: int = 4):
        super().__init__()
        periodic = bc == "periodic"
        pad = 0 if periodic else int(spectral_pad)
        self.spec = RealSpectralConv1d(width, width, modes, periodic=periodic, pad=pad)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.transpose(1, 2)
        y = self.spec(x) + self.w(x)
        return y.transpose(1, 2)


class _MRLocalOp1d(nn.Module):
    def __init__(self, width: int, kernel_size: int = 5, *, bc: str = "periodic"):
        super().__init__()
        assert kernel_size % 2 == 1
        p = kernel_size // 2
        pmode = "circular" if bc == "periodic" else "replicate"
        self.conv_low = nn.Conv1d(width, width, kernel_size=kernel_size, padding=p, padding_mode=pmode)
        self.conv_low2 = nn.Conv1d(width, width, kernel_size=kernel_size, padding=p, padding_mode=pmode)
        self.fuse = nn.Conv1d(2 * width, width, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.transpose(1, 2)
        if x.size(-1) % 2 != 0:
            raise ValueError(f"Spatial length N={x.size(-1)} must be even for MRLocal op.")
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        y = F.silu(self.conv_low(x1))
        y = F.silu(self.conv_low2(y))
        y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        z = torch.cat([x, y], dim=1)
        out = self.fuse(z)
        return out.transpose(1, 2)


class HybridBackbone1d(nn.Module):
    """
    Spectral + MR-local + gated hybrid trunk only: ``u -> {tilde_q, h}``.

    No conservation projection and no time step — compose with :class:`PeriodicZeroMeanQ1d`,
    :class:`OutflowAffineLearnedQ1d`, or your own module, then apply any integrator
    (e.g. ``u - q``, ``u + dt * rhs``, …).
    """

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        mr_kernel: int = 5,
        n_cons: int = 1,
        bc: str = "periodic",
        spectral_pad: int = 4,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("bc must be 'periodic' or 'outflow'")
        self.width = width
        self.n_layers = n_layers
        self.n_cons = n_cons
        self.bc = bc
        self.lift = nn.Conv1d(n_cons, width, kernel_size=1)
        self.spec_ops = nn.ModuleList(
            [_SpectralOp1d(width, modes, bc=bc, spectral_pad=spectral_pad) for _ in range(n_layers)]
        )
        self.mr_local_ops = nn.ModuleList(
            [_MRLocalOp1d(width, kernel_size=mr_kernel, bc=bc) for _ in range(n_layers)]
        )
        self.w_g = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.w_l = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.mix = nn.ModuleList([nn.Conv1d(4 * width, width, kernel_size=1) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(width, n_cons, kernel_size=1),
        )

    def forward(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        if u.dim() != 3 or u.size(-1) != self.n_cons:
            raise ValueError(f"Expected u (B, N, {self.n_cons}), got {tuple(u.shape)}")
        h = self.lift(u.transpose(1, 2)).transpose(1, 2)
        h = F.silu(h)
        for t in range(self.n_layers):
            g = F.silu(self.spec_ops[t](h))
            lfeat = F.silu(self.mr_local_ops[t](h))
            gc = self.w_g[t](g.transpose(1, 2)).transpose(1, 2)
            lc = self.w_l[t](lfeat.transpose(1, 2)).transpose(1, 2)
            c = gc * lc
            z = torch.cat([h, g, lfeat, c], dim=-1).transpose(1, 2)
            h = h + self.mix[t](z).transpose(1, 2)
        tilde_q = self.head(h.transpose(1, 2)).transpose(1, 2)
        return {"tilde_q": tilde_q, "h": h}


class PeriodicZeroMeanQ1d(nn.Module):
    """``q = tilde_q - mean_x(tilde_q)`` (per batch and channel)."""

    def forward(
        self,
        tilde_q: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        n_cells: int | None = None,
        u_ref: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del h, n_cells, u_ref
        q = zero_mean_q(tilde_q)
        return {"q": q}


class OutflowAffineLearnedQ1d(nn.Module):
    """
    Learned boundary fluxes from pooled features: ``q = affine_q_outflow(tilde_q, F_L, F_R, ...)``.
    """

    def __init__(self, n_cons: int, width: int, dx: float):
        super().__init__()
        self.n_cons = n_cons
        self.register_buffer("cell_dx", torch.tensor(float(dx), dtype=torch.float64))
        self.flux_head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2 * n_cons),
        )

    def forward(
        self,
        tilde_q: torch.Tensor,
        *,
        h: torch.Tensor,
        n_cells: int,
        u_ref: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        dtype = tilde_q.dtype
        device = tilde_q.device
        dx = self.cell_dx.to(dtype=dtype, device=device)
        pooled = h.mean(dim=1)
        flux_raw = self.flux_head(pooled)
        frl, frr = flux_raw.chunk(2, dim=-1)
        q = affine_q_outflow(tilde_q, frl, frr, n_cells, dx)
        return {"q": q, "F_L": frl, "F_R": frr}


def subtract_q_step(u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Default explicit map ``u^{n+1} = u^n - q`` (state and q same shape)."""
    return u - q


class HybridFixedStepMap1d(nn.Module):
    """
    Convenience bundle: ``HybridBackbone1d`` + :class:`PeriodicZeroMeanQ1d` or
    :class:`OutflowAffineLearnedQ1d`, and default evolution ``u_next = u - q``.

    For custom time stepping, use :class:`HybridBackbone1d` and your own ``q`` / update,
    or replace ``self.q_projector`` / override ``evolve``.
    """

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        mr_kernel: int = 5,
        n_cons: int = 1,
        bc: str = "periodic",
        dx: float = 1.0,
        spectral_pad: int = 4,
        *,
        backbone: HybridBackbone1d | None = None,
        q_projector: nn.Module | None = None,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("bc must be 'periodic' or 'outflow'")
        self.n_cons = n_cons
        self.bc = bc
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = HybridBackbone1d(
                width=width,
                n_layers=n_layers,
                modes=modes,
                mr_kernel=mr_kernel,
                n_cons=n_cons,
                bc=bc,
                spectral_pad=spectral_pad,
            )
        if q_projector is not None:
            self.q_projector = q_projector
        elif bc == "periodic":
            self.q_projector = PeriodicZeroMeanQ1d()
        else:
            self.q_projector = OutflowAffineLearnedQ1d(n_cons=n_cons, width=width, dx=dx)
        if isinstance(self.backbone, HybridBackbone1d):
            self.width = self.backbone.width
            self.n_layers = self.backbone.n_layers
        elif isinstance(self.backbone, FNOFluxBackbone1d):
            self.width = self.backbone.width
            self.n_layers = self.backbone.n_layers
        else:
            self.width = width
            self.n_layers = n_layers

    def evolve(self, u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Override in a subclass to change ``u^{n+1} = f(u, q)`` (default: ``u - q``)."""
        return subtract_q_step(u, q)

    def forward(self, u: torch.Tensor, return_aux: bool = False):
        if u.dim() != 3 or u.size(-1) != self.n_cons:
            raise ValueError(f"Expected u (B, N, {self.n_cons}), got {tuple(u.shape)}")
        out = self.backbone(u)
        tilde_q, h = out["tilde_q"], out["h"]
        proj = self.q_projector(tilde_q, h=h, n_cells=u.size(1), u_ref=u)
        q = proj["q"]
        u_next = self.evolve(u, q)
        if not return_aux:
            return u_next
        aux: Dict[str, torch.Tensor] = {"tilde_q": tilde_q, "q": q, "h": h}
        if "F_L" in proj:
            aux["F_L"] = proj["F_L"]
            aux["F_R"] = proj["F_R"]
        return u_next, aux


class _SpectralOp2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, *, bc: str = "periodic", spectral_pad: int = 4):
        super().__init__()
        pad = 0 if bc == "periodic" else int(spectral_pad)
        self.spec = RealSpectralConv2d(width, width, modes1, modes2, pad=pad)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.permute(0, 3, 1, 2).contiguous()
        y = self.spec(x) + self.w(x)
        return y.permute(0, 2, 3, 1).contiguous()

class _MRLocalOp2d(nn.Module):
    def __init__(self, width: int, kernel_size: int = 5, *, bc: str = "periodic"):
        super().__init__()
        assert kernel_size % 2 == 1
        p = kernel_size // 2
        pmode = "circular" if bc == "periodic" else "replicate"
        self.conv_low = nn.Conv2d(width, width, kernel_size=kernel_size, padding=p, padding_mode=pmode)
        self.conv_low2 = nn.Conv2d(width, width, kernel_size=kernel_size, padding=p, padding_mode=pmode)
        self.fuse = nn.Conv2d(2 * width, width, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.permute(0, 3, 1, 2).contiguous()
        H, W = x.shape[-2], x.shape[-1]
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"MRLocal2d needs even H,W; got ({H}, {W})")
        x1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        y = F.silu(self.conv_low(x1))
        y = F.silu(self.conv_low2(y))
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        z = torch.cat([x, y], dim=1)
        out = self.fuse(z)
        return out.permute(0, 2, 3, 1).contiguous()

class _LearnedGhostPadX(nn.Module):
    """Predict left/right ghost columns from a boundary-adjacent interior strip."""

    def __init__(self, channels: int, ghost_width: int, ctx_width: int, *, side: str):
        super().__init__()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.channels = int(channels)
        self.ghost_width = int(ghost_width)
        self.ctx_width = int(max(ctx_width, ghost_width))
        self.side = side
        self.net = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=(1, self.ctx_width), padding=0),
            nn.SiLU(),
            nn.Conv2d(self.channels, self.channels * self.ghost_width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {C}")
        if self.side == "left":
            if W < self.ctx_width:
                ctx = F.pad(x, (0, self.ctx_width - W, 0, 0), mode="replicate")[..., : self.ctx_width]
            else:
                ctx = x[..., : self.ctx_width]
        else:
            if W < self.ctx_width:
                ctx = F.pad(x, (self.ctx_width - W, 0, 0, 0), mode="replicate")[..., -self.ctx_width :]
            else:
                ctx = x[..., -self.ctx_width :]

        ghost = self.net(ctx)
        ghost = ghost.squeeze(-1).contiguous()
        ghost = ghost.view(B, C, self.ghost_width, H).permute(0, 1, 3, 2).contiguous()
        return ghost


class _LearnedGhostPadY(nn.Module):
    """Predict top/bottom ghost rows from a boundary-adjacent interior strip."""

    def __init__(self, channels: int, ghost_width: int, ctx_width: int, *, side: str):
        super().__init__()
        if side not in ("top", "bottom"):
            raise ValueError("side must be 'top' or 'bottom'")
        self.channels = int(channels)
        self.ghost_width = int(ghost_width)
        self.ctx_width = int(max(ctx_width, ghost_width))
        self.side = side
        self.net = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=(self.ctx_width, 1), padding=0),
            nn.SiLU(),
            nn.Conv2d(self.channels, self.channels * self.ghost_width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {C}")
        if self.side == "top":
            if H < self.ctx_width:
                ctx = F.pad(x, (0, 0, 0, self.ctx_width - H), mode="replicate")[:, :, : self.ctx_width, :]
            else:
                ctx = x[:, :, : self.ctx_width, :]
        else:
            if H < self.ctx_width:
                ctx = F.pad(x, (0, 0, self.ctx_width - H, 0), mode="replicate")[:, :, -self.ctx_width :, :]
            else:
                ctx = x[:, :, -self.ctx_width :, :]

        ghost = self.net(ctx)
        ghost = ghost.squeeze(-2).contiguous()
        ghost = ghost.view(B, C, self.ghost_width, W).contiguous()
        return ghost


class _MRLocalOp2dOutflow(nn.Module):
    def __init__(
        self,
        width: int,
        kernel_size: int = 5,
        *,
        bc: str = "periodic",
        outflow_ctx_width: int | None = None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        if bc not in ("periodic", "outflow"):
            raise ValueError("bc must be 'periodic' or 'outflow'")
        self.bc = bc
        self.k = int(kernel_size)
        self.p = self.k // 2
        self.width = int(width)
        self.conv_low = nn.Conv2d(width, width, kernel_size=self.k, padding=0)
        self.conv_low2 = nn.Conv2d(width, width, kernel_size=self.k, padding=0)
        self.fuse = nn.Conv2d(2 * width, width, kernel_size=1)
        if self.bc == "outflow":
            ctx = outflow_ctx_width if outflow_ctx_width is not None else max(2 * self.k, 8)
            self.ghost1_x = nn.ModuleDict(
                {
                    "left": _LearnedGhostPadX(width, ghost_width=self.p, ctx_width=ctx, side="left"),
                    "right": _LearnedGhostPadX(width, ghost_width=self.p, ctx_width=ctx, side="right"),
                }
            )
            self.ghost1_y = nn.ModuleDict(
                {
                    "top": _LearnedGhostPadY(width, ghost_width=self.p, ctx_width=ctx, side="top"),
                    "bottom": _LearnedGhostPadY(width, ghost_width=self.p, ctx_width=ctx, side="bottom"),
                }
            )
            self.ghost2_x = nn.ModuleDict(
                {
                    "left": _LearnedGhostPadX(width, ghost_width=self.p, ctx_width=ctx, side="left"),
                    "right": _LearnedGhostPadX(width, ghost_width=self.p, ctx_width=ctx, side="right"),
                }
            )
            self.ghost2_y = nn.ModuleDict(
                {
                    "top": _LearnedGhostPadY(width, ghost_width=self.p, ctx_width=ctx, side="top"),
                    "bottom": _LearnedGhostPadY(width, ghost_width=self.p, ctx_width=ctx, side="bottom"),
                }
            )

    def _pad_periodic(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.p, self.p, self.p, self.p), mode="circular")

    def _pad_outflow(
        self,
        x: torch.Tensor,
        ghost_x: nn.ModuleDict,
        ghost_y: nn.ModuleDict,
    ) -> torch.Tensor:
        g_left = ghost_x["left"](x)
        g_right = ghost_x["right"](x)
        x_lr = torch.cat([g_left, x, g_right], dim=-1)
        g_top = ghost_y["top"](x_lr)
        g_bottom = ghost_y["bottom"](x_lr)
        return torch.cat([g_top, x_lr, g_bottom], dim=-2)

    def _apply_conv_block(
        self,
        x: torch.Tensor,
        conv: nn.Conv2d,
        ghost_x: nn.ModuleDict | None = None,
        ghost_y: nn.ModuleDict | None = None,
    ) -> torch.Tensor:
        if self.bc == "periodic":
            x_pad = self._pad_periodic(x)
        else:
            if ghost_x is None or ghost_y is None:
                raise ValueError("Outflow padding requires ghost predictors.")
            x_pad = self._pad_outflow(x, ghost_x, ghost_y)
        return conv(x_pad)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.permute(0, 3, 1, 2).contiguous()
        H, W = x.shape[-2], x.shape[-1]
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"MRLocal2d needs even H,W; got ({H}, {W})")
        x1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        if self.bc == "periodic":
            y = F.silu(self._apply_conv_block(x1, self.conv_low))
            y = F.silu(self._apply_conv_block(y, self.conv_low2))
        else:
            y = F.silu(self._apply_conv_block(x1, self.conv_low, self.ghost1_x, self.ghost1_y))
            y = F.silu(self._apply_conv_block(y, self.conv_low2, self.ghost2_x, self.ghost2_y))
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        z = torch.cat([x, y], dim=1)
        out = self.fuse(z)
        return out.permute(0, 2, 3, 1).contiguous()

class HybridBackbone2d(nn.Module):
    """
    Hybrid trunk only: `u -> {"tilde_rhs", "h"}. No boundary projection and no dt step.    Compose with :class:PeriodicRhs2d, :class:OutflowAffineLearnedRhs2d, or a custom projector,
    then apply your integrator (explicit Euler `u + dt*rhs, RK, etc.).    """

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        mr_kernel: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        bc: str = "periodic",
        spectral_pad: int = 4,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("bc must be 'periodic' or 'outflow'")
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bc = bc
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spec_ops = nn.ModuleList(
            [_SpectralOp2d(width, modes1, modes2, bc=bc, spectral_pad=spectral_pad) for _ in range(n_layers)]
        )
        self.mr_local_ops = nn.ModuleList(
            [_MRLocalOp2d(width, kernel_size=mr_kernel, bc=bc) for _ in range(n_layers)]
        )
        self.w_g = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.w_l = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.mix = nn.ModuleList([nn.Conv2d(4 * width, width, kernel_size=1) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        if u.dim() != 4 or u.size(-1) != self.in_channels:
            raise ValueError(f"Expected u (B, Ny, Nx, {self.in_channels}), got {tuple(u.shape)}")
        x = u.permute(0, 3, 1, 2).contiguous()
        h = self.lift(x)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = F.silu(h)
        for t in range(self.n_layers):
            g = F.silu(self.spec_ops[t](h))
            lfeat = F.silu(self.mr_local_ops[t](h))
            hp = h.permute(0, 3, 1, 2)
            gp = g.permute(0, 3, 1, 2)
            lp = lfeat.permute(0, 3, 1, 2)
            gc = self.w_g[t](gp)
            lc = self.w_l[t](lp)
            c = (gc * lc).permute(0, 2, 3, 1)
            z = torch.cat([h, g, lfeat, c], dim=-1).permute(0, 3, 1, 2)
            h = h + self.mix[t](z).permute(0, 2, 3, 1)
        tilde_rhs = self.head(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return {"tilde_rhs": tilde_rhs, "h": h}

class HybridBackbone2dOutflow(nn.Module):
    """
    Hybrid trunk only: ``u -> {"tilde_rhs", "h"}``. No boundary projection and no ``dt`` step.
    Compose with :class:`PeriodicRhs2d`, :class:`OutflowAffineLearnedRhs2d`, or a custom projector,
    then apply your integrator (explicit Euler ``u + dt*rhs``, RK, etc.).
    """

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        mr_kernel: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        bc: str = "periodic",
        spectral_pad: int = 4,
        outflow_ctx_width: int | None = None,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("bc must be 'periodic' or 'outflow'")
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bc = bc
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spec_ops = nn.ModuleList(
            [_SpectralOp2d(width, modes1, modes2, bc=bc, spectral_pad=spectral_pad) for _ in range(n_layers)]
        )
        self.mr_local_ops = nn.ModuleList(
            [
                _MRLocalOp2dOutflow(width, kernel_size=mr_kernel, bc=bc, outflow_ctx_width=outflow_ctx_width)
                for _ in range(n_layers)
            ]
        )
        self.w_g = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.w_l = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.mix = nn.ModuleList([nn.Conv2d(4 * width, width, kernel_size=1) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        if u.dim() != 4 or u.size(-1) != self.in_channels:
            raise ValueError(f"Expected u (B, Ny, Nx, {self.in_channels}), got {tuple(u.shape)}")
        x = u.permute(0, 3, 1, 2).contiguous()
        h = self.lift(x)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = F.silu(h)
        for t in range(self.n_layers):
            g = F.silu(self.spec_ops[t](h))
            lfeat = F.silu(self.mr_local_ops[t](h))
            hp = h.permute(0, 3, 1, 2)
            gp = g.permute(0, 3, 1, 2)
            lp = lfeat.permute(0, 3, 1, 2)
            gc = self.w_g[t](gp)
            lc = self.w_l[t](lp)
            c = (gc * lc).permute(0, 2, 3, 1)
            z = torch.cat([h, g, lfeat, c], dim=-1).permute(0, 3, 1, 2)
            h = h + self.mix[t](z).permute(0, 2, 3, 1)
        tilde_rhs = self.head(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return {"tilde_rhs": tilde_rhs, "h": h}

class PeriodicRhs2d(nn.Module):
    """Periodic domain: optionally subtract spatial mean per channel (mass-conserving residual)."""

    def forward(
        self,
        tilde_rhs: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        nx: int | None = None,
        ny: int | None = None,
        zero_mean_rhs: bool = True,
        u_ref: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del h, nx, ny, u_ref
        rhs = zero_mean_rhs_2d(tilde_rhs) if zero_mean_rhs else tilde_rhs
        return {"rhs": rhs}


class OutflowAffineLearnedRhs2d(nn.Module):
    """Learned west/east/south/north scalars → affine correction of ``tilde_rhs`` (see ``affine_rhs_outflow_2d``)."""

    def __init__(self, out_channels: int, width: int, dx: float, dy: float):
        super().__init__()
        self.out_channels = out_channels
        self.register_buffer("cell_dx", torch.tensor(float(dx), dtype=torch.float64))
        self.register_buffer("cell_dy", torch.tensor(float(dy), dtype=torch.float64))
        self.flux_head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 4 * out_channels),
        )

    def forward(
        self,
        tilde_rhs: torch.Tensor,
        *,
        h: torch.Tensor,
        nx: int,
        ny: int,
        zero_mean_rhs: bool = True,
        u_ref: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del u_ref
        if not zero_mean_rhs:
            return {"rhs": tilde_rhs}
        pooled = h.mean(dim=(1, 2))
        flux_raw = self.flux_head(pooled)
        fw, fe, fs, fn = flux_raw.chunk(4, dim=-1)
        dtype = tilde_rhs.dtype
        device = tilde_rhs.device
        dx_b = self.cell_dx.to(dtype=dtype, device=device)
        dy_b = self.cell_dy.to(dtype=dtype, device=device)
        rhs = affine_rhs_outflow_2d(tilde_rhs, fw, fe, fs, fn, nx, ny, dx_b, dy_b)
        return {"rhs": rhs, "F_w": fw, "F_e": fe, "F_s": fs, "F_n": fn}


class IdentityRhs2d(nn.Module):
    """Pass through ``tilde_rhs`` without any projection or affine correction."""

    def forward(
        self,
        tilde_rhs: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        nx: int | None = None,
        ny: int | None = None,
        zero_mean_rhs: bool = True,
        u_ref: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del h, nx, ny, zero_mean_rhs, u_ref
        return {"rhs": tilde_rhs}


def explicit_euler_dt_step(u: torch.Tensor, dt: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """``u^{n+1} = u^n + dt * rhs`` (broadcast ``dt`` like trainers)."""
    return u + _broadcast_dt(dt, u) * rhs


class PeriodicRhs1d(nn.Module):
    """1D periodic: optional zero-mean of ``tilde_rhs`` per batch/channel (cf. :class:`PeriodicRhs2d`)."""

    def forward(
        self,
        tilde_rhs: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        n_cells: int | None = None,
        u_ref: torch.Tensor | None = None,
        zero_mean_rhs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del h, n_cells, u_ref
        rhs = zero_mean_q(tilde_rhs) if zero_mean_rhs else tilde_rhs
        return {"rhs": rhs}


class IdentityRhs1d(nn.Module):
    """Pass through ``tilde_rhs`` (optional no-op; same role as :class:`IdentityRhs2d`)."""

    def forward(
        self,
        tilde_rhs: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        n_cells: int | None = None,
        u_ref: torch.Tensor | None = None,
        zero_mean_rhs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del h, n_cells, u_ref, zero_mean_rhs
        return {"rhs": tilde_rhs}


class HybridDtStep1d(nn.Module):
    """
    Same pattern as :class:`HybridDtStep2d`: :class:`HybridBackbone1d` produces a raw field
    (``tilde_q`` tensor, interpreted as ``tilde_rhs``), :class:`PeriodicRhs1d` or
    :class:`OutflowAffineLearnedQ1d` projects to ``rhs``, then explicit Euler ``u + dt * rhs``.

    The backbone head is shared with the flux path; only the one-step map and training target
    differ (see trainers: ``u^{n+1} ≈ u^n + dt * rhs`` vs ``u^{n+1} ≈ u^n - q``).
    """

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        mr_kernel: int = 5,
        n_cons: int = 1,
        bc: str = "periodic",
        dx: float = 1.0,
        spectral_pad: int = 4,
        *,
        zero_mean_rhs: bool = True,
        project_outflow_rhs: bool = True,
        backbone: HybridBackbone1d | None = None,
        rhs_projector: nn.Module | None = None,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("bc must be 'periodic' or 'outflow'")
        self.n_cons = n_cons
        self.bc = bc
        self.zero_mean_rhs = bool(zero_mean_rhs)
        self.project_outflow_rhs = bool(project_outflow_rhs)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = HybridBackbone1d(
                width=width,
                n_layers=n_layers,
                modes=modes,
                mr_kernel=mr_kernel,
                n_cons=n_cons,
                bc=bc,
                spectral_pad=spectral_pad,
            )
        if rhs_projector is not None:
            self.rhs_projector = rhs_projector
        elif bc == "periodic":
            self.rhs_projector = PeriodicRhs1d()
        elif not self.project_outflow_rhs:
            self.rhs_projector = IdentityRhs1d()
        else:
            self.rhs_projector = OutflowAffineLearnedQ1d(n_cons=n_cons, width=width, dx=dx)
        if isinstance(self.backbone, HybridBackbone1d):
            self.width = self.backbone.width
            self.n_layers = self.backbone.n_layers
        else:
            self.width = width
            self.n_layers = n_layers

    def evolve(self, u: torch.Tensor, dt: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return u + _broadcast_dt_1d(dt, u) * rhs

    def forward(self, u: torch.Tensor, dt: torch.Tensor, return_aux: bool = False):
        if u.dim() != 3 or u.size(-1) != self.n_cons:
            raise ValueError(f"Expected u (B, N, {self.n_cons}), got {tuple(u.shape)}")
        n_cells = u.size(1)
        out = self.backbone(u)
        tilde_rhs = out["tilde_q"]
        h = out["h"]
        if isinstance(self.rhs_projector, OutflowAffineLearnedQ1d):
            proj = self.rhs_projector(tilde_rhs, h=h, n_cells=n_cells, u_ref=u)
            rhs = proj["q"]
        else:
            proj = self.rhs_projector(
                tilde_rhs,
                h=h,
                n_cells=n_cells,
                u_ref=u,
                zero_mean_rhs=self.zero_mean_rhs,
            )
            rhs = proj["rhs"]
        u_next = self.evolve(u, dt, rhs)
        if not return_aux:
            return u_next
        aux: Dict[str, torch.Tensor] = {"tilde_rhs": tilde_rhs, "rhs": rhs, "h": h}
        for k in ("F_L", "F_R"):
            if k in proj:
                aux[k] = proj[k]
        return u_next, aux


class HybridDtStep2d(nn.Module):
    """
    Default bundle: :class:`HybridBackbone2d` (or an injected :class:`HybridBackbone2dOutflow`)
    + :class:`PeriodicRhs2d` or :class:`OutflowAffineLearnedRhs2d`, and explicit Euler ``u + dt * rhs``.

    Override :meth:`evolve` for RK / other integrators; replace ``rhs_projector`` or use
    a custom backbone alone for full control.
    """

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        mr_kernel: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        bc: str = "periodic",
        dx: float = 1.0,
        dy: float = 1.0,
        spectral_pad: int = 4,
        *,
        zero_mean_rhs: bool = True,
        project_outflow_rhs: bool = True,
        backbone: HybridBackbone2d | HybridBackbone2dOutflow | None = None,
        rhs_projector: nn.Module | None = None,
    ):
        super().__init__()
        if bc not in ("periodic", "outflow"):
            raise ValueError("bc must be 'periodic' or 'outflow'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bc = bc
        self.zero_mean_rhs = bool(zero_mean_rhs)
        self.project_outflow_rhs = bool(project_outflow_rhs)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = HybridBackbone2d(
                width=width,
                n_layers=n_layers,
                modes1=modes1,
                modes2=modes2,
                mr_kernel=mr_kernel,
                in_channels=in_channels,
                out_channels=out_channels,
                bc=bc,
                spectral_pad=spectral_pad,
            )
        if rhs_projector is not None:
            self.rhs_projector = rhs_projector
        elif bc == "periodic":
            self.rhs_projector = PeriodicRhs2d()
        elif not self.project_outflow_rhs:
            self.rhs_projector = IdentityRhs2d()
        else:
            self.rhs_projector = OutflowAffineLearnedRhs2d(
                out_channels=out_channels, width=width, dx=dx, dy=dy
            )
        if isinstance(self.backbone, (HybridBackbone2d, HybridBackbone2dOutflow)):
            self.width = self.backbone.width
            self.n_layers = self.backbone.n_layers
        else:
            self.width = width
            self.n_layers = n_layers

    def evolve(self, u: torch.Tensor, dt: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Override for multi-stage schemes; default is explicit Euler."""
        return explicit_euler_dt_step(u, dt, rhs)

    def forward(self, u: torch.Tensor, dt: torch.Tensor, return_aux: bool = False):
        if u.dim() != 4 or u.size(-1) != self.in_channels:
            raise ValueError(f"Expected u (B, Ny, Nx, {self.in_channels}), got {tuple(u.shape)}")
        _, ny, nx, _ = u.shape
        out = self.backbone(u)
        tilde_rhs, h = out["tilde_rhs"], out["h"]
        proj = self.rhs_projector(
            tilde_rhs,
            h=h,
            nx=nx,
            ny=ny,
            zero_mean_rhs=self.zero_mean_rhs,
            u_ref=u,
        )
        rhs = proj["rhs"]
        u_next = self.evolve(u, dt, rhs)
        if not return_aux:
            return u_next
        aux: Dict[str, torch.Tensor] = {"tilde_rhs": tilde_rhs, "rhs": rhs, "h": h}
        for k in ("F_w", "F_e", "F_s", "F_n"):
            if k in proj:
                aux[k] = proj[k]
        return u_next, aux


# ---------------------------------------------------------------------------
# Dt-step wrappers for FNO / CNN (trainer-compatible)
# ---------------------------------------------------------------------------


class FNODtStep1d(nn.Module):
    """``u + dt * FNO(u)`` with optional zero-mean ``rhs``."""

    def __init__(
        self,
        *,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        padding: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        bc: str = "periodic",
        zero_mean_rhs: bool = True,
    ):
        super().__init__()
        self.zero_mean_rhs = bool(zero_mean_rhs)
        self.fno = FNO1d(
            modes=modes,
            width=width,
            channel=in_channels,
            layers=n_layers,
            padding=padding,
            bc=bc,
            in_channel=in_channels,
            out_channel=out_channels,
        )

    def forward(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        rhs = self.fno(u)
        if self.zero_mean_rhs:
            rhs = rhs - rhs.mean(dim=1, keepdim=True)
        d = _broadcast_dt_1d(dt, u)
        return u + d * rhs


class FNODtStep2d(nn.Module):
    """Same interface as ``Hyperbolic/Burgers2D/burgers2d_fno_dt.Burgers2DFNODtStep``."""

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        bc: str = "periodic",
        padding: int = 4,
        *,
        zero_mean_rhs: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.zero_mean_rhs = bool(zero_mean_rhs)
        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            channel=in_channels,
            layers=n_layers,
            bc=bc,
            padding=padding,
            in_channel=in_channels,
            out_channel=out_channels,
        )

    def forward(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if u.dim() != 4 or u.size(-1) != self.in_channels:
            raise ValueError(f"Expected u (B, Ny, Nx, {self.in_channels}), got {tuple(u.shape)}")
        rhs = self.fno(u)
        if self.zero_mean_rhs:
            rhs = rhs - rhs.mean(dim=(1, 2), keepdim=True)
        d = _broadcast_dt(dt, u)
        return u + d * rhs


class CNNDtStep2d(nn.Module):
    """CNN backbone with same dt-step interface as ``FNODtStep2d``."""

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        kernel_size: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        bc: str = "periodic",
        *,
        zero_mean_rhs: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.zero_mean_rhs = bool(zero_mean_rhs)
        self.cnn = PeriodicCNN2d(
            width=width,
            channel=in_channels,
            layers=n_layers,
            kernel_size=kernel_size,
            bc=bc,
            in_channel=in_channels,
            out_channel=out_channels,
        )

    def forward(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        rhs = self.cnn(u)
        if self.zero_mean_rhs:
            rhs = rhs - rhs.mean(dim=(1, 2), keepdim=True)
        d = _broadcast_dt(dt, u)
        return u + d * rhs


class CNNDtStep1d(nn.Module):
    """1D CNN residual step ``u + dt * CNN(u)``."""

    def __init__(
        self,
        width: int = 64,
        n_layers: int = 4,
        kernel_size: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        bc: str = "periodic",
        *,
        zero_mean_rhs: bool = True,
    ):
        super().__init__()
        self.zero_mean_rhs = bool(zero_mean_rhs)
        self.cnn = PeriodicCNN1d(
            width=width,
            channel=in_channels,
            layers=n_layers,
            kernel_size=kernel_size,
            bc=bc,
            in_channel=in_channels,
            out_channel=out_channels,
        )

    def forward(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        rhs = self.cnn(u)
        if self.zero_mean_rhs:
            rhs = rhs - rhs.mean(dim=1, keepdim=True)
        d = _broadcast_dt_1d(dt, u)
        return u + d * rhs


__all__ = [
    "affine_q_outflow",
    "affine_rhs_outflow_2d",
    "CNNDtStep1d",
    "CNNDtStep2d",
    "count_params",
    "explicit_euler_dt_step",
    "FNO1d",
    "FNO2d",
    "FNOFluxBackbone1d",
    "FNODtStep1d",
    "FNODtStep2d",
    "HybridBackbone1d",
    "HybridBackbone2d",
    "HybridDtStep1d",
    "HybridDtStep2d",
    "HybridFixedStepMap1d",
    "IdentityRhs1d",
    "PeriodicRhs1d",
    "maybe_torch_compile",
    "OutflowAffineLearnedQ1d",
    "OutflowAffineLearnedRhs2d",
    "PeriodicRhs2d",
    "PeriodicZeroMeanQ1d",
    "PeriodicCNN1d",
    "PeriodicCNN2d",
    "RealSpectralConv1d",
    "RealSpectralConv2d",
    "state_dict_for_ckpt",
    "subtract_q_step",
    "zero_mean_q",
    "zero_mean_rhs_2d",
]
