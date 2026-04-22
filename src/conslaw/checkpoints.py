from __future__ import annotations

from typing import Any, Tuple

import torch

from conslaw.models import (
    CNNDtStep2d,
    FNOFluxBackbone1d,
    FNODtStep2d,
    HybridBackbone2dOutflow,
    HybridDtStep2d,
    HybridFixedStepMap1d,
    maybe_torch_compile,
)


def _split_complex_tensor(v: Any) -> tuple[Any, Any]:
    if torch.is_complex(v):
        return v.real, v.imag
    if torch.is_floating_point(v):
        return v, torch.zeros_like(v)
    raise TypeError(f"Expected floating or complex tensor, got {type(v)!r}.")


def _convert_legacy_fno_complex_weights(sd: dict[str, Any]) -> dict[str, Any]:
    """Convert old complex FNO weights to real/imag parameter pairs used by conslaw models."""
    out: dict[str, Any] = {}
    for k, v in sd.items():
        if k.endswith("weights1") or k.endswith("weights2") or k.endswith("weights_low"):
            real, imag = _split_complex_tensor(v)
            out[k + "_r"] = real
            out[k + "_i"] = imag
        else:
            out[k] = v
    return out


def _remap_legacy_fixedstep_1d(sd: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    outflow = any(k.startswith("flux_head.") for k in sd)
    for k, v in sd.items():
        if k == "cell_dx":
            if outflow:
                out["q_projector.cell_dx"] = v
            continue
        if k.startswith("flux_head."):
            out["q_projector." + k] = v  # OutflowAffineLearnedQ1d.flux_head
        else:
            out["backbone." + k] = v
    return out


def _remap_legacy_hybrid_dt_2d(sd: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith("flux_head."):
            out["rhs_projector." + k] = v
        else:
            out["backbone." + k] = v
    return out


def _prepare_dt_step_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    return _convert_legacy_fno_complex_weights(sd)


def _prepare_legacy_hybrid_dt_2d_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    return _convert_legacy_fno_complex_weights(_remap_legacy_hybrid_dt_2d(sd))


def _use_fno_flux_backbone(args: dict[str, Any], kind: str) -> bool:
    if args.get("backbone") == "fno":
        return True
    return "fno_flux" in str(kind)


def _use_fno_dt_backbone(args: dict[str, Any], kind: str) -> bool:
    if args.get("backbone") == "fno":
        return True
    return "fno_dt" in str(kind)


def _use_cnn_dt_backbone(args: dict[str, Any], kind: str) -> bool:
    if args.get("backbone") == "cnn":
        return True
    return "cnn_dt" in str(kind)


def _use_outflow_hybrid_dt_backbone(sd: dict[str, Any]) -> bool:
    return any(".ghost" in k for k in sd)


def _infer_outflow_ctx_width_2d(sd: dict[str, Any]) -> int | None:
    for key, value in sd.items():
        if key.endswith("ghost1_x.left.net.0.weight") or key.endswith("ghost2_x.left.net.0.weight"):
            if hasattr(value, "shape") and len(value.shape) == 4:
                return int(value.shape[-1])
        if key.endswith("ghost1_y.top.net.0.weight") or key.endswith("ghost2_y.top.net.0.weight"):
            if hasattr(value, "shape") and len(value.shape) == 4:
                return int(value.shape[-2])
    return None


def load_hybrid_fixedstep_map_1d(
    path: str,
    device: torch.device | str = "cpu",
    map_location: str | None = None,
) -> Tuple[HybridFixedStepMap1d, dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location or device)
    args = ckpt.get("args", {})
    kind = ckpt.get("kind", "")
    if "euler" in kind:
        n_cons = int(args.get("n_cons", 3))
    elif "swe" in kind:
        n_cons = int(args.get("n_cons", 2))
    elif "burgers" in kind or "pureconvection" in kind:
        n_cons = int(args.get("n_cons", 1))
    else:
        n_cons = int(args.get("n_cons", infer_n_cons_fixedstep_1d(ckpt.get("model", {}))))

    use_fno = _use_fno_flux_backbone(args, str(kind))
    if use_fno:
        backbone = FNOFluxBackbone1d(
            modes=int(args.get("modes", 32)),
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", args.get("layers", 4))),
            n_cons=n_cons,
            bc=str(ckpt.get("bc", args.get("bc", "periodic"))),
            padding=int(args.get("fno_padding", args.get("padding", 2))),
        )
        model = HybridFixedStepMap1d(
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", args.get("layers", 4))),
            modes=int(args.get("modes", 16)),
            mr_kernel=int(args.get("mr_kernel", 5)),
            n_cons=n_cons,
            bc=str(ckpt.get("bc", args.get("bc", "periodic"))),
            dx=float(ckpt.get("dx", args.get("dx", 1.0))),
            spectral_pad=int(args.get("spectral_pad", 4)),
            backbone=backbone,
        )
    else:
        model = HybridFixedStepMap1d(
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", args.get("layers", 4))),
            modes=int(args.get("modes", 16)),
            mr_kernel=int(args.get("mr_kernel", 5)),
            n_cons=n_cons,
            bc=str(ckpt.get("bc", args.get("bc", "periodic"))),
            dx=float(ckpt.get("dx", args.get("dx", 1.0))),
            spectral_pad=int(args.get("spectral_pad", 4)),
        )
    sd = ckpt["model"]
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(_remap_legacy_fixedstep_1d(sd), strict=True)
    model.to(device)
    return model, ckpt


def infer_n_cons_fixedstep_1d(sd: dict[str, Any]) -> int:
    for key in ("backbone.head.4.weight", "head.4.weight", "backbone.tilde_q_head.2.weight"):
        if key in sd:
            return int(sd[key].shape[0])
    return 1


def load_hybrid_dt_step_2d(
    path: str,
    device: torch.device | str = "cpu",
    map_location: str | None = None,
) -> Tuple[torch.nn.Module, dict[str, Any]]:
    """Restore :class:`HybridDtStep2d` (e.g. ``kind``: ``burgers2d_hybrid_dt``)."""
    ckpt = torch.load(path, map_location=map_location or device)
    args = ckpt.get("args", {})
    modes = int(args.get("modes", 16))
    modes1 = int(args.get("modes1", modes))
    modes2 = int(args.get("modes2", modes))
    raw_sd = ckpt["model"]
    use_outflow_hybrid = _use_outflow_hybrid_dt_backbone(raw_sd)
    outflow_ctx_width = _infer_outflow_ctx_width_2d(raw_sd)
    if _use_fno_dt_backbone(args, str(ckpt.get("kind", ""))):
        model = FNODtStep2d(
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", args.get("layers", 4))),
            modes1=modes1,
            modes2=modes2,
            in_channels=int(args.get("in_channels", 1)),
            out_channels=int(args.get("out_channels", 1)),
            bc=str(args.get("bc", ckpt.get("bc", "periodic"))),
            padding=int(args.get("padding", args.get("spectral_pad", 4))),
            zero_mean_rhs=bool(args.get("zero_mean_rhs", True)),
        )
    elif _use_cnn_dt_backbone(args, str(ckpt.get("kind", ""))):
        model = CNNDtStep2d(
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", args.get("layers", 4))),
            kernel_size=int(args.get("kernel_size", 5)),
            in_channels=int(args.get("in_channels", 1)),
            out_channels=int(args.get("out_channels", 1)),
            bc=str(args.get("bc", ckpt.get("bc", "periodic"))),
            zero_mean_rhs=bool(args.get("zero_mean_rhs", True)),
        )
    else:
        backbone = None
        if use_outflow_hybrid:
            backbone = HybridBackbone2dOutflow(
                width=int(args.get("width", 64)),
                n_layers=int(args.get("n_layers", args.get("layers", 4))),
                modes1=modes1,
                modes2=modes2,
                mr_kernel=int(args.get("mr_kernel", 5)),
                in_channels=int(args.get("in_channels", 1)),
                out_channels=int(args.get("out_channels", 1)),
                bc=str(args.get("bc", ckpt.get("bc", "periodic"))),
                spectral_pad=int(args.get("spectral_pad", 4)),
                outflow_ctx_width=outflow_ctx_width,
            )
        model = HybridDtStep2d(
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", args.get("layers", 4))),
            modes1=modes1,
            modes2=modes2,
            mr_kernel=int(args.get("mr_kernel", 5)),
            in_channels=int(args.get("in_channels", 1)),
            out_channels=int(args.get("out_channels", 1)),
            bc=str(args.get("bc", ckpt.get("bc", "periodic"))),
            dx=float(ckpt.get("dx", args.get("dx", 1.0))),
            dy=float(ckpt.get("dy", args.get("dy", 1.0))),
            spectral_pad=int(args.get("spectral_pad", 4)),
            zero_mean_rhs=bool(args.get("zero_mean_rhs", True)),
            project_outflow_rhs=bool(args.get("project_outflow_rhs", True)),
            backbone=backbone,
        )
    sd = _prepare_dt_step_state_dict(ckpt["model"])
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(_prepare_legacy_hybrid_dt_2d_state_dict(ckpt["model"]), strict=True)
    model.to(device)
    return model, ckpt


def compile_if_requested(
    model: torch.nn.Module,
    device: torch.device,
    *,
    no_compile: bool = False,
    compile_mode: str = "auto",
) -> torch.nn.Module:
    return maybe_torch_compile(model, device, no_compile=no_compile, compile_mode=compile_mode, fullgraph=False)
