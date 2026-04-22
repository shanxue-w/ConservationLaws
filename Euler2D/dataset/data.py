"""
2D ideal-gas Euler on [x_left, x_right] × [y_bottom, y_top] with FD-WENOZ dimension splitting.

Conserved variables (cell / point values):

    U = (ρ, ρu, ρv, E)^T,

    U_t + F(U)_x + G(U)_y = 0,

    F = (ρu, ρu²+p, ρuv, (E+p)u)^T,
    G = (ρv, ρuv, ρv²+p, (E+p)v)^T,

    p = (γ-1)(E - 1/2 ρ(u²+v²)).

Default reconstruction: **characteristic** (Roe-averaged ∂F/∂U / ∂G/∂U at each interface,
eigendecomposition for left/right matrices passed to ``FD_WENOZ``).

``FD_WENOZ`` only supports characteristic WENO for states of shape ``(n_comp, N)``;
this module therefore applies the 1D solver **line-by-line** in x and y (same splitting
idea as ``FD_WENOZ2D`` for scalars).

Boundary conditions follow ``clop.solver.FD_WENOZ`` / ``FD_WENOZ2D`` (e.g. ``outflow`` =
edge extrapolation of ghosts).

Dataset generation (``build_euler2d_quadrant_split`` / ``python -m Euler2D.dataset.data``):
evolve on a fine grid with **outflow** BC, **quadrant** piecewise-constant IC (split at
``(x0, y0)``). By default each trajectory samples **four independent random** primitive
states (log-uniform ``ρ,p``, uniform ``u,v``; default ranges are milder than
``(0.08,1)×(0.02,1)×[-1,1]`` to improve stability); use ``--fixed_quadrant_ic``
for the classic benchmark ``U1..U4``. Coarse data store conservative cell averages only.
Default ``dt_snap = 1e-2``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from clop.solver import FD_WENOZ, SOLVER_DTYPE, downsample_cell_average2d

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
GAMMA = 1.4
RHO_MIN = 1e-10
P_MIN = 1e-12

# Default primitive sampling for random quadrant ICs (log-uniform ρ,p; uniform u,v).
# Tighter than the old (0.08,1)×(0.02,1)×[-1,1] to avoid extreme jumps / near-vacuum
# states that often blow up under FD-WENOZ + splitting.
DEFAULT_RHO_RANGE = (0.08, 1.0)
DEFAULT_P_RANGE = (0.04, 0.5)
DEFAULT_UV_RANGE = (-0.5, 0.5)

# Random quadrant primitives: enforce |u|/c < this (each quadrant independent).
QUADRANT_MACH_MAX = 3.5


def pressure_conserved(u: np.ndarray, gamma: float = GAMMA, rho_min: float = RHO_MIN) -> np.ndarray:
    """u: (4, ...) -> p same trailing shape."""
    rho = np.maximum(u[0], rho_min)
    mx, my, E = u[1], u[2], u[3]
    kin = 0.5 * (mx * mx + my * my) / rho
    return np.maximum((gamma - 1.0) * (E - kin), P_MIN)


def euler_flux_x(u: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """F(U), u shape (4, N) or (4, ny, nx)."""
    rho = np.maximum(u[0], RHO_MIN)
    mx, my, E = u[1], u[2], u[3]
    uu = mx / rho
    vv = my / rho
    p = pressure_conserved(u, gamma=gamma)
    f0 = mx
    f1 = mx * uu + p
    f2 = mx * vv
    f3 = (E + p) * uu
    return np.stack([f0, f1, f2, f3], axis=0)


def euler_flux_y(u: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """G(U)."""
    rho = np.maximum(u[0], RHO_MIN)
    mx, my, E = u[1], u[2], u[3]
    # uu = mx / rho
    vv = my / rho
    p = pressure_conserved(u, gamma=gamma)
    g0 = my
    g1 = mx * vv
    g2 = my * vv + p
    g3 = (E + p) * vv
    return np.stack([g0, g1, g2, g3], axis=0)


def euler_alpha_x(u: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """|λ|_max along x for CFL: |u|+c on the same grid as u (trailing dims)."""
    rho = np.maximum(u[0], RHO_MIN)
    mx = u[1]
    uu = mx / rho
    p = pressure_conserved(u, gamma=gamma)
    c = np.sqrt(np.maximum(gamma * p / rho, 1e-30))
    return np.abs(uu) + c


def euler_alpha_y(u: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """|v|+c."""
    rho = np.maximum(u[0], RHO_MIN)
    my = u[2]
    vv = my / rho
    p = pressure_conserved(u, gamma=gamma)
    c = np.sqrt(np.maximum(gamma * p / rho, 1e-30))
    return np.abs(vv) + c


def roe_average_state(uL: np.ndarray, uR: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    Roe-averaged conservative state at each interface.
    uL, uR: (4, Nint)
    returns U_roe: (4, Nint)
    """
    gm1 = gamma - 1.0
    rhoL = np.maximum(uL[0], RHO_MIN)
    rhoR = np.maximum(uR[0], RHO_MIN)
    mxL, myL, EL = uL[1], uL[2], uL[3]
    mxR, myR, ER = uR[1], uR[2], uR[3]
    uL1, vL = mxL / rhoL, myL / rhoL
    uR1, vR = mxR / rhoR, myR / rhoR
    pL = pressure_conserved(uL, gamma=gamma)
    pR = pressure_conserved(uR, gamma=gamma)
    HL = (EL + pL) / rhoL
    HR = (ER + pR) / rhoR

    srL = np.sqrt(rhoL)
    srR = np.sqrt(rhoR)
    den = np.maximum(srL + srR, 1e-14)
    uu = (srL * uL1 + srR * uR1) / den
    vv = (srL * vL + srR * vR) / den
    H = (srL * HL + srR * HR) / den
    rho_roe = srL * srR

    pr = gm1 / gamma * (H - 0.5 * (uu * uu + vv * vv))
    p_roe = np.maximum(rho_roe * pr, P_MIN)
    mx = rho_roe * uu
    my = rho_roe * vv
    E = p_roe / gm1 + 0.5 * rho_roe * (uu * uu + vv * vv)
    return np.stack([rho_roe, mx, my, E], axis=0)


def euler_jacobian_fx_batch(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    ∂F/∂U at each column (last axis = spatial index).
    U: (4, N) -> J: (N, 4, 4)
    """
    gm1 = gamma - 1.0
    q1 = np.maximum(U[0], RHO_MIN)
    q2, q3, q4 = U[1], U[2], U[3]
    kin = 0.5 * (q2 * q2 + q3 * q3) / q1
    p = np.maximum(gm1 * (q4 - kin), P_MIN)

    dp_d1 = gm1 * (0.5 * (q2 * q2 + q3 * q3) / (q1 * q1))
    dp_d2 = -gm1 * q2 / q1
    dp_d3 = -gm1 * q3 / q1
    dp_d4 = np.full_like(q1, gm1)

    N = U.shape[1]
    J = np.zeros((N, 4, 4), dtype=U.dtype)

    # F0 = q2
    J[:, 0, 1] = 1.0

    # F1 = q2^2/q1 + p
    J[:, 1, 0] = -(q2 * q2) / (q1 * q1) + dp_d1
    J[:, 1, 1] = 2.0 * q2 / q1 + dp_d2
    J[:, 1, 2] = dp_d3
    J[:, 1, 3] = dp_d4

    # F2 = q2*q3/q1
    J[:, 2, 0] = -q2 * q3 / (q1 * q1)
    J[:, 2, 1] = q3 / q1
    J[:, 2, 2] = q2 / q1

    # F3 = q2*(q4+p)/q1
    K = q4 + p
    dK_d1 = dp_d1
    dK_d2 = dp_d2
    dK_d3 = dp_d3
    dK_d4 = 1.0 + dp_d4
    J[:, 3, 0] = q2 * (q1 * dK_d1 - K) / (q1 * q1)
    J[:, 3, 1] = q2 * dK_d2 / q1 + K / q1
    J[:, 3, 2] = q2 * dK_d3 / q1
    J[:, 3, 3] = q2 * dK_d4 / q1

    return J


def euler_jacobian_gy_batch(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """∂G/∂U, U: (4, N) -> (N, 4, 4)."""
    gm1 = gamma - 1.0
    q1 = np.maximum(U[0], RHO_MIN)
    q2, q3, q4 = U[1], U[2], U[3]
    kin = 0.5 * (q2 * q2 + q3 * q3) / q1
    p = np.maximum(gm1 * (q4 - kin), P_MIN)

    dp_d1 = gm1 * (0.5 * (q2 * q2 + q3 * q3) / (q1 * q1))
    dp_d2 = -gm1 * q2 / q1
    dp_d3 = -gm1 * q3 / q1
    dp_d4 = np.full_like(q1, gm1)

    N = U.shape[1]
    J = np.zeros((N, 4, 4), dtype=U.dtype)

    # G0 = q3
    J[:, 0, 2] = 1.0

    # G1 = q2*q3/q1
    J[:, 1, 0] = -q2 * q3 / (q1 * q1)
    J[:, 1, 1] = q3 / q1
    J[:, 1, 2] = q2 / q1

    # G2 = q3^2/q1 + p
    J[:, 2, 0] = -(q3 * q3) / (q1 * q1) + dp_d1
    J[:, 2, 1] = dp_d2
    J[:, 2, 2] = 2.0 * q3 / q1 + dp_d3
    J[:, 2, 3] = dp_d4

    K = q4 + p
    dK_d1 = dp_d1
    dK_d2 = dp_d2
    dK_d3 = dp_d3
    dK_d4 = 1.0 + dp_d4
    J[:, 3, 0] = q3 * (q1 * dK_d1 - K) / (q1 * q1)
    J[:, 3, 1] = q3 * dK_d2 / q1
    J[:, 3, 2] = q3 * dK_d3 / q1 + K / q1
    J[:, 3, 3] = q3 * dK_d4 / q1

    return J


def euler_characteristic_matrices_fx_batch(
    uL: np.ndarray, uR: np.ndarray, gamma: float = GAMMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U = roe_average_state(uL, uR, gamma=gamma)
    rho = np.maximum(U[0], RHO_MIN)
    mx, my, E = U[1], U[2], U[3]
    uu = mx / rho
    vv = my / rho
    p = pressure_conserved(U, gamma=gamma)
    H = (E + p) / rho
    a = np.sqrt(np.maximum((gamma - 1.0) * (H - 0.5 * (uu * uu + vv * vv)), 1e-30))

    Nint = U.shape[1]
    R = np.zeros((Nint, 4, 4), dtype=np.float64)
    # x-direction right eigenvectors (columns): u-a, u, u, u+a
    R[:, :, 0] = np.stack(
        [np.ones_like(uu), uu - a, vv, H - uu * a],
        axis=1,
    )
    R[:, :, 1] = np.stack(
        [np.ones_like(uu), uu, vv, 0.5 * (uu * uu + vv * vv)],
        axis=1,
    )
    R[:, :, 2] = np.stack(
        [np.zeros_like(uu), np.zeros_like(uu), np.ones_like(uu), vv],
        axis=1,
    )
    R[:, :, 3] = np.stack(
        [np.ones_like(uu), uu + a, vv, H + uu * a],
        axis=1,
    )

    # Batch inverse in NumPy C core; much faster than per-interface eig.
    L = np.linalg.inv(R)
    # Fallback regularization for near-singular interfaces.
    bad = ~np.isfinite(L).all(axis=(1, 2))
    if np.any(bad):
        R[bad] += 1e-10 * np.eye(4)[None, :, :]
        L[bad] = np.linalg.inv(R[bad])

    a_int = np.abs(uu) + a
    return L, R, a_int


def euler_characteristic_matrices_gy_batch(
    uL: np.ndarray, uR: np.ndarray, gamma: float = GAMMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U = roe_average_state(uL, uR, gamma=gamma)
    rho = np.maximum(U[0], RHO_MIN)
    mx, my, E = U[1], U[2], U[3]
    uu = mx / rho
    vv = my / rho
    p = pressure_conserved(U, gamma=gamma)
    H = (E + p) / rho
    a = np.sqrt(np.maximum((gamma - 1.0) * (H - 0.5 * (uu * uu + vv * vv)), 1e-30))

    Nint = U.shape[1]
    R = np.zeros((Nint, 4, 4), dtype=np.float64)
    # y-direction right eigenvectors (columns): v-a, v, v, v+a
    R[:, :, 0] = np.stack(
        [np.ones_like(vv), uu, vv - a, H - vv * a],
        axis=1,
    )
    R[:, :, 1] = np.stack(
        [np.ones_like(vv), uu, vv, 0.5 * (uu * uu + vv * vv)],
        axis=1,
    )
    R[:, :, 2] = np.stack(
        [np.zeros_like(vv), np.ones_like(vv), np.zeros_like(vv), uu],
        axis=1,
    )
    R[:, :, 3] = np.stack(
        [np.ones_like(vv), uu, vv + a, H + vv * a],
        axis=1,
    )

    L = np.linalg.inv(R)
    bad = ~np.isfinite(L).all(axis=(1, 2))
    if np.any(bad):
        R[bad] += 1e-10 * np.eye(4)[None, :, :]
        L[bad] = np.linalg.inv(R[bad])

    a_int = np.abs(vv) + a
    return L, R, a_int


def build_euler2d_solver(
    gamma: float = GAMMA,
    bc: str = "outflow",
    reconstruction: str = "characteristic",
    WENOtype: str = "WENO-Z",
    eps: float = 1e-20,
    flux_split: str = "local_lf",
    *,
    verbose: bool = True,
    line_batch_size: int = 16,
):
    """
    Line-wise FD-WENOZ for 2D Euler; state layout ``u`` is ``(4, ny, nx)``.
    """
    if reconstruction not in ("characteristic", "component"):
        raise ValueError('reconstruction must be "characteristic" or "component".')

    def fx(u):
        return euler_flux_x(u, gamma=gamma)

    def fy(u):
        return euler_flux_y(u, gamma=gamma)

    char_x = None
    char_y = None
    if reconstruction == "characteristic":
        char_x = lambda uL, uR: euler_characteristic_matrices_fx_batch(uL, uR, gamma=gamma)
        char_y = lambda uL, uR: euler_characteristic_matrices_gy_batch(uL, uR, gamma=gamma)

    solver_x = FD_WENOZ(
        flux=fx,
        dflux=None,
        alpha=lambda u: euler_alpha_x(u, gamma=gamma),
        char_decomp_batch=char_x,
        n_comp=4,
        flux_split=flux_split,
        eps=eps,
        bc=bc,
        WENOtype=WENOtype,
        dtype=np.float64,
    )
    solver_y = FD_WENOZ(
        flux=fy,
        dflux=None,
        alpha=lambda u: euler_alpha_y(u, gamma=gamma),
        char_decomp_batch=char_y,
        n_comp=4,
        flux_split=flux_split,
        eps=eps,
        bc=bc,
        WENOtype=WENOtype,
        dtype=np.float64,
    )
    return Euler2DWENOOperator(
        solver_x,
        solver_y,
        verbose=verbose,
        line_batch_size=line_batch_size,
    )


class Euler2DWENOOperator:
    """Dimension-split SSPRK3 + line-wise ``FD_WENOZ`` for ``(4, ny, nx)`` states."""

    def __init__(
        self,
        solver_x: FD_WENOZ,
        solver_y: FD_WENOZ,
        verbose: bool = True,
        line_batch_size: int = 16,
    ):
        self.solver_x = solver_x
        self.solver_y = solver_y
        self.verbose = bool(verbose)
        self.line_batch_size = max(1, int(line_batch_size))
        self._step_count = 0

    def _reset_step_count(self):
        self._step_count = 0

    def _print_after_step(self, t_phys: float, dt_used: float, u: np.ndarray):
        """One line per RK3 step (each internal CFL sub-step)."""
        if not self.verbose:
            return
        self._step_count += 1
        rho = u[0]
        p = pressure_conserved(u)
        mx, my = u[1], u[2]
        speed = np.sqrt(np.maximum(mx * mx + my * my, 0.0) / np.maximum(rho, RHO_MIN))
        print(
            f"[Euler2D] step {self._step_count:6d}  t={t_phys:.6e}  dt={dt_used:.6e}  "
            f"rho[min,max]=({rho.min():.4e},{rho.max():.4e})  "
            f"p[min,max]=({p.min():.4e},{p.max():.4e})  |U|_max={speed.max():.4e}",
            flush=True,
        )

    @staticmethod
    def _to_numpy(u):
        return np.asarray(u, dtype=SOLVER_DTYPE)

    def _rhs_lines_batched(self, lines: np.ndarray, d: float, solver: FD_WENOZ) -> np.ndarray:
        """
        Apply 1D rhs to a stack of lines.

        lines: (n_lines, 4, n_cells)
        return: same shape
        """
        n_lines = lines.shape[0]
        out = np.empty_like(lines)
        rhs = solver.rhs
        bs = self.line_batch_size
        for s in range(0, n_lines, bs):
            e = min(s + bs, n_lines)
            blk = lines[s:e]
            out[s:e] = np.stack([rhs(blk[k], d) for k in range(blk.shape[0])], axis=0)
        return out

    def _rhs_x_lines(self, u: np.ndarray, dx: float) -> np.ndarray:
        # Pack y-lines as a batch: (ny, 4, nx)
        lines = np.transpose(u, (1, 0, 2))
        out_lines = self._rhs_lines_batched(lines, dx, self.solver_x)
        return np.transpose(out_lines, (1, 0, 2))

    def _rhs_y_lines(self, u: np.ndarray, dy: float) -> np.ndarray:
        # Pack x-lines as a batch with y as last axis: (nx, 4, ny)
        lines = np.transpose(u, (2, 0, 1))
        out_lines = self._rhs_lines_batched(lines, dy, self.solver_y)
        return np.transpose(out_lines, (1, 2, 0))

    def rhs(self, u: np.ndarray, dx: float, dy: float) -> np.ndarray:
        u = self._to_numpy(u)
        return self._rhs_x_lines(u, dx) + self._rhs_y_lines(u, dy)

    def _amax_pair(self, u: np.ndarray) -> tuple[float, float]:
        ax = float(np.max(euler_alpha_x(u)))
        ay = float(np.max(euler_alpha_y(u)))
        return ax, ay

    def _compute_dt(self, cur_u, dx, dy, dt=None, cfl=0.4, t_remaining=None):
        if dt is not None:
            return dt if t_remaining is None else min(dt, t_remaining)
        amax_x, amax_y = self._amax_pair(cur_u)
        denom = 0.0
        if amax_x > 1e-14:
            denom += amax_x / dx
        if amax_y > 1e-14:
            denom += amax_y / dy
        if denom < 1e-14:
            return t_remaining if t_remaining is not None else 0.0
        dt_step = cfl / denom
        return dt_step if t_remaining is None else min(dt_step, t_remaining)

    def step(self, u: np.ndarray, dx: float, dy: float, dt: float) -> np.ndarray:
        k1 = self.rhs(u, dx, dy)
        u1 = u + dt * k1
        k2 = self.rhs(u1, dx, dy)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
        k3 = self.rhs(u2, dx, dy)
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)

    def advance(
        self,
        u: np.ndarray,
        dx: float,
        dy: float,
        T: float | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        cfl: float = 0.4,
        return_all: bool = False,
    ):
        u = self._to_numpy(u).copy()
        if T is None and n_steps is None:
            raise ValueError("Provide T or n_steps.")
        self._reset_step_count()
        traj = [u.copy()] if return_all else None
        if T is not None:
            t = 0.0
            while t < T - 1e-15:
                dt_step = self._compute_dt(u, dx, dy, dt=dt, cfl=cfl, t_remaining=T - t)
                if dt_step <= 0.0:
                    break
                u = self.step(u, dx, dy, dt_step)
                t += dt_step
                self._print_after_step(t, dt_step, u)
                if return_all:
                    traj.append(u.copy())
            return np.stack(traj, axis=0) if return_all else u
        dt_step = self._compute_dt(u, dx, dy, dt=dt, cfl=cfl)
        if dt_step <= 0.0:
            raise ValueError("Wave speed ~0; set dt manually.")
        t = 0.0
        for _ in range(n_steps):
            use_dt = dt if dt is not None else dt_step
            u = self.step(u, dx, dy, use_dt)
            t += use_dt
            self._print_after_step(t, use_dt, u)
            if return_all:
                traj.append(u.copy())
        return np.stack(traj, axis=0) if return_all else u

    def solve_snapshots(
        self,
        u0: np.ndarray,
        dx: float,
        dy: float,
        dt_snap: float,
        n_snaps: int,
        cfl: float = 0.4,
        fixed_dt: float | None = None,
        *,
        log_ic_state: bool = True,
        time_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Advance by ``dt_snap`` between saves (substeps use CFL unless ``fixed_dt`` set).
        Returns ``(n_snaps, 4, ny, nx)``.

        ``log_ic_state=False`` skips the initial state print (for chained snapshot segments).
        ``time_offset`` is added to printed physical time (for demo segments after t>0).
        """
        self._reset_step_count()
        u = self._to_numpy(u0).copy()
        t_global = 0.0
        traj = [u.copy()]
        t_off = float(time_offset)
        if self.verbose and log_ic_state:
            rho = u[0]
            p = pressure_conserved(u)
            print(
                f"[Euler2D] IC  t={t_off:.6e}  rho[min,max]=({rho.min():.4e},{rho.max():.4e})  "
                f"p[min,max]=({p.min():.4e},{p.max():.4e})  "
                f"(n_snaps={int(n_snaps)}, dt_snap={dt_snap:g}, cfl={cfl:g})",
                flush=True,
            )
        for snap_idx in range(int(n_snaps) - 1):
            t_rem = float(dt_snap)
            while t_rem > 1e-15:
                dt_sub = (
                    fixed_dt
                    if fixed_dt is not None
                    else self._compute_dt(u, dx, dy, cfl=cfl, t_remaining=t_rem)
                )
                if dt_sub <= 0.0:
                    raise RuntimeError("dt_sub <= 0 in solve_snapshots")
                dt_use = min(dt_sub, t_rem)
                u = self.step(u, dx, dy, dt_use)
                t_global += dt_use
                t_rem -= dt_use
                self._print_after_step(t_off + t_global, dt_use, u)
            traj.append(u.copy())
        return np.stack(traj, axis=0)


def primitives_to_conserved_2d(
    rho: float | np.ndarray,
    u_vel: float | np.ndarray,
    v_vel: float | np.ndarray,
    p: float | np.ndarray,
    gamma: float = GAMMA,
) -> np.ndarray:
    """(ρ, u, v, p) → (ρ, ρu, ρv, E); scalars or broadcastable arrays."""
    rho = np.maximum(np.asarray(rho, dtype=np.float64), RHO_MIN)
    u_vel = np.asarray(u_vel, dtype=np.float64)
    v_vel = np.asarray(v_vel, dtype=np.float64)
    p = np.maximum(np.asarray(p, dtype=np.float64), P_MIN)
    mx = rho * u_vel
    my = rho * v_vel
    E = p / (gamma - 1.0) + 0.5 * rho * (u_vel * u_vel + v_vel * v_vel)
    return np.stack([rho, mx, my, E], axis=0)


def mach_number_primitive(
    rho: float,
    u_vel: float,
    v_vel: float,
    p: float,
    gamma: float = GAMMA,
) -> float:
    """|u| / c with c = sqrt(gamma p / rho) (ideal gas)."""
    rho = max(float(rho), RHO_MIN)
    p = max(float(p), P_MIN)
    c = math.sqrt(gamma * p / rho)
    return math.hypot(float(u_vel), float(v_vel)) / max(c, 1e-300)


def sample_quadrant_conserved(
    rng: np.random.Generator,
    *,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
    gamma: float = GAMMA,
    mach_max: float = QUADRANT_MACH_MAX,
) -> np.ndarray:
    """
    One physically valid constant state (conserved) for a quadrant.
    ρ and p are log-uniform in the given ranges; u,v are uniform.
    Rejects samples until ``|u|/c < mach_max`` (default 3.5); if many rejections,
    scales (u,v) to slightly below ``mach_max`` so the bound always holds.
    """
    if v_range is None:
        v_range = u_range
    r0, r1 = float(rho_range[0]), float(rho_range[1])
    p0, p1 = float(p_range[0]), float(p_range[1])
    if not (r0 > 0 and r1 > r0 and p0 > 0 and p1 > p0):
        raise ValueError(f"Invalid rho_range={rho_range} or p_range={p_range}.")
    mach_cap = float(mach_max)
    if not (mach_cap > 0.0):
        raise ValueError(f"mach_max must be positive, got {mach_max}.")
    for _ in range(50_000):
        rho = float(np.exp(rng.uniform(np.log(r0), np.log(r1))))
        p = float(np.exp(rng.uniform(np.log(p0), np.log(p1))))
        u_vel = float(rng.uniform(float(u_range[0]), float(u_range[1])))
        v_vel = float(rng.uniform(float(v_range[0]), float(v_range[1])))
        if mach_number_primitive(rho, u_vel, v_vel, p, gamma) < mach_cap:
            return primitives_to_conserved_2d(rho, u_vel, v_vel, p, gamma=gamma).astype(np.float64)
    rho = float(np.exp(rng.uniform(np.log(r0), np.log(r1))))
    p = float(np.exp(rng.uniform(np.log(p0), np.log(p1))))
    u_vel = float(rng.uniform(float(u_range[0]), float(u_range[1])))
    v_vel = float(rng.uniform(float(v_range[0]), float(v_range[1])))
    c = math.sqrt(gamma * max(p, P_MIN) / max(rho, RHO_MIN))
    w = math.hypot(u_vel, v_vel)
    cap = mach_cap * c * (1.0 - 1e-9)
    if w > cap and w > 0.0:
        s = cap / w
        u_vel *= s
        v_vel *= s
    return primitives_to_conserved_2d(rho, u_vel, v_vel, p, gamma=gamma).astype(np.float64)


def sample_four_quadrant_states(
    rng: np.random.Generator,
    *,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent random state per quadrant (4,) each."""
    kw = dict(rho_range=rho_range, p_range=p_range, u_range=u_range, v_range=v_range)
    return (
        sample_quadrant_conserved(rng, **kw),
        sample_quadrant_conserved(rng, **kw),
        sample_quadrant_conserved(rng, **kw),
        sample_quadrant_conserved(rng, **kw),
    )


def enforce_physical_euler2d(u: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Clamp ρ and E so pressure stays nonnegative (same idea as 1D Euler dataset)."""
    u = np.asarray(u, dtype=np.float64).copy()
    rho = np.maximum(u[0], RHO_MIN)
    mx, my, E = u[1], u[2], u[3]
    kinetic = 0.5 * (mx * mx + my * my) / rho
    E_min = kinetic + P_MIN / (gamma - 1.0)
    E = np.maximum(E, E_min)
    u[0] = rho
    u[3] = E
    return u


# Classic benchmark quadrant states (conserved), e.g. Schulz-Rinne-type setup.
CLASSIC_QUADRANT_U1 = np.array([1.5, 0.0, 0.0, 3.75], dtype=np.float64)
CLASSIC_QUADRANT_U2 = np.array([0.5323, 0.641954, 0.0, 1.1371], dtype=np.float64)
CLASSIC_QUADRANT_U3 = np.array([0.5323, 0.0, 0.641954, 1.1371], dtype=np.float64)
CLASSIC_QUADRANT_U4 = np.array([0.138, 0.166428, 0.166428, 0.273212], dtype=np.float64)


def ic_quadrant_piecewise(
    nx: int,
    ny: int,
    U1: np.ndarray,
    U2: np.ndarray,
    U3: np.ndarray,
    U4: np.ndarray,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    x0: float = 0.8,
    y0: float = 0.8,
) -> tuple[np.ndarray, float, float]:
    """
    Piecewise constant quadrant IC: U1..U4 are conserved vectors (ρ, ρu, ρv, E), shape (4,).

    Quadrant numbering: U1 top-right (x>=x0, y>=y0), U2 top-left, U3 bottom-right, U4 bottom-left.
    """
    U1 = np.asarray(U1, dtype=np.float64).ravel()
    U2 = np.asarray(U2, dtype=np.float64).ravel()
    U3 = np.asarray(U3, dtype=np.float64).ravel()
    U4 = np.asarray(U4, dtype=np.float64).ravel()
    for name, U in ("U1", U1), ("U2", U2), ("U3", U3), ("U4", U4):
        if U.shape != (4,):
            raise ValueError(f"{name} must have shape (4,), got {U.shape}")

    xl, xr = xlim
    yb, yt = ylim
    dx = (xr - xl) / nx
    dy = (yt - yb) / ny
    xc = xl + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    yc = yb + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    XX, YY = np.meshgrid(xc, yc, indexing="xy")

    u0 = np.empty((4, ny, nx), dtype=np.float64)
    hi_y = YY >= y0
    lo_y = ~hi_y
    hi_x = XX >= x0
    lo_x = XX < x0
    for k in range(4):
        q = np.empty((ny, nx), dtype=np.float64)
        q[hi_y & hi_x] = U1[k]
        q[hi_y & lo_x] = U2[k]
        q[lo_y & hi_x] = U3[k]
        q[lo_y & lo_x] = U4[k]
        u0[k] = q
    return u0, dx, dy


def ic_quadrant_riemann(
    nx: int,
    ny: int,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    x0: float = 0.8,
    y0: float = 0.8,
) -> tuple[np.ndarray, float, float]:
    """
    Classic fixed quadrant Riemann IC (same four states as before); jumps at (x0, y0).
    """
    return ic_quadrant_piecewise(
        nx,
        ny,
        CLASSIC_QUADRANT_U1,
        CLASSIC_QUADRANT_U2,
        CLASSIC_QUADRANT_U3,
        CLASSIC_QUADRANT_U4,
        xlim=xlim,
        ylim=ylim,
        x0=x0,
        y0=y0,
    )


def downsample_conserved2d(u: np.ndarray, factor_y: int, factor_x: int | None = None) -> np.ndarray:
    """Conservative block average for each conserved component; u shape (4, Ny, Nx)."""
    if factor_x is None:
        factor_x = factor_y
    return np.stack(
        [downsample_cell_average2d(u[i], factor_y, factor_x) for i in range(4)],
        axis=0,
    )

def make_quadrant_grid(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> tuple[float, float]:
    """Cell spacing on [0,lx]×[0,ly] with cell-centered WENO data layout (same as ic_quadrant_riemann)."""
    dx = lx / nx
    dy = ly / ny
    return dx, dy


def _spawn_euler2d_quadrant_seed(base_seed: int, traj_idx: int, n_ic_total: int):
    """Reproducible per-trajectory seed for dynamic worker scheduling."""
    return np.random.SeedSequence(base_seed).spawn(n_ic_total)[traj_idx]


def _build_euler2d_quadrant_meta(
    *,
    n_ic: int,
    nx_low: int,
    ny_low: int,
    upsample: int,
    T: float,
    dt_snap: float,
    cfl: float,
    lx: float,
    ly: float,
    x0: float,
    y0: float,
    random_split: bool,
    split_margin: float,
    random_quadrant_states: bool,
    rho_range: tuple[float, float],
    p_range: tuple[float, float],
    u_range: tuple[float, float],
    v_range: tuple[float, float] | None,
    reconstruction: str,
    bc: str,
    num_workers: int,
) -> dict:
    nx_fine = nx_low * upsample
    ny_fine = ny_low * upsample
    n_snaps = int(round(T / dt_snap)) + 1
    dx_low = lx / nx_low
    dy_low = ly / ny_low
    return {
        "equation": "2D Euler (rho, rho*u, rho*v, E), dimension-split FD-WENOZ",
        "boundary": bc,
        "reconstruction": reconstruction,
        "domain": [0.0, lx, 0.0, ly],
        "nx": int(nx_low),
        "ny": int(ny_low),
        "dx": float(dx_low),
        "dy": float(dy_low),
        "dt": float(dt_snap),
        "T": float(T),
        "n_snaps": int(n_snaps),
        "upsample": int(upsample),
        "nx_fine": int(nx_fine),
        "ny_fine": int(ny_fine),
        "ic": "quadrant_riemann",
        "x0_default": float(x0),
        "y0_default": float(y0),
        "random_split": bool(random_split),
        "split_margin": float(split_margin),
        "random_quadrant_states": bool(random_quadrant_states),
        "rho_range": [float(rho_range[0]), float(rho_range[1])],
        "p_range": [float(p_range[0]), float(p_range[1])],
        "u_range": [float(u_range[0]), float(u_range[1])],
        "v_range": None if v_range is None else [float(v_range[0]), float(v_range[1])],
        "quadrant_mach_max": float(QUADRANT_MACH_MAX),
        "reference": "quadrant IC + fine outflow WENO + cell-average downsample",
        "cfl": float(cfl),
        "gamma": float(GAMMA),
        "num_workers": int(num_workers),
        "n_ic": int(n_ic),
    }


def _save_euler2d_quadrant_trajectory_pt(
    out_path: str,
    states: np.ndarray,
    meta: dict,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "states": torch.tensor(states, dtype=torch.float64),
            "meta": meta,
        },
        out_path,
    )


def _prepare_euler2d_quadrant_split_stream(
    *,
    out_dir: str,
    split_name: str,
    n_ic: int,
    seed: int,
    manifest_name: str | None = None,
    assemble_pt_name: str | None = None,
    nx_low: int = 64,
    ny_low: int = 64,
    upsample: int = 4,
    T: float = 0.5,
    dt_snap: float = 1e-2,
    cfl: float = 0.4,
    lx: float = 1.0,
    ly: float = 1.0,
    x0: float = 0.8,
    y0: float = 0.8,
    random_split: bool = False,
    split_margin: float = 0.30,
    random_quadrant_states: bool = True,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
    reconstruction: str = "characteristic",
    bc: str = "outflow",
    num_workers: int = 1,
) -> dict:
    """Prepare per-split output paths, metadata, and one job per trajectory."""
    num_workers = max(1, int(num_workers))
    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(out_dir, f"{split_name}_outflow_trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    if manifest_name is None:
        manifest_name = f"{split_name}_outflow_manifest.pt"
    manifest_path = os.path.join(out_dir, manifest_name)
    split_meta = _build_euler2d_quadrant_meta(
        n_ic=n_ic,
        nx_low=nx_low,
        ny_low=ny_low,
        upsample=upsample,
        T=T,
        dt_snap=dt_snap,
        cfl=cfl,
        lx=lx,
        ly=ly,
        x0=x0,
        y0=y0,
        random_split=random_split,
        split_margin=split_margin,
        random_quadrant_states=random_quadrant_states,
        rho_range=rho_range,
        p_range=p_range,
        u_range=u_range,
        v_range=v_range,
        reconstruction=reconstruction,
        bc=bc,
        num_workers=num_workers,
    )
    split_meta["split"] = split_name
    base_job = dict(
        nx_low=nx_low,
        ny_low=ny_low,
        upsample=upsample,
        T=T,
        dt_snap=dt_snap,
        cfl=cfl,
        lx=lx,
        ly=ly,
        x0=x0,
        y0=y0,
        random_split=random_split,
        split_margin=split_margin,
        random_quadrant_states=random_quadrant_states,
        rho_range=rho_range,
        p_range=p_range,
        u_range=u_range,
        v_range=v_range,
        reconstruction=reconstruction,
        bc=bc,
        show_progress=True,
        worker_name="",
    )
    jobs = []
    for t in range(n_ic):
        out_path = os.path.join(traj_dir, f"{split_name}_outflow_traj_{t:06d}.pt")
        jobs.append(
            {
                **base_job,
                "traj_idx": t,
                "n_ic_total": n_ic,
                "base_seed": seed,
                "out_path": out_path,
                "split_name": split_name,
                "split_meta": split_meta,
            }
        )
    return {
        "split_name": split_name,
        "n_ic": int(n_ic),
        "nx_low": int(nx_low),
        "ny_low": int(ny_low),
        "out_dir": out_dir,
        "traj_dir": traj_dir,
        "manifest_name": manifest_name,
        "manifest_path": manifest_path,
        "assemble_pt_name": assemble_pt_name,
        "split_meta": split_meta,
        "jobs": jobs,
        "rel_paths": [""] * n_ic,
        "n_snaps_per_traj": [0] * n_ic,
    }


def _finalize_euler2d_quadrant_split_stream(
    plan: dict,
    *,
    assemble_split_pt: bool = False,
) -> dict:
    """Write the split manifest and optionally assemble a monolithic split .pt."""
    split_name = str(plan["split_name"])
    out_dir = str(plan["out_dir"])
    traj_dir = str(plan["traj_dir"])
    split_meta = dict(plan["split_meta"])
    rel_paths = list(plan["rel_paths"])
    n_snaps_per_traj = list(plan["n_snaps_per_traj"])
    manifest = {
        "meta": {
            **split_meta,
            "storage": "per_trajectory_files",
            "manifest_version": 1,
            "trajectory_dir": os.path.relpath(traj_dir, out_dir),
            "dataset_layout": "(n_ic, n_snaps, 4, ny, nx)",
        },
        "trajectory_files": rel_paths,
        "trajectory_n_snaps": n_snaps_per_traj,
    }
    torch.save(manifest, str(plan["manifest_path"]))
    print(f"[Euler2D dataset] saved manifest -> {plan['manifest_path']}", flush=True)
    if assemble_split_pt:
        assemble_pt_name = plan.get("assemble_pt_name")
        if not assemble_pt_name:
            raise ValueError("assemble_split_pt=True requires assemble_pt_name.")
        out_path = os.path.join(out_dir, str(assemble_pt_name))
        print(
            f"[Euler2D dataset] assembling monolithic states split -> {out_path} "
            "(loads one trajectory at a time, but final tensor still occupies RAM)",
            flush=True,
        )
        n_ic = int(plan["n_ic"])
        nx_low = int(plan["nx_low"])
        ny_low = int(plan["ny_low"])
        n_snaps = int(split_meta["n_snaps"])
        states = torch.empty(
            (n_ic, n_snaps, 4, ny_low, nx_low),
            dtype=torch.float64,
        )
        for traj_idx, rel_path in enumerate(rel_paths):
            item = torch.load(os.path.join(out_dir, rel_path), map_location="cpu")
            states[traj_idx] = item["states"]
        data = {
            "states": states,
            "meta": {**split_meta, "split": split_name},
        }
        torch.save(data, out_path)
        print(f"[Euler2D dataset] saved monolithic split -> {out_path}", flush=True)
    return manifest


def _outflow_manifest_name_from_output_name(output_name: str) -> str:
    """Derive a manifest filename that always includes the outflow tag."""
    stem, _ext = os.path.splitext(str(output_name))
    if "outflow" in stem:
        return f"{stem}_manifest.pt"
    return f"{stem}_outflow_manifest.pt"


def _euler2d_quadrant_one_traj_job(job: dict) -> tuple[int, np.ndarray]:
    """
    ProcessPool entry: build exactly one trajectory. ``job`` must include
    ``traj_idx``, ``n_ic_total``, ``base_seed`` (removed before calling the chunk).
    Uses ``SeedSequence(base_seed).spawn(n_ic_total)[traj_idx]`` so parallel runs are
    reproducible and streams do not overlap (differs from one RNG advancing in a
    single-process multi-traj chunk).
    """
    traj_idx = int(job.pop("traj_idx"))
    n_ic_total = int(job.pop("n_ic_total"))
    base_seed = job.pop("base_seed")
    ss = _spawn_euler2d_quadrant_seed(base_seed, traj_idx, n_ic_total)
    job["n_ic"] = 1
    job["seed"] = ss
    if not job.get("worker_name"):
        job["worker_name"] = f"traj {traj_idx + 1}/{n_ic_total}"
    states = _build_euler2d_quadrant_chunk(**job)
    return traj_idx, states


def _euler2d_quadrant_one_traj_to_file_job(job: dict) -> tuple[int, str, int]:
    """ProcessPool entry: build one trajectory and save it immediately to disk."""
    traj_idx = int(job.pop("traj_idx"))
    n_ic_total = int(job.pop("n_ic_total"))
    base_seed = job.pop("base_seed")
    out_path = str(job.pop("out_path"))
    split_name = str(job.pop("split_name"))
    split_meta = dict(job.pop("split_meta"))
    ss = _spawn_euler2d_quadrant_seed(base_seed, traj_idx, n_ic_total)
    job["n_ic"] = 1
    job["seed"] = ss
    if not job.get("worker_name"):
        job["worker_name"] = f"traj {traj_idx + 1}/{n_ic_total}"
    states = _build_euler2d_quadrant_chunk(**job)
    traj_meta = dict(split_meta)
    traj_meta.update(
        {
            "split": split_name,
            "n_ic": 1,
            "trajectory_index": int(traj_idx),
            "trajectory_count_in_split": int(n_ic_total),
            "n_snaps": int(states.shape[0]),
            "seed_mode": "SeedSequence.spawn",
            "seed_index": int(traj_idx),
        }
    )
    _save_euler2d_quadrant_trajectory_pt(out_path, states, traj_meta)
    return traj_idx, out_path, int(states.shape[0])


def _build_euler2d_quadrant_chunk(
    n_ic: int,
    seed: int,
    nx_low: int = 64,
    ny_low: int = 64,
    upsample: int = 4,
    T: float = 0.5,
    dt_snap: float = 1e-2,
    cfl: float = 0.4,
    lx: float = 1.0,
    ly: float = 1.0,
    x0: float = 0.8,
    y0: float = 0.8,
    random_split: bool = False,
    split_margin: float = 0.30,
    random_quadrant_states: bool = True,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
    reconstruction: str = "characteristic",
    bc: str = "outflow",
    show_progress: bool = True,
    worker_name: str = "worker",
):
    """
    Fine-grid FD-WENOZ (outflow) + quadrant IC -> coarse trajectory states.

    Returns shape ``(n_ic, n_snaps, 4, ny_low, nx_low)``.
    """
    rng = np.random.default_rng(seed)
    nx_fine = nx_low * upsample
    ny_fine = ny_low * upsample
    dx_fine, dy_fine = make_quadrant_grid(nx_fine, ny_fine, lx=lx, ly=ly)

    n_snaps = int(round(T / dt_snap)) + 1
    if n_snaps < 2:
        raise ValueError(f"Need T/dt_snap >= 1 for at least one pair; got T={T}, dt_snap={dt_snap}")

    states_all = np.zeros((n_ic, n_snaps, 4, ny_low, nx_low), dtype=np.float64)

    op = build_euler2d_solver(
        gamma=GAMMA,
        bc=bc,
        reconstruction=reconstruction,
        verbose=False,
    )

    if show_progress:
        print(
            f"[Euler2D dataset][{worker_name}] chunk: n_ic={n_ic}, nx_fine={nx_fine}, ny_fine={ny_fine}, "
            f"n_snaps={n_snaps}, dt_snap={dt_snap:g}, bc={bc}",
            flush=True,
        )

    for s in range(n_ic):
        xs0, ys0 = float(x0), float(y0)
        if random_split:
            lo_x = split_margin * lx
            hi_x = (1.0 - split_margin) * lx
            lo_y = split_margin * ly
            hi_y = (1.0 - split_margin) * ly
            if hi_x <= lo_x or hi_y <= lo_y:
                raise ValueError(f"Invalid split_margin={split_margin} for domain lx={lx}, ly={ly}.")
            xs0 = float(rng.uniform(lo_x, hi_x))
            ys0 = float(rng.uniform(lo_y, hi_y))

        if random_quadrant_states:
            U1, U2, U3, U4 = sample_four_quadrant_states(
                rng,
                rho_range=rho_range,
                p_range=p_range,
                u_range=u_range,
                v_range=v_range,
            )
            u0_fine, _, _ = ic_quadrant_piecewise(
                nx_fine,
                ny_fine,
                U1,
                U2,
                U3,
                U4,
                xlim=(0.0, lx),
                ylim=(0.0, ly),
                x0=xs0,
                y0=ys0,
            )
        else:
            u0_fine, _, _ = ic_quadrant_riemann(
                nx_fine, ny_fine, xlim=(0.0, lx), ylim=(0.0, ly), x0=xs0, y0=ys0
            )
        u0_fine = enforce_physical_euler2d(u0_fine)
        state_fine = np.asarray(u0_fine, dtype=np.float64).copy()

        if show_progress:
            rq = "random U1–U4" if random_quadrant_states else "classic U1–U4"
            print(
                f"[Euler2D dataset][{worker_name}] traj {s + 1}/{n_ic}: x0={xs0:.4g}, y0={ys0:.4g} ({rq})",
                flush=True,
            )

        # Advance one ``dt_snap`` segment at a time and emit each coarse pair immediately.
        # This avoids storing the full fine-grid trajectory in memory while still
        # producing the full coarse trajectory.
        u_prev = np.asarray(state_fine, dtype=np.float64).copy()
        states_all[s, 0] = downsample_conserved2d(u_prev, upsample, upsample)
        for k in range(n_snaps - 1):
            seg = op.solve_snapshots(
                u_prev,
                dx_fine,
                dy_fine,
                dt_snap=dt_snap,
                n_snaps=2,
                cfl=cfl,
                log_ic_state=False,
                time_offset=float(k) * float(dt_snap),
            )
            u_next = enforce_physical_euler2d(np.asarray(seg[-1], dtype=np.float64).copy())
            states_all[s, k + 1] = downsample_conserved2d(u_next, upsample, upsample)
            u_prev = u_next
            if show_progress:
                t_done = (k + 1) * dt_snap
                print(
                    f"[Euler2D dataset][{worker_name}] traj {s + 1}/{n_ic}  "
                    f"t={t_done:g}  (interval {k + 1}/{n_snaps - 1}, dt_snap={dt_snap:g})",
                    flush=True,
                )

    return states_all


def build_euler2d_quadrant_split_to_dir(
    *,
    out_dir: str,
    split_name: str,
    n_ic: int,
    seed: int,
    manifest_name: str | None = None,
    assemble_pt_name: str | None = None,
    nx_low: int = 64,
    ny_low: int = 64,
    upsample: int = 4,
    T: float = 0.5,
    dt_snap: float = 1e-2,
    cfl: float = 0.4,
    lx: float = 1.0,
    ly: float = 1.0,
    x0: float = 0.8,
    y0: float = 0.8,
    random_split: bool = False,
    split_margin: float = 0.30,
    random_quadrant_states: bool = True,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
    reconstruction: str = "characteristic",
    bc: str = "outflow",
    num_workers: int = 1,
    assemble_split_pt: bool = False,
) -> dict:
    """
    Stream one trajectory at a time to ``out_dir``.

    Always writes per-trajectory ``.pt`` files and a split manifest. Optionally also
    assembles a monolithic split ``.pt`` with ``states`` of shape
    ``(n_ic, n_snaps, 4, ny, nx)``.
    """
    num_workers = max(1, int(num_workers))
    plan = _prepare_euler2d_quadrant_split_stream(
        out_dir=out_dir,
        split_name=split_name,
        n_ic=n_ic,
        seed=seed,
        manifest_name=manifest_name,
        assemble_pt_name=assemble_pt_name,
        nx_low=nx_low,
        ny_low=ny_low,
        upsample=upsample,
        T=T,
        dt_snap=dt_snap,
        cfl=cfl,
        lx=lx,
        ly=ly,
        x0=x0,
        y0=y0,
        random_split=random_split,
        split_margin=split_margin,
        random_quadrant_states=random_quadrant_states,
        rho_range=rho_range,
        p_range=p_range,
        u_range=u_range,
        v_range=v_range,
        reconstruction=reconstruction,
        bc=bc,
        num_workers=num_workers,
    )
    print(
        "[Euler2D dataset] stream "
        f"split={split_name}, n_ic={n_ic}, nx_low={nx_low}, ny_low={ny_low}, upsample={upsample}, "
        f"T={T}, dt_snap={dt_snap}, bc={bc}, recon={reconstruction}, workers={num_workers}",
        flush=True,
    )
    if n_ic <= 0:
        return _finalize_euler2d_quadrant_split_stream(plan, assemble_split_pt=False)
    jobs = plan["jobs"]
    if num_workers == 1 or n_ic == 1:
        for job in jobs:
            traj_idx, out_path, n_snaps_traj = _euler2d_quadrant_one_traj_to_file_job(dict(job))
            plan["rel_paths"][traj_idx] = os.path.relpath(out_path, out_dir)
            plan["n_snaps_per_traj"][traj_idx] = n_snaps_traj
            print(
                f"[Euler2D dataset] saved trajectory {traj_idx + 1}/{n_ic} -> {plan['rel_paths'][traj_idx]}",
                flush=True,
            )
    else:
        pool_workers = min(num_workers, n_ic)
        print(
            f"[Euler2D dataset] pool: {n_ic} trajectories, {pool_workers} worker process(es) "
            "(dynamic: idle workers pick next trajectory; each worker saves directly to disk)",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=pool_workers) as ex:
            futures = [ex.submit(_euler2d_quadrant_one_traj_to_file_job, j) for j in jobs]
            n_done = 0
            for fut in as_completed(futures):
                traj_idx, out_path, n_snaps_traj = fut.result()
                plan["rel_paths"][traj_idx] = os.path.relpath(out_path, out_dir)
                plan["n_snaps_per_traj"][traj_idx] = n_snaps_traj
                n_done += 1
                print(
                    f"[Euler2D dataset] progress: {n_done}/{n_ic} trajectories saved "
                    f"({plan['rel_paths'][traj_idx]})",
                    flush=True,
                )
    return _finalize_euler2d_quadrant_split_stream(plan, assemble_split_pt=assemble_split_pt)


def build_euler2d_quadrant_splits_to_dir(
    *,
    out_dir: str,
    split_specs: list[tuple[str, int, int, str]],
    nx_low: int = 64,
    ny_low: int = 64,
    upsample: int = 4,
    T: float = 0.5,
    dt_snap: float = 1e-2,
    cfl: float = 0.4,
    lx: float = 1.0,
    ly: float = 1.0,
    x0: float = 0.8,
    y0: float = 0.8,
    random_split: bool = False,
    split_margin: float = 0.30,
    random_quadrant_states: bool = True,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
    reconstruction: str = "characteristic",
    bc: str = "outflow",
    num_workers: int = 1,
    assemble_split_pt: bool = False,
) -> dict[str, dict]:
    """
    Build all requested splits with one global trajectory queue, then finalize each split.

    This keeps the worker pool busy across train/val/test instead of waiting for one split to
    finish before starting the next.
    """
    num_workers = max(1, int(num_workers))
    os.makedirs(out_dir, exist_ok=True)
    plans: list[dict] = []
    for split_name, n_ic, seed, fname in split_specs:
        if n_ic <= 0:
            continue
        manifest_name = _outflow_manifest_name_from_output_name(fname)
        plan = _prepare_euler2d_quadrant_split_stream(
            out_dir=out_dir,
            split_name=split_name,
            n_ic=n_ic,
            seed=seed,
            manifest_name=manifest_name,
            assemble_pt_name=fname,
            nx_low=nx_low,
            ny_low=ny_low,
            upsample=upsample,
            T=T,
            dt_snap=dt_snap,
            cfl=cfl,
            lx=lx,
            ly=ly,
            x0=x0,
            y0=y0,
            random_split=random_split,
            split_margin=split_margin,
            random_quadrant_states=random_quadrant_states,
            rho_range=rho_range,
            p_range=p_range,
            u_range=u_range,
            v_range=v_range,
            reconstruction=reconstruction,
            bc=bc,
            num_workers=num_workers,
        )
        plans.append(plan)
        print(
            "[Euler2D dataset] stream "
            f"split={split_name}, n_ic={n_ic}, nx_low={nx_low}, ny_low={ny_low}, upsample={upsample}, "
            f"T={T}, dt_snap={dt_snap}, bc={bc}, recon={reconstruction}, workers={num_workers}",
            flush=True,
        )
    if not plans:
        return {}

    total_jobs = sum(len(plan["jobs"]) for plan in plans)
    if num_workers == 1 or total_jobs == 1:
        for plan in plans:
            split_name = str(plan["split_name"])
            n_ic = int(plan["n_ic"])
            for job in plan["jobs"]:
                traj_idx, out_path, n_snaps_traj = _euler2d_quadrant_one_traj_to_file_job(dict(job))
                plan["rel_paths"][traj_idx] = os.path.relpath(out_path, out_dir)
                plan["n_snaps_per_traj"][traj_idx] = n_snaps_traj
                print(
                    f"[Euler2D dataset] split={split_name} saved trajectory {traj_idx + 1}/{n_ic} "
                    f"-> {plan['rel_paths'][traj_idx]}",
                    flush=True,
                )
    else:
        pool_workers = min(num_workers, total_jobs)
        print(
            f"[Euler2D dataset] global pool: {total_jobs} trajectories across {len(plans)} split(s), "
            f"{pool_workers} worker process(es) "
            "(dynamic: idle workers pick the next train/val/test trajectory)",
            flush=True,
        )
        future_to_plan_idx: dict = {}
        done_per_split = [0] * len(plans)
        with ProcessPoolExecutor(max_workers=pool_workers) as ex:
            for plan_idx, plan in enumerate(plans):
                for job in plan["jobs"]:
                    future = ex.submit(_euler2d_quadrant_one_traj_to_file_job, dict(job))
                    future_to_plan_idx[future] = plan_idx
            n_done_total = 0
            for fut in as_completed(future_to_plan_idx):
                plan_idx = int(future_to_plan_idx[fut])
                plan = plans[plan_idx]
                split_name = str(plan["split_name"])
                traj_idx, out_path, n_snaps_traj = fut.result()
                plan["rel_paths"][traj_idx] = os.path.relpath(out_path, out_dir)
                plan["n_snaps_per_traj"][traj_idx] = n_snaps_traj
                done_per_split[plan_idx] += 1
                n_done_total += 1
                print(
                    f"[Euler2D dataset] progress: total {n_done_total}/{total_jobs}, "
                    f"split {split_name} {done_per_split[plan_idx]}/{plan['n_ic']} "
                    f"({plan['rel_paths'][traj_idx]})",
                    flush=True,
                )

    manifests: dict[str, dict] = {}
    for plan in plans:
        split_name = str(plan["split_name"])
        manifests[split_name] = _finalize_euler2d_quadrant_split_stream(
            plan,
            assemble_split_pt=assemble_split_pt,
        )
    return manifests


def build_euler2d_quadrant_split(
    n_ic: int,
    seed: int,
    nx_low: int = 64,
    ny_low: int = 64,
    upsample: int = 4,
    T: float = 0.5,
    dt_snap: float = 1e-2,
    cfl: float = 0.4,
    lx: float = 1.0,
    ly: float = 1.0,
    x0: float = 0.8,
    y0: float = 0.8,
    random_split: bool = False,
    split_margin: float = 0.30,
    random_quadrant_states: bool = True,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
    reconstruction: str = "characteristic",
    bc: str = "outflow",
    num_workers: int = 1,
):
    num_workers = max(1, int(num_workers))
    print(
        "[Euler2D dataset] start "
        f"n_ic={n_ic}, nx_low={nx_low}, ny_low={ny_low}, upsample={upsample}, "
        f"T={T}, dt_snap={dt_snap}, bc={bc}, recon={reconstruction}, "
        f"random_quadrant_states={random_quadrant_states}, workers={num_workers}",
        flush=True,
    )

    if num_workers == 1 or n_ic <= 1:
        states_all = _build_euler2d_quadrant_chunk(
            n_ic=n_ic,
            seed=seed,
            nx_low=nx_low,
            ny_low=ny_low,
            upsample=upsample,
            T=T,
            dt_snap=dt_snap,
            cfl=cfl,
            lx=lx,
            ly=ly,
            x0=x0,
            y0=y0,
            random_split=random_split,
            split_margin=split_margin,
            random_quadrant_states=random_quadrant_states,
            rho_range=rho_range,
            p_range=p_range,
            u_range=u_range,
            v_range=v_range,
            reconstruction=reconstruction,
            bc=bc,
            show_progress=True,
            worker_name="main",
        )
    else:
        pool_workers = min(num_workers, n_ic)
        print(
            f"[Euler2D dataset] pool: {n_ic} trajectories, {pool_workers} worker process(es) "
            "(dynamic: idle workers pick next trajectory)",
            flush=True,
        )
        base_job = dict(
            nx_low=nx_low,
            ny_low=ny_low,
            upsample=upsample,
            T=T,
            dt_snap=dt_snap,
            cfl=cfl,
            lx=lx,
            ly=ly,
            x0=x0,
            y0=y0,
            random_split=random_split,
            split_margin=split_margin,
            random_quadrant_states=random_quadrant_states,
            rho_range=rho_range,
            p_range=p_range,
            u_range=u_range,
            v_range=v_range,
            reconstruction=reconstruction,
            bc=bc,
            show_progress=True,
            worker_name="",
        )
        jobs = [
            {**base_job, "traj_idx": t, "n_ic_total": n_ic, "base_seed": seed} for t in range(n_ic)
        ]
        slots: list[np.ndarray | None] = [None] * n_ic
        n_done = 0
        with ProcessPoolExecutor(max_workers=pool_workers) as ex:
            futures = [ex.submit(_euler2d_quadrant_one_traj_job, j) for j in jobs]
            for fut in as_completed(futures):
                traj_idx, states = fut.result()
                slots[traj_idx] = states[0]
                n_done += 1
                print(
                    f"[Euler2D dataset] progress: {n_done}/{n_ic} trajectories complete",
                    flush=True,
                )
        assert all(s is not None for s in slots)
        states_all = np.stack([slots[i] for i in range(n_ic)], axis=0)

    data = {
        "states": torch.tensor(states_all, dtype=torch.float64),
        "meta": _build_euler2d_quadrant_meta(
            n_ic=n_ic,
            nx_low=nx_low,
            ny_low=ny_low,
            upsample=upsample,
            T=T,
            dt_snap=dt_snap,
            cfl=cfl,
            lx=lx,
            ly=ly,
            x0=x0,
            y0=y0,
            random_split=random_split,
            split_margin=split_margin,
            random_quadrant_states=random_quadrant_states,
            rho_range=rho_range,
            p_range=p_range,
            u_range=u_range,
            v_range=v_range,
            reconstruction=reconstruction,
            bc=bc,
            num_workers=num_workers,
        ),
    }
    return data


def _field_imshow_limits(
    field: np.ndarray,
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
) -> tuple[float, float]:
    """Robust vmin/vmax from percentiles so small-scale structure is visible."""
    r = np.asarray(field, dtype=np.float64).ravel()
    lo = float(np.percentile(r, pct_lo))
    hi = float(np.percentile(r, pct_hi))
    if hi <= lo + 1e-30:
        lo, hi = float(r.min()), float(r.max())
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def run_quadrant_outflow_demo(
    nx: int = 128,
    ny: int = 128,
    dt_snap: float = 1e-2,
    t_final: float = 1.0,
    cfl: float = 0.4,
    outdir: str | None = None,
    reconstruction: str = "characteristic",
    verbose: bool = True,
    x0: float = 0.8,
    y0: float = 0.8,
    random_quadrant_ic: bool = False,
    ic_seed: int | None = None,
    random_split: bool = False,
    split_margin: float = 0.30,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    u_range: tuple[float, float] = DEFAULT_UV_RANGE,
    v_range: tuple[float, float] | None = None,
    *,
    plot_cmap: str = "turbo",
    plot_pct_lo: float = 1.0,
    plot_pct_hi: float = 99.0,
    plot_global_scale: bool = False,
    plot_dpi: int = 200,
    plot_avg_factor: int = 1,
    clean_export: bool = False,
):
    """
    Test: [0,1]^2 outflow, quadrant IC, save state plots every ``dt_snap`` until ``t_final``.

    With ``clean_export=True``, the default output directory is ``quadrant_outflow_demo_clean``
    (unless ``outdir`` is set explicitly). Each frame is saved as both **PDF** and **PNG** with
    no axes, colorbar, or title—only the rho colormap image.

    Plotting uses percentile-based ``vmin/vmax`` (default 1–99%%) to improve contrast;
    set ``plot_global_scale=True`` to use one scale from all snapshots (fair comparison
    across time, less local contrast). With global scale the full trajectory is computed
    first, then PDFs are written (color limits need all frames).

    By default, figures are saved as **PDF** 2x2 panels for primitive variables
    ``(rho, u, v, p)`` (vector axes, colorbar, text; the colormap fields may be rasterized
    inside the PDF). With ``clean_export``, each frame is **PDF + PNG** with a minimal
    rho-only layout. Each frame is written as soon as it is available when
    ``plot_global_scale=False``.

    ``plot_avg_factor>1``: conservative ``downsample_cell_average2d`` before ``imshow``
    (e.g. factor ``2`` maps 512×512 data to 256×256 for plotting only; solve grid unchanged).

    Set ``random_quadrant_ic=True`` to sample four independent quadrant states (same sampler as
    the dataset). ``ic_seed=None`` uses a nondeterministic RNG; pass an int for reproducibility.
    IC domain is ``[0,1]^2`` (same as classic demo).

    ``random_split=True`` samples ``(x0,y0)`` uniformly in
    ``[split_margin, 1-split_margin]^2``. When ``random_quadrant_ic=True``, this is **on by
    default** from the CLI unless ``--fix_xy`` is passed (see demo argparse).
    """
    paf = int(plot_avg_factor)
    if paf < 1:
        raise ValueError("plot_avg_factor must be >= 1 (1 = no plot downsampling).")

    n_snaps = int(round(t_final / dt_snap)) + 1
    xs0, ys0 = float(x0), float(y0)
    need_rng = bool(random_quadrant_ic or random_split)
    rng = None
    if need_rng:
        rng = np.random.default_rng(ic_seed) if ic_seed is not None else np.random.default_rng()
    if random_split:
        lo = float(split_margin)
        hi = 1.0 - float(split_margin)
        if hi <= lo:
            raise ValueError(f"split_margin={split_margin} leaves empty interior on [0,1]^2.")
        xs0 = float(rng.uniform(lo, hi))
        ys0 = float(rng.uniform(lo, hi))
    if random_quadrant_ic:
        if rng is None:
            rng = np.random.default_rng(ic_seed) if ic_seed is not None else np.random.default_rng()
        U1, U2, U3, U4 = sample_four_quadrant_states(
            rng,
            rho_range=rho_range,
            p_range=p_range,
            u_range=u_range,
            v_range=v_range,
        )
        u0, dx, dy = ic_quadrant_piecewise(
            nx,
            ny,
            U1,
            U2,
            U3,
            U4,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            x0=xs0,
            y0=ys0,
        )
    else:
        u0, dx, dy = ic_quadrant_riemann(nx, ny, x0=xs0, y0=ys0)
    if outdir is None:
        default_sub = (
            "quadrant_outflow_demo_clean" if clean_export else "quadrant_outflow_demo_out_back"
        )
        if need_rng and ic_seed is not None:
            default_sub = f"{default_sub}_seed{ic_seed}"
        outdir = os.path.join(os.path.dirname(__file__), default_sub)
    os.makedirs(outdir, exist_ok=True)
    if paf > 1 and (ny % paf != 0 or nx % paf != 0):
        raise ValueError(
            f"plot_avg_factor={paf} requires ny,nx divisible by paf; got ny={ny}, nx={nx}."
        )
    op = build_euler2d_solver(bc="outflow", reconstruction=reconstruction, verbose=verbose)
    u_note = "random U1–U4" if random_quadrant_ic else "classic U1–U4"
    xy_note = f"random (x0,y0)" if random_split else f"fixed (x0,y0)=({xs0:.4g},{ys0:.4g})"
    ic_note = f"{u_note}, {xy_note}"
    seed_note = ""
    if need_rng and ic_seed is not None:
        seed_note = f", ic_seed={ic_seed}"
    elif need_rng:
        seed_note = ", ic_seed=nondeterministic"
    print(
        f"[Euler2D demo] nx={nx}, ny={ny}, dx={dx:.5g}, dy={dy:.5g}, "
        f"dt_snap={dt_snap}, n_snaps={n_snaps}, bc=outflow, recon={reconstruction}, IC={ic_note}{seed_note}"
    )
    times = np.arange(n_snaps, dtype=np.float64) * dt_snap

    def _component_for_plot(field_raw: np.ndarray) -> np.ndarray:
        r = np.asarray(field_raw, dtype=np.float64)
        if paf == 1:
            return r
        return downsample_cell_average2d(r, paf, paf)

    def _primitive_state_for_plot(state_raw: np.ndarray) -> np.ndarray:
        state = np.asarray(state_raw, dtype=np.float64)
        rho = np.maximum(state[0], RHO_MIN)
        u_vel = state[1] / rho
        v_vel = state[2] / rho
        p = pressure_conserved(state)
        return np.stack(
            [
                _component_for_plot(rho),
                _component_for_plot(u_vel),
                _component_for_plot(v_vel),
                _component_for_plot(p),
            ],
            axis=0,
        )

    def _save_state_pdf(
        state_field: np.ndarray,
        t_phys: float,
        *,
        use_global_limits: bool,
        global_limits: list[tuple[float, float]] | None,
    ) -> None:
        state_plot = _primitive_state_for_plot(state_field)
        t_tag = f"{t_phys:.2f}".replace(".", "p")

        if clean_export:
            rho = state_plot[0]
            if use_global_limits and global_limits is not None:
                vmin, vmax = global_limits[0]
            else:
                vmin, vmax = _field_imshow_limits(rho, plot_pct_lo, plot_pct_hi)
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            fig = plt.figure(figsize=(6.0, 5.0), frameon=False)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.imshow(
                rho,
                origin="lower",
                extent=[0, 1, 0, 1],
                aspect="equal",
                cmap=plot_cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax.set_axis_off()
            for ext in ("pdf", "png"):
                fname = os.path.join(outdir, f"state2x2_rho_t{t_tag}.{ext}")
                fig.savefig(
                    fname,
                    format=ext,
                    dpi=int(plot_dpi),
                    bbox_inches="tight",
                    pad_inches=0,
                )
                print(f"  saved {fname}")
            plt.close(fig)
            return

        labels = ("rho", "u", "v", "p")
        titles = ("rho", "u", "v", "p")
        ny_p, nx_p = state_plot[0].shape
        ds_note = f", plot {ny_p}×{nx_p} (avg×{paf})" if paf > 1 else ""
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True)
        fig.suptitle(
            f"Euler2D primitive state at t = {t_phys:.2f}  "
            f"(color: {plot_pct_lo:g}-{plot_pct_hi:g}% {'global' if use_global_limits else 'frame'} scale{ds_note})"
        )
        for c, ax in enumerate(axes.flat):
            field = state_plot[c]
            if use_global_limits and global_limits is not None:
                vmin, vmax = global_limits[c]
            else:
                vmin, vmax = _field_imshow_limits(field, plot_pct_lo, plot_pct_hi)
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            im = ax.imshow(
                field,
                origin="lower",
                extent=[0, 1, 0, 1],
                aspect="equal",
                cmap=plot_cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax.set_title(titles[c])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(labels[c])
        fname = os.path.join(outdir, f"state2x2_t{t_tag}.pdf")
        fig.savefig(fname, format="pdf", dpi=int(plot_dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    if plot_global_scale:
        snaps = op.solve_snapshots(u0, dx, dy, dt_snap=dt_snap, n_snaps=n_snaps, cfl=cfl)
        state_ds = [_primitive_state_for_plot(snaps[i]) for i in range(n_snaps)]
        global_limits = []
        for c in range(4):
            field_all = np.concatenate([x[c].ravel() for x in state_ds])
            g_lo = float(np.percentile(field_all, plot_pct_lo))
            g_hi = float(np.percentile(field_all, plot_pct_hi))
            if g_hi <= g_lo + 1e-30:
                g_lo, g_hi = float(field_all.min()), float(field_all.max())
                if g_hi <= g_lo:
                    g_hi = g_lo + 1.0
            global_limits.append((g_lo, g_hi))
        for k in range(n_snaps):
            _save_state_pdf(
                snaps[k],
                float(times[k]),
                use_global_limits=True,
                global_limits=global_limits,
            )
    else:
        u_cur = np.asarray(u0, dtype=np.float64).copy()
        traj_list: list[np.ndarray] = [u_cur.copy()]
        _save_state_pdf(
            u_cur,
            float(times[0]),
            use_global_limits=False,
            global_limits=None,
        )
        for k in range(1, n_snaps):
            seg = op.solve_snapshots(
                u_cur,
                dx,
                dy,
                dt_snap=dt_snap,
                n_snaps=2,
                cfl=cfl,
                log_ic_state=False,
                time_offset=float(times[k - 1]),
            )
            u_cur = seg[-1]
            traj_list.append(u_cur.copy())
            _save_state_pdf(
                u_cur,
                float(times[k]),
                use_global_limits=False,
                global_limits=None,
            )
        snaps = np.stack(traj_list, axis=0)

    meta_path = os.path.join(outdir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"nx={nx} ny={ny} dt_snap={dt_snap} t_final={t_final} cfl={cfl}\n")
        f.write(f"x0={xs0} y0={ys0}\n")
        f.write(f"random_quadrant_ic={bool(random_quadrant_ic)}\n")
        f.write(f"random_split={bool(random_split)} split_margin={split_margin}\n")
        if need_rng:
            f.write(f"ic_seed={ic_seed!r}\n")
        if random_quadrant_ic:
            f.write(f"rho_range={list(rho_range)} p_range={list(p_range)} u_range={list(u_range)}\n")
            f.write(
                f"v_range={None if v_range is None else list(v_range)}\n"
            )
        f.write(f"reconstruction={reconstruction}\n")
        f.write(f"plot_avg_factor={paf}\n")
        f.write(f"plot_format={'pdf+png' if clean_export else 'pdf'}\n")
        f.write(f"clean_export={bool(clean_export)}\n")
        f.write(f"plot_global_scale={bool(plot_global_scale)}\n")
    print(f"[Euler2D demo] done. Plots in {outdir}")
    return snaps, times


def _main_dataset_cli():
    ap = argparse.ArgumentParser(
        description="Build Euler2D quadrant-Riemann + outflow datasets (stream per trajectory by default).",
        epilog="Figure demo: python -m Euler2D.dataset.data --demo [options]",
    )
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_val", type=int, default=2)
    ap.add_argument("--n_test", type=int, default=2)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--dt_snap", type=float, default=1e-2)
    ap.add_argument("--nx_low", type=int, default=64)
    ap.add_argument("--ny_low", type=int, default=64)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--lx", type=float, default=1.0)
    ap.add_argument("--ly", type=float, default=1.0)
    ap.add_argument(
        "--x0",
        type=float,
        default=0.8,
        help="Riemann split x when split is fixed (see --fix_xy / --fixed_quadrant_ic)",
    )
    ap.add_argument(
        "--y0",
        type=float,
        default=0.8,
        help="Riemann split y when split is fixed",
    )
    ap.add_argument(
        "--random_split",
        action="store_true",
        help="force random (x0,y0) each trajectory (default ON when using random quadrant U; use --fix_xy to pin)",
    )
    ap.add_argument(
        "--fix_xy",
        action="store_true",
        help="pin (x0,y0) to --x0/--y0 (default when using random quadrant U: split is random unless this flag)",
    )
    ap.add_argument(
        "--split_margin",
        type=float,
        default=0.30,
        help="fraction of domain width/height kept as margin when --random_split",
    )
    ap.add_argument(
        "--fixed_quadrant_ic",
        action="store_true",
        help="use classic fixed U1–U4 (benchmark) instead of random four-quadrant states",
    )
    ap.add_argument(
        "--rho_lo",
        type=float,
        default=DEFAULT_RHO_RANGE[0],
        help="log-uniform rho lower bound (default: mild ICs)",
    )
    ap.add_argument("--rho_hi", type=float, default=DEFAULT_RHO_RANGE[1], help="log-uniform rho upper bound")
    ap.add_argument(
        "--p_lo",
        type=float,
        default=DEFAULT_P_RANGE[0],
        help="log-uniform p lower bound (default: mild ICs)",
    )
    ap.add_argument("--p_hi", type=float, default=DEFAULT_P_RANGE[1], help="log-uniform p upper bound")
    ap.add_argument(
        "--u_lo",
        type=float,
        default=DEFAULT_UV_RANGE[0],
        help="uniform u lower bound (default: mild ICs)",
    )
    ap.add_argument("--u_hi", type=float, default=DEFAULT_UV_RANGE[1], help="uniform u upper bound")
    ap.add_argument(
        "--v_lo",
        type=float,
        default=None,
        help="uniform v lower bound (default: same as u)",
    )
    ap.add_argument("--v_hi", type=float, default=None, help="uniform v upper bound (default: same as u)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num_workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--component", action="store_true", help="component-wise WENO instead of characteristic")
    ap.add_argument(
        "--assemble_split_pt",
        action="store_true",
        help="also assemble legacy monolithic split .pt files after per-trajectory streaming (RAM-heavy)",
    )
    ap.add_argument("--train_name", type=str, default="euler2d_quadrant_outflow_train_1e-2.pt")
    ap.add_argument("--val_name", type=str, default="euler2d_quadrant_outflow_val_1e-2.pt")
    ap.add_argument("--test_name", type=str, default="euler2d_quadrant_outflow_test_1e-2.pt")
    args = ap.parse_args()
    if (args.v_lo is None) ^ (args.v_hi is None):
        raise SystemExit("Euler2D dataset: pass both --v_lo and --v_hi, or neither (defaults v to same range as u).")
    if args.fix_xy and args.random_split:
        raise SystemExit("Euler2D dataset: use only one of --fix_xy or --random_split.")

    if args.fixed_quadrant_ic:
        random_split_eff = bool(args.random_split)
    else:
        # Random quadrant U: random (x0,y0) by default; --fix_xy keeps --x0/--y0.
        if args.fix_xy:
            random_split_eff = False
        else:
            random_split_eff = True

    split_specs = [
        ("train", args.n_train, args.seed, args.train_name),
        ("val", args.n_val, args.seed + 1000, args.val_name),
        ("test", args.n_test, args.seed + 2000, args.test_name),
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    rho_range = (args.rho_lo, args.rho_hi)
    p_range = (args.p_lo, args.p_hi)
    u_range = (args.u_lo, args.u_hi)
    if args.v_lo is None or args.v_hi is None:
        v_range = None
    else:
        v_range = (args.v_lo, args.v_hi)
    manifests = build_euler2d_quadrant_splits_to_dir(
        out_dir=args.out_dir,
        split_specs=split_specs,
        nx_low=args.nx_low,
        ny_low=args.ny_low,
        upsample=args.upsample,
        T=args.T,
        dt_snap=args.dt_snap,
        cfl=args.cfl,
        lx=args.lx,
        ly=args.ly,
        x0=args.x0,
        y0=args.y0,
        random_split=random_split_eff,
        split_margin=args.split_margin,
        random_quadrant_states=not args.fixed_quadrant_ic,
        rho_range=rho_range,
        p_range=p_range,
        u_range=u_range,
        v_range=v_range,
        reconstruction="component" if args.component else "characteristic",
        bc="outflow",
        num_workers=args.num_workers,
        assemble_split_pt=bool(args.assemble_split_pt),
    )
    for split_name, n_ic, _seed, fname in split_specs:
        if n_ic <= 0:
            continue
        manifest_name = _outflow_manifest_name_from_output_name(fname)
        manifest = manifests[split_name]
        print(
            f"[Euler2D dataset] split={split_name} done: manifest="
            f"{os.path.join(args.out_dir, manifest_name)}  "
            f"n_trajectories={len(manifest['trajectory_files'])}",
            flush=True,
        )


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--demo":
        argv = argv[1:]
        ap_demo = argparse.ArgumentParser(description="2D Euler quadrant Riemann + outflow demo")
        ap_demo.add_argument("--nx", type=int, default=128)
        ap_demo.add_argument("--ny", type=int, default=128)
        ap_demo.add_argument("--dt_snap", type=float, default=1e-2)
        ap_demo.add_argument("--t_final", type=float, default=1.0)
        ap_demo.add_argument("--cfl", type=float, default=0.4)
        ap_demo.add_argument("--outdir", type=str, default="")
        ap_demo.add_argument("--x0", type=float, default=0.8)
        ap_demo.add_argument("--y0", type=float, default=0.8)
        ap_demo.add_argument(
            "--random_ic",
            action="store_true",
            help="random four-quadrant primitive states (same sampler as dataset; default IC is classic benchmark)",
        )
        ap_demo.add_argument(
            "--ic_seed",
            type=int,
            default=None,
            help="RNG seed for random IC (omit for nondeterministic); same stream for (x0,y0) and U1–U4 when both random",
        )
        ap_demo.add_argument(
            "--fix_xy",
            action="store_true",
            help="with --random_ic: keep --x0/--y0 instead of randomizing split (default: random split when --random_ic)",
        )
        ap_demo.add_argument(
            "--random_split",
            action="store_true",
            help="randomize (x0,y0) in [margin,1-margin]^2 (also implied by --random_ic unless --fix_xy)",
        )
        ap_demo.add_argument(
            "--split_margin",
            type=float,
            default=0.30,
            help="interior margin for random (x0,y0) (demo domain [0,1]^2)",
        )
        ap_demo.add_argument(
            "--rho_lo",
            type=float,
            default=DEFAULT_RHO_RANGE[0],
            help="with --random_ic: log-uniform rho lower",
        )
        ap_demo.add_argument("--rho_hi", type=float, default=DEFAULT_RHO_RANGE[1], help="with --random_ic: rho upper")
        ap_demo.add_argument(
            "--p_lo",
            type=float,
            default=DEFAULT_P_RANGE[0],
            help="with --random_ic: log-uniform p lower",
        )
        ap_demo.add_argument("--p_hi", type=float, default=DEFAULT_P_RANGE[1], help="with --random_ic: p upper")
        ap_demo.add_argument("--u_lo", type=float, default=DEFAULT_UV_RANGE[0], help="with --random_ic: u lower")
        ap_demo.add_argument("--u_hi", type=float, default=DEFAULT_UV_RANGE[1], help="with --random_ic: u upper")
        ap_demo.add_argument(
            "--v_lo",
            type=float,
            default=None,
            help="with --random_ic: v lower (default: same as u; set both v_lo and v_hi)",
        )
        ap_demo.add_argument("--v_hi", type=float, default=None, help="with --random_ic: v upper")
        ap_demo.add_argument("--component", action="store_true", help="component-wise WENO instead of characteristic")
        ap_demo.add_argument("--quiet", action="store_true", help="no per-step prints from the solver")
        ap_demo.add_argument("--plot_cmap", type=str, default="turbo", help="matplotlib colormap for rho")
        ap_demo.add_argument("--plot_pct_lo", type=float, default=1.0, help="percentile for vmin (contrast)")
        ap_demo.add_argument("--plot_pct_hi", type=float, default=99.0, help="percentile for vmax (contrast)")
        ap_demo.add_argument(
            "--plot_global_scale",
            action="store_true",
            help="one vmin/vmax from all frames (comparable across t; less local contrast)",
        )
        ap_demo.add_argument("--plot_dpi", type=int, default=200)
        ap_demo.add_argument(
            "--plot_avg_factor",
            type=int,
            default=1,
            help="conservative 2D block average before plotting (2 => 512×512 → 256×256)",
        )
        ap_demo.add_argument(
            "--clean_export",
            action="store_true",
            help=(
                "save each frame as PDF and PNG with no axes/colorbar/title; "
                "default output dir is quadrant_outflow_demo_clean (override with --outdir)"
            ),
        )
        dargs = ap_demo.parse_args(argv)
        if (dargs.v_lo is None) ^ (dargs.v_hi is None):
            raise SystemExit("Euler2D demo: pass both --v_lo and --v_hi, or neither (v defaults to u-range).")
        if dargs.random_ic and dargs.fix_xy and dargs.random_split:
            raise SystemExit("Euler2D demo: use only one of --fix_xy or --random_split with --random_ic.")
        if dargs.random_ic and not dargs.fix_xy:
            random_split_demo = True
        else:
            random_split_demo = bool(dargs.random_split)
        outdir_raw = (dargs.outdir or "").strip()
        outdir = outdir_raw if outdir_raw else None
        if dargs.clean_export and not outdir:
            clean_dir = "quadrant_outflow_demo_clean"
            if dargs.ic_seed is not None and (dargs.random_ic or random_split_demo):
                clean_dir = f"{clean_dir}_seed{dargs.ic_seed}"
            outdir = os.path.join(os.path.dirname(__file__), clean_dir)
        v_rng = None if dargs.v_lo is None or dargs.v_hi is None else (dargs.v_lo, dargs.v_hi)
        run_quadrant_outflow_demo(
            nx=dargs.nx,
            ny=dargs.ny,
            dt_snap=dargs.dt_snap,
            t_final=dargs.t_final,
            cfl=dargs.cfl,
            outdir=outdir,
            reconstruction="component" if dargs.component else "characteristic",
            verbose=not dargs.quiet,
            x0=dargs.x0,
            y0=dargs.y0,
            random_quadrant_ic=dargs.random_ic,
            ic_seed=dargs.ic_seed,
            random_split=random_split_demo,
            split_margin=dargs.split_margin,
            rho_range=(dargs.rho_lo, dargs.rho_hi),
            p_range=(dargs.p_lo, dargs.p_hi),
            u_range=(dargs.u_lo, dargs.u_hi),
            v_range=v_rng,
            plot_cmap=dargs.plot_cmap,
            plot_pct_lo=dargs.plot_pct_lo,
            plot_pct_hi=dargs.plot_pct_hi,
            plot_global_scale=dargs.plot_global_scale,
            plot_dpi=dargs.plot_dpi,
            plot_avg_factor=dargs.plot_avg_factor,
            clean_export=dargs.clean_export,
        )
    else:
        _main_dataset_cli()
