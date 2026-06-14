"""
FD-WENOZ 1D scalar conservation law: u_t + f(u)_x = 0.

- Finite difference: point values u_i; WENO-Z reconstructs flux at interfaces.
- NumPy backend, float64, batch (B, N) supported.
- Boundary conditions: periodic, zero, and transmissive (constant extrapolation).
- CFL step uses the minimum over the batch of max |f'(u)|.
"""

import os

import numpy as np

try:
    from numba import config as numba_config
    from numba import get_num_threads as _numba_get_num_threads
    from numba import njit, prange
    from numba import set_num_threads as _numba_set_num_threads

    if not os.environ.get("NUMBA_THREADING_LAYER"):
        numba_config.THREADING_LAYER = "workqueue"

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


def set_numba_thread_count(n_threads):
    if not NUMBA_AVAILABLE:
        return None
    n_threads = max(1, int(n_threads))
    _numba_set_num_threads(n_threads)
    return _numba_get_num_threads()


def get_numba_thread_count():
    if not NUMBA_AVAILABLE:
        return None
    return _numba_get_num_threads()

# Default dtype for the solver
SOLVER_DTYPE = np.float64


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _weno5_left_point_numba(vm2, vm1, v0, vp1, vp2, eps, use_wenoz):
        p0 = (1.0 / 3.0) * vm2 - (7.0 / 6.0) * vm1 + (11.0 / 6.0) * v0
        p1 = -(1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0 + (1.0 / 3.0) * vp1
        p2 = (1.0 / 3.0) * v0 + (5.0 / 6.0) * vp1 - (1.0 / 6.0) * vp2

        beta0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v0) ** 2 + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0) ** 2
        beta1 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2 + 0.25 * (vm1 - vp1) ** 2
        beta2 = (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2 + 0.25 * (3.0 * v0 - 4.0 * vp1 + vp2) ** 2

        d0, d1, d2 = 0.1, 0.6, 0.3
        if use_wenoz:
            tau5 = abs(beta0 - beta2)
            a0 = d0 * (1.0 + (tau5 / (eps + beta0)) ** 2)
            a1 = d1 * (1.0 + (tau5 / (eps + beta1)) ** 2)
            a2 = d2 * (1.0 + (tau5 / (eps + beta2)) ** 2)
        else:
            a0 = d0 / (eps + beta0) ** 2
            a1 = d1 / (eps + beta1) ** 2
            a2 = d2 / (eps + beta2) ** 2

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum
        return w0 * p0 + w1 * p1 + w2 * p2


    @njit(cache=True, fastmath=True)
    def _weno5_right_point_numba(vm1, v0, vp1, vp2, vp3, eps, use_wenoz):
        p0 = (11.0 / 6.0) * vp1 - (7.0 / 6.0) * vp2 + (1.0 / 3.0) * vp3
        p1 = (1.0 / 3.0) * v0 + (5.0 / 6.0) * vp1 - (1.0 / 6.0) * vp2
        p2 = -(1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0 + (1.0 / 3.0) * vp1

        beta0 = (13.0 / 12.0) * (vp1 - 2.0 * vp2 + vp3) ** 2 + 0.25 * (3.0 * vp1 - 4.0 * vp2 + vp3) ** 2
        beta1 = (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2 + 0.25 * (v0 - vp2) ** 2
        beta2 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2 + 0.25 * (vm1 - 4.0 * v0 + 3.0 * vp1) ** 2

        d0, d1, d2 = 0.1, 0.6, 0.3
        if use_wenoz:
            tau5 = abs(beta0 - beta2)
            a0 = d0 * (1.0 + (tau5 / (eps + beta0)) ** 2)
            a1 = d1 * (1.0 + (tau5 / (eps + beta1)) ** 2)
            a2 = d2 * (1.0 + (tau5 / (eps + beta2)) ** 2)
        else:
            a0 = d0 / (eps + beta0) ** 2
            a1 = d1 / (eps + beta1) ** 2
            a2 = d2 / (eps + beta2) ** 2

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum
        return w0 * p0 + w1 * p1 + w2 * p2


    @njit(cache=True, fastmath=True)
    def _periodic_fill_ghosts_numba(line, ext):
        n = line.shape[0]
        for i in range(3):
            ext[i] = line[n - 3 + i]
            ext[n + 3 + i] = line[i]
        for i in range(n):
            ext[i + 3] = line[i]


    @njit(cache=True, fastmath=True)
    def _rhs_line_burgers_periodic_numba(line, h, eps, use_wenoz, use_global_lf):
        n = line.shape[0]
        u_ext = np.empty(n + 6, dtype=np.float64)
        f_ext = np.empty(n + 6, dtype=np.float64)
        fhat = np.empty(n + 1, dtype=np.float64)
        rhs = np.empty(n, dtype=np.float64)

        _periodic_fill_ghosts_numba(line, u_ext)
        for i in range(n + 6):
            f_ext[i] = 0.5 * u_ext[i] * u_ext[i]

        amax = 0.0
        if use_global_lf:
            for i in range(n):
                ui = abs(line[i])
                if ui > amax:
                    amax = ui

        for m in range(n + 1):
            if use_global_lf:
                a0 = amax
                a1 = amax
                a2 = amax
                a3 = amax
                a4 = amax
                a5 = amax
            else:
                # Match scalar FD_WENOZ ``_numerical_flux_all_interfaces``: ``_split_flux(u_ext)``
                # uses cellwise alpha on the padded line (not the single ``a_int`` per interface
                # from ``_numerical_flux_all_interfaces_internal``, which is used for characteristic
                # systems).
                a0 = abs(u_ext[m + 0])
                a1 = abs(u_ext[m + 1])
                a2 = abs(u_ext[m + 2])
                a3 = abs(u_ext[m + 3])
                a4 = abs(u_ext[m + 4])
                a5 = abs(u_ext[m + 5])

            fp0 = 0.5 * (f_ext[m + 0] + a0 * u_ext[m + 0])
            fp1 = 0.5 * (f_ext[m + 1] + a1 * u_ext[m + 1])
            fp2 = 0.5 * (f_ext[m + 2] + a2 * u_ext[m + 2])
            fp3 = 0.5 * (f_ext[m + 3] + a3 * u_ext[m + 3])
            fp4 = 0.5 * (f_ext[m + 4] + a4 * u_ext[m + 4])

            fm0 = 0.5 * (f_ext[m + 1] - a1 * u_ext[m + 1])
            fm1 = 0.5 * (f_ext[m + 2] - a2 * u_ext[m + 2])
            fm2 = 0.5 * (f_ext[m + 3] - a3 * u_ext[m + 3])
            fm3 = 0.5 * (f_ext[m + 4] - a4 * u_ext[m + 4])
            fm4 = 0.5 * (f_ext[m + 5] - a5 * u_ext[m + 5])

            fp_half = _weno5_left_point_numba(fp0, fp1, fp2, fp3, fp4, eps, use_wenoz)
            fm_half = _weno5_right_point_numba(fm0, fm1, fm2, fm3, fm4, eps, use_wenoz)
            fhat[m] = fp_half + fm_half

        for i in range(n):
            rhs[i] = -(fhat[i + 1] - fhat[i]) / h
        return rhs


    @njit(cache=True, fastmath=True, parallel=True)
    def _rhs_x_burgers_periodic_numba(u, dx, eps, use_wenoz, use_global_lf):
        ny, nx = u.shape
        out = np.empty((ny, nx), dtype=np.float64)
        for j in prange(ny):
            out[j, :] = _rhs_line_burgers_periodic_numba(u[j, :], dx, eps, use_wenoz, use_global_lf)
        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _rhs_y_burgers_periodic_numba(u, dy, eps, use_wenoz, use_global_lf):
        ny, nx = u.shape
        out = np.empty((ny, nx), dtype=np.float64)
        for i in prange(nx):
            out[:, i] = _rhs_line_burgers_periodic_numba(u[:, i], dy, eps, use_wenoz, use_global_lf)
        return out


    @njit(cache=True, fastmath=True)
    def _max_abs_2d_numba(u):
        ny, nx = u.shape
        amax = 0.0
        for j in range(ny):
            for i in range(nx):
                val = abs(u[j, i])
                if val > amax:
                    amax = val
        return amax


    @njit(cache=True, fastmath=True)
    def _step_burgers_periodic_2d_numba(u, dx, dy, dt, eps, use_wenoz, use_global_lf):
        k1 = _rhs_x_burgers_periodic_numba(u, dx, eps, use_wenoz, use_global_lf) + _rhs_y_burgers_periodic_numba(u, dy, eps, use_wenoz, use_global_lf)
        u1 = u + dt * k1

        k2 = _rhs_x_burgers_periodic_numba(u1, dx, eps, use_wenoz, use_global_lf) + _rhs_y_burgers_periodic_numba(u1, dy, eps, use_wenoz, use_global_lf)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)

        k3 = _rhs_x_burgers_periodic_numba(u2, dx, eps, use_wenoz, use_global_lf) + _rhs_y_burgers_periodic_numba(u2, dy, eps, use_wenoz, use_global_lf)
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)


    @njit(cache=True, fastmath=True)
    def _advance_burgers_periodic_2d_numba(u0, dx, dy, T, cfl, eps, use_wenoz, use_global_lf):
        u = u0.copy()
        t = 0.0
        while t < T - 1e-15:
            amax = _max_abs_2d_numba(u)
            if amax < 1e-14:
                dt_step = T - t
            else:
                dt_step = cfl / (amax / dx + amax / dy)
                if dt_step > T - t:
                    dt_step = T - t
            u = _step_burgers_periodic_2d_numba(u, dx, dy, dt_step, eps, use_wenoz, use_global_lf)
            t += dt_step
        return u


    @njit(cache=True, fastmath=True)
    def _solve_snapshots_burgers_periodic_2d_numba(u0, dx, dy, dt_snap, n_snaps, cfl, eps, use_wenoz, use_global_lf):
        traj = np.empty((n_snaps, u0.shape[0], u0.shape[1]), dtype=np.float64)
        traj[0] = u0
        u = u0.copy()
        for i in range(1, n_snaps):
            u = _advance_burgers_periodic_2d_numba(u, dx, dy, dt_snap, cfl, eps, use_wenoz, use_global_lf)
            traj[i] = u
        return traj


# =============================================================================
# Cell-average downsample (conservation law)
# =============================================================================
def downsample_cell_average(u, factor):
    """
    Conservative downsampling by cell average.

    u: (..., N) with N divisible by factor.
    Returns: (..., N // factor), each value = mean of `factor` consecutive cells.

    Works for NumPy arrays (and Torch via WENO5.downsample).
    """
    if u.shape[-1] % factor != 0:
        raise ValueError(
            f"Last dimension {u.shape[-1]} must be divisible by factor={factor}."
        )
    n = u.shape[-1] // factor
    return u.reshape(*u.shape[:-1], n, factor).mean(axis=-1)


def downsample_cell_average2d(u, factor_y, factor_x=None):
    """
    Conservative 2D block-average downsampling on the last two axes.

    u: (..., Ny, Nx)
    Returns: (..., Ny // factor_y, Nx // factor_x)
    """
    if factor_x is None:
        factor_x = factor_y
    if u.shape[-2] % factor_y != 0 or u.shape[-1] % factor_x != 0:
        raise ValueError(
            f"Last two dimensions {(u.shape[-2], u.shape[-1])} must be divisible by "
            f"(factor_y, factor_x)=({factor_y}, {factor_x})."
        )
    ny = u.shape[-2] // factor_y
    nx = u.shape[-1] // factor_x
    return u.reshape(*u.shape[:-2], ny, factor_y, nx, factor_x).mean(axis=-1).mean(axis=-2)


from numpy.lib.stride_tricks import sliding_window_view

class FD_WENOZ:
    def __init__(
        self,
        flux,
        dflux=None,
        alpha=None,
        char_decomp=None,
        char_decomp_batch=None,
        n_comp=1,
        flux_split="local_lf",
        eps=1e-40,
        bc="periodic",
        reflect_sign=1.0,
        WENOtype="WENO-Z",
        dtype=np.float64,
    ):
        self.flux = flux
        self.dflux = dflux
        self.alpha = alpha
        self.char_decomp = char_decomp
        self.char_decomp_batch = char_decomp_batch
        self.n_comp = int(n_comp)
        self.use_characteristics = (
            self.n_comp > 1
            and (self.char_decomp_batch is not None or self.char_decomp is not None)
        )

        self.flux_split = flux_split
        self.eps = eps
        self.bc = bc
        self.reflect_sign = reflect_sign
        self.ng = 3
        self.WENOtype = WENOtype
        self.dtype = dtype

        if self.n_comp < 1:
            raise ValueError("n_comp must be >= 1.")

        if bc == "edge":
            bc = "outflow"
            self.bc = bc

        if bc not in ("periodic", "outflow", "zero", "reflect"):
            raise ValueError('bc must be "periodic", "outflow", "zero", or "reflect".')

        if flux_split not in ("global_lf", "local_lf"):
            raise ValueError("flux_split must be 'global_lf' or 'local_lf'.")

        if WENOtype not in ("WENO-Z", "WENO-JS"):
            raise ValueError("WENOtype must be 'WENO-Z' or 'WENO-JS'.")

        if self.alpha is None and self.dflux is None:
            raise ValueError(
                "Need alpha or dflux for LF flux splitting. "
                "They must return a spectral-radius bound, not a Jacobian matrix."
            )

    def _to_numpy(self, u):
        return np.asarray(u, dtype=SOLVER_DTYPE)

    def _get_alpha_raw(self, u):
        """
        Alpha for Lax-Friedrichs splitting; shape broadcastable to u.
        """
        if self.alpha is not None:
            a = self.alpha(u) if callable(self.alpha) else self.alpha
        else:
            if self.dflux is None:
                raise ValueError("Need dflux or alpha for Lax-Friedrichs flux splitting.")
            a = self.dflux(u)
        return np.abs(np.asarray(a, dtype=u.dtype))

    def _get_alpha_max(self, u):
        """
        Global max wave speed over all entries.
        """
        a = self._get_alpha_raw(u)
        return float(np.max(a))

    def _flux_array(self, u):
        """
        Flux evaluated on an array of states.

        For scalar problems u may be shape (N,) or (..., N).
        For systems with n_comp > 1, u is typically (n_comp, N).
        """
        return np.asarray(self.flux(u), dtype=np.asarray(u).dtype)

    def _alpha_vector(self, u):
        """
        Wave-speed bound on the last axis.

        Returns shape (N,) for system states (n_comp, N), or a shape
        broadcastable to u for scalar problems.
        """
        a = self._get_alpha_raw(u)
        if self.n_comp > 1 and np.ndim(a) > 1:
            raise ValueError(
                "For system problems, alpha must return a 1D array over space "
                f"or a scalar. Got shape {np.shape(a)}."
            )
        return np.asarray(a, dtype=np.asarray(u).dtype)

    def _reflect_sign_array(self, u):
        """
        Return reflect_sign in a dtype/shape usable for broadcasting.

        Typical uses:
          scalar: reflect_sign = +/-1
          Euler with u shape (3, N): reflect_sign = np.array([1,-1,1])[:, None]
          Euler with u shape (B, 3, N): reflect_sign = np.array([1,-1,1])[None, :, None]
        """
        s = np.asarray(self.reflect_sign, dtype=u.dtype)
        return s

    def _extend_state(self, u):
        """
        Create 3 ghost cells on each side.
        Input u shape: (..., N)
        Output shape: (..., N + 2*ng)
        """
        u = self._to_numpy(u)
        ng = self.ng

        if u.shape[-1] < 3:
            raise ValueError("Need at least 3 spatial points for WENO5.")

        if self.bc == "periodic":
            return np.pad(u, [(0, 0)] * (u.ndim - 1) + [(ng, ng)], mode="wrap")

        if self.bc == "outflow":
            return np.pad(u, [(0, 0)] * (u.ndim - 1) + [(ng, ng)], mode="edge")

        if self.bc == "zero":
            return np.pad(
                u,
                [(0, 0)] * (u.ndim - 1) + [(ng, ng)],
                mode="constant",
                constant_values=0.0,
            )

        # True reflection via mirrored ghost cells
        s = self._reflect_sign_array(u)
        left = s * np.flip(u[..., :ng], axis=-1)
        right = s * np.flip(u[..., -ng:], axis=-1)
        return np.concatenate([left, u, right], axis=-1)

    def _weno5_left_from_ext(self, v_ext, N):
        """
        Left-biased WENO-Z reconstruction at ALL interfaces.

        v_ext has shape (..., N + 6) because ng = 3.
        Returns shape (..., N+1), where output[..., m] approximates v^- at interface m
        (m = 0,...,N corresponding to x_{-1/2}, x_{1/2}, ..., x_{N-1/2}).
        """
        # For interface m, use cells m-3, m-2, m-1, m, m+1
        vm2 = v_ext[..., 0 : N + 1]
        vm1 = v_ext[..., 1 : N + 2]
        v0 = v_ext[..., 2 : N + 3]
        vp1 = v_ext[..., 3 : N + 4]
        vp2 = v_ext[..., 4 : N + 5]

        p0 = (1.0 / 3.0) * vm2 - (7.0 / 6.0) * vm1 + (11.0 / 6.0) * v0
        p1 = -(1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0 + (1.0 / 3.0) * vp1
        p2 = (1.0 / 3.0) * v0 + (5.0 / 6.0) * vp1 - (1.0 / 6.0) * vp2

        beta0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v0) ** 2 + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0) ** 2
        beta1 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2 + 0.25 * (vm1 - vp1) ** 2
        beta2 = (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2 + 0.25 * (3.0 * v0 - 4.0 * vp1 + vp2) ** 2

        d0, d1, d2 = 0.1, 0.6, 0.3
        if self.WENOtype == 'WENO-Z':
            tau5 = np.abs(beta0 - beta2)
            p = 2.0
            a0 = d0 * (1.0 + (tau5 / (self.eps + beta0)) ** p)
            a1 = d1 * (1.0 + (tau5 / (self.eps + beta1)) ** p)
            a2 = d2 * (1.0 + (tau5 / (self.eps + beta2)) ** p)
        else:
            a0 = d0 / (self.eps + beta0)**2
            a1 = d1 / (self.eps + beta1)**2
            a2 = d2 / (self.eps + beta2)**2

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum
        return w0 * p0 + w1 * p1 + w2 * p2

    def _weno5_right_from_ext(self, v_ext, N):
        """
        Right-biased WENO-Z reconstruction at ALL interfaces.

        Returns shape (..., N+1), where output[..., m] approximates v^+ at interface m.
        """
        # For interface m, use cells m-2, m-1, m, m+1, m+2
        vm1 = v_ext[..., 1 : N + 2]
        v0 = v_ext[..., 2 : N + 3]
        vp1 = v_ext[..., 3 : N + 4]
        vp2 = v_ext[..., 4 : N + 5]
        vp3 = v_ext[..., 5 : N + 6]

        p0 = (11.0 / 6.0) * vp1 - (7.0 / 6.0) * vp2 + (1.0 / 3.0) * vp3
        p1 = (1.0 / 3.0) * v0 + (5.0 / 6.0) * vp1 - (1.0 / 6.0) * vp2
        p2 = -(1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0 + (1.0 / 3.0) * vp1

        beta0 = (13.0 / 12.0) * (vp1 - 2.0 * vp2 + vp3) ** 2 + 0.25 * (3.0 * vp1 - 4.0 * vp2 + vp3) ** 2
        beta1 = (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2 + 0.25 * (v0 - vp2) ** 2
        beta2 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2 + 0.25 * (vm1 - 4.0 * v0 + 3.0 * vp1) ** 2

        d0, d1, d2 = 0.1, 0.6, 0.3
        if self.WENOtype == 'WENO-Z':
            tau5 = np.abs(beta0 - beta2)
            p = 2.0
            a0 = d0 * (1.0 + (tau5 / (self.eps + beta0)) ** p)
            a1 = d1 * (1.0 + (tau5 / (self.eps + beta1)) ** p)
            a2 = d2 * (1.0 + (tau5 / (self.eps + beta2)) ** p)
        else:
            a0 = d0 / (self.eps + beta0)**2
            a1 = d1 / (self.eps + beta1)**2
            a2 = d2 / (self.eps + beta2)**2

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum
        return w0 * p0 + w1 * p1 + w2 * p2

    def _weno5_left_stencil_batch(self, v):
        """
        v shape (..., 5)
        """
        vm2 = v[..., 0]
        vm1 = v[..., 1]
        v0  = v[..., 2]
        vp1 = v[..., 3]
        vp2 = v[..., 4]

        p0 = (1.0 / 3.0) * vm2 - (7.0 / 6.0) * vm1 + (11.0 / 6.0) * v0
        p1 = -(1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0 + (1.0 / 3.0) * vp1
        p2 = (1.0 / 3.0) * v0 + (5.0 / 6.0) * vp1 - (1.0 / 6.0) * vp2

        beta0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v0) ** 2 + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0) ** 2
        beta1 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2 + 0.25 * (vm1 - vp1) ** 2
        beta2 = (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2 + 0.25 * (3.0 * v0 - 4.0 * vp1 + vp2) ** 2

        d0, d1, d2 = 0.1, 0.6, 0.3

        if self.WENOtype == "WENO-Z":
            tau5 = np.abs(beta0 - beta2)
            p = 2.0
            a0 = d0 * (1.0 + (tau5 / (self.eps + beta0)) ** p)
            a1 = d1 * (1.0 + (tau5 / (self.eps + beta1)) ** p)
            a2 = d2 * (1.0 + (tau5 / (self.eps + beta2)) ** p)
        else:
            a0 = d0 / (self.eps + beta0) ** 2
            a1 = d1 / (self.eps + beta1) ** 2
            a2 = d2 / (self.eps + beta2) ** 2

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum
        return w0 * p0 + w1 * p1 + w2 * p2


    def _weno5_right_stencil_batch(self, v):
        """
        v shape (..., 5)
        interpret as [v_{m-1}, v_m, v_{m+1}, v_{m+2}, v_{m+3}]
        """
        vm1 = v[..., 0]
        v0  = v[..., 1]
        vp1 = v[..., 2]
        vp2 = v[..., 3]
        vp3 = v[..., 4]

        p0 = (11.0 / 6.0) * vp1 - (7.0 / 6.0) * vp2 + (1.0 / 3.0) * vp3
        p1 = (1.0 / 3.0) * v0 + (5.0 / 6.0) * vp1 - (1.0 / 6.0) * vp2
        p2 = -(1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0 + (1.0 / 3.0) * vp1

        beta0 = (13.0 / 12.0) * (vp1 - 2.0 * vp2 + vp3) ** 2 + 0.25 * (3.0 * vp1 - 4.0 * vp2 + vp3) ** 2
        beta1 = (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2 + 0.25 * (v0 - vp2) ** 2
        beta2 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2 + 0.25 * (vm1 - 4.0 * v0 + 3.0 * vp1) ** 2

        d0, d1, d2 = 0.1, 0.6, 0.3

        if self.WENOtype == "WENO-Z":
            tau5 = np.abs(beta0 - beta2)
            p = 2.0
            a0 = d0 * (1.0 + (tau5 / (self.eps + beta0)) ** p)
            a1 = d1 * (1.0 + (tau5 / (self.eps + beta1)) ** p)
            a2 = d2 * (1.0 + (tau5 / (self.eps + beta2)) ** p)
        else:
            a0 = d0 / (self.eps + beta0) ** 2
            a1 = d1 / (self.eps + beta1) ** 2
            a2 = d2 / (self.eps + beta2) ** 2

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum
        return w0 * p0 + w1 * p1 + w2 * p2

    def _char_mats_batch(self, uL, uR):
        """
        uL, uR shape: (n_comp, Nint)
        return:
            L, R, a_int
            L.shape = (Nint, n_comp, n_comp)
            R.shape = (Nint, n_comp, n_comp)
            a_int.shape = (Nint,) or None
        """
        if self.char_decomp_batch is not None:
            out = self.char_decomp_batch(uL, uR)
        else:
            Nint = uL.shape[1]
            Ls, Rs, As = [], [], []
            has_a = True
            for i in range(Nint):
                tmp = self.char_decomp(uL[:, i], uR[:, i])
                if len(tmp) == 2:
                    L, R = tmp
                    a = None
                    has_a = False
                else:
                    L, R, a = tmp
                Ls.append(np.asarray(L, dtype=self.dtype))
                Rs.append(np.asarray(R, dtype=self.dtype))
                As.append(a)

            L = np.stack(Ls, axis=0)
            R = np.stack(Rs, axis=0)
            a_int = None if not has_a else np.asarray(As, dtype=self.dtype)
            return L, R, a_int

        if not isinstance(out, (tuple, list)):
            raise ValueError("char_decomp_batch must return (L, R) or (L, R, a_int).")

        if len(out) == 2:
            L, R = out
            a_int = None
        elif len(out) == 3:
            L, R, a_int = out
        else:
            raise ValueError("char_decomp_batch must return (L, R) or (L, R, a_int).")

        L = np.asarray(L, dtype=self.dtype)
        R = np.asarray(R, dtype=self.dtype)

        Nint = uL.shape[1]
        if L.shape != (Nint, self.n_comp, self.n_comp):
            raise ValueError(
                f"L must have shape ({Nint}, {self.n_comp}, {self.n_comp}), got {L.shape}."
            )
        if R.shape != (Nint, self.n_comp, self.n_comp):
            raise ValueError(
                f"R must have shape ({Nint}, {self.n_comp}, {self.n_comp}), got {R.shape}."
            )

        if a_int is not None:
            a_int = np.asarray(a_int, dtype=self.dtype)
            if a_int.shape != (Nint,):
                raise ValueError(f"a_int must have shape ({Nint},), got {a_int.shape}.")

        return L, R, a_int


    def _numerical_flux_all_interfaces_internal(self, u):
        """
        Internal u shape:
        scalar: (N,)
        system: (n_comp, N)

        Output:
        scalar: (N+1,)
        system: (n_comp, N+1)
        """
        u_ext = self._extend_state(u)
        f_ext = self._flux_array(u_ext)
        a_ext = self._alpha_vector(u_ext)

        N = u.shape[-1]
        Nint = N + 1

        # ---------- scalar fully vectorized ----------
        if self.n_comp == 1:
            u6 = sliding_window_view(u_ext, 6, axis=-1)   # (N+1, 6)
            f6 = sliding_window_view(f_ext, 6, axis=-1)   # (N+1, 6)

            if self.flux_split == "global_lf":
                a_int = np.full(Nint, np.max(a_ext), dtype=self.dtype)
            else:
                # One LF coefficient per interface (max over the 6-point window). Note: scalar
                # ``n_comp==1`` problems without characteristics use ``_split_flux(u_ext)``
                # instead (cellwise alpha), which is not identical to this formula.
                a6 = sliding_window_view(a_ext, 6, axis=-1)   # (N+1, 6)
                a_int = np.max(a6, axis=-1)                   # (N+1,)

            fp5 = 0.5 * (f6[:, :5] + a_int[:, None] * u6[:, :5])   # (N+1, 5)
            fm5 = 0.5 * (f6[:, 1:] - a_int[:, None] * u6[:, 1:])   # (N+1, 5)

            fp_half = self._weno5_left_stencil_batch(fp5)          # (N+1,)
            fm_half = self._weno5_right_stencil_batch(fm5)         # (N+1,)
            return fp_half + fm_half

        # ---------- system fully vectorized ----------
        # u_ext, f_ext: (C, N+6)
        C = self.n_comp

        u6 = sliding_window_view(u_ext, 6, axis=-1)   # (C, N+1, 6)
        f6 = sliding_window_view(f_ext, 6, axis=-1)   # (C, N+1, 6)

        if self.flux_split == "global_lf":
            a_int = np.full(Nint, np.max(a_ext), dtype=self.dtype)
        else:
            a6 = sliding_window_view(a_ext, 6, axis=-1)   # (N+1, 6)
            a_int = np.max(a6, axis=-1)                   # (N+1,)

        uL = u_ext[:, 2 : 2 + Nint]   # (C, N+1)
        uR = u_ext[:, 3 : 3 + Nint]   # (C, N+1)

        L, R, a_char = self._char_mats_batch(uL, uR)
        if a_char is not None:
            a_int = np.maximum(a_int, a_char)

        # split flux on each interface/stencil
        # u6,f6: (C, M, 6), a_int: (M,)
        fp5 = 0.5 * (f6[:, :, :5] + a_int[None, :, None] * u6[:, :, :5])   # (C, M, 5)
        fm5 = 0.5 * (f6[:, :, 1:] - a_int[None, :, None] * u6[:, :, 1:])   # (C, M, 5)

        # L: (M,C,C); fp5: (C,M,5) -> (M,C,5) for batched matmul (faster than einsum here).
        Fp = np.swapaxes(fp5, 0, 1)
        Fm = np.swapaxes(fm5, 0, 1)
        gp5 = np.matmul(L, Fp)
        gm5 = np.matmul(L, Fm)

        # WENO on each characteristic field, all interfaces at once (last axis = stencil)
        ghat_p = self._weno5_left_stencil_batch(gp5)
        ghat_m = self._weno5_right_stencil_batch(gm5)
        ghat = ghat_p + ghat_m

        # R (M,C,C) @ ghat (M,C) -> (M,C) then (C,M)
        fhat = np.matmul(R, ghat[..., np.newaxis]).squeeze(-1).T
        return fhat

    def _split_flux(self, u):
        """
        LF splitting on state values u (not yet extended):
            f^± = 0.5 * (f ± a u)
        """
        u = self._to_numpy(u)
        f = np.asarray(self.flux(u), dtype=u.dtype)

        if self.flux_split == "global_lf":
            amax = self._get_alpha_max(u)
            a = np.full_like(u, amax)
        else:
            a = self._get_alpha_raw(u)

        fp = 0.5 * (f + a * u)
        fm = 0.5 * (f - a * u)
        return fp, fm

    def _numerical_flux_all_interfaces(self, u):
        """
        Build numerical flux at ALL N+1 interfaces.
        Input u shape (..., N)
        Output fhat shape (..., N+1)
        """
        u = self._to_numpy(u)

        if self.use_characteristics:
            if u.ndim != 2 or u.shape[0] != self.n_comp:
                raise NotImplementedError(
                    "Characteristic WENO currently expects system states with shape "
                    f"({self.n_comp}, N). Got {u.shape}."
                )
            return self._numerical_flux_all_interfaces_internal(u)

        u_ext = self._extend_state(u)
        N = u.shape[-1]

        fp_ext, fm_ext = self._split_flux(u_ext)

        fp_half = self._weno5_left_from_ext(fp_ext, N)   # (..., N+1)
        fm_half = self._weno5_right_from_ext(fm_ext, N)  # (..., N+1)
        return fp_half + fm_half

    def rhs(self, u, dx):
        """
        Semi-discrete conservative RHS:
            du/dt = -(fhat_{i+1/2} - fhat_{i-1/2}) / dx

        Returns same shape as u.
        """
        u = self._to_numpy(u)
        fhat = self._numerical_flux_all_interfaces(u)   # (..., N+1)
        return -(fhat[..., 1:] - fhat[..., :-1]) / dx

    def step(self, u, dx, dt):
        """
        One SSPRK(3,3) step.
        """
        u = self._to_numpy(u)

        k1 = self.rhs(u, dx)
        u1 = u + dt * k1

        k2 = self.rhs(u1, dx)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)

        k3 = self.rhs(u2, dx)
        unew = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)
        return unew

    def solve(
        self,
        u0,
        dx,
        T=None,
        dt=None,
        n_steps=None,
        cfl=0.4,
        return_all=False,
    ):
        """
        Integrate in time.

        Parameters
        ----------
        u0 : array
            shape (N,) or (..., N)
        dx : float
        T : float or None
            final time
        dt : float or None
            fixed time step if provided
        n_steps : int or None
            number of fixed steps if T is None
        cfl : float
            used only when dt is None
        return_all : bool
            if True, return trajectory with time as axis 0

        Returns
        -------
        out : array
            final state or trajectory
        """
        u = self._to_numpy(u0).astype(SOLVER_DTYPE)
        squeeze_out = False
        if u.ndim == 1:
            u = u[None, :]
            squeeze_out = True

        if T is None and n_steps is None:
            raise ValueError("Provide T or n_steps.")

        traj = [u.copy()] if return_all else None

        if T is not None:
            t = 0.0
            while t < T - 1e-15:
                if dt is None:
                    amax = self._get_alpha_max(u)
                    if amax < 1e-14:
                        dt_step = T - t
                    else:
                        dt_step = min(cfl * dx / amax, T - t)
                else:
                    dt_step = min(dt, T - t)
                
                u = self.step(u, dx, dt_step)
                t += dt_step

                if return_all:
                    traj.append(u.copy())

            out = np.stack(traj, axis=0) if return_all else u

        else:
            if n_steps is None:
                raise ValueError("When T is None, n_steps must be provided.")

            if dt is None:
                amax = self._get_alpha_max(u)
                if amax < 1e-14:
                    raise ValueError("Wave speed is ~0; set dt manually.")
                dt = cfl * dx / amax

            for _ in range(n_steps):
                u = self.step(u, dx, dt)
                if return_all:
                    traj.append(u.copy())

            out = np.stack(traj, axis=0) if return_all else u

        if squeeze_out:
            if return_all:
                out = out[:, 0, :]
            else:
                out = out[0]
        return out

    @staticmethod
    def downsample(u, factor):
        return downsample_cell_average(u, factor)


class FD_WENOZ2D:
    """
    Minimal 2D finite-difference WENO solver for scalar conservation laws

        u_t + f(u)_x + g(u)_y = 0

    on a Cartesian grid. Spatial discretization is **dimension-by-dimension** using
    the existing 1D ``FD_WENOZ`` along each axis (same as classic Strang splitting in
    space for the semi-discrete RHS).

    **Boundary conditions** (same strings as ``FD_WENOZ``): ``periodic``, ``outflow``
    (``edge`` alias: constant extrapolation / transmissive ghosts), ``zero``, ``reflect``.
    For non-periodic BCs, each 1D line sweep applies that 1D BC on the **last axis** of
    the slice: x-sweeps impose BC on the west/east edges; y-sweeps (after axis swap)
    on the south/north edges. This is the standard split extension of 1D open BCs, not
    a full multi-D Riemann invariant treatment at corners.

    ``numba_backend="scalar_periodic_burgers"`` enables a fast path only when
    ``bc="periodic"``, Burgers flux ``0.5 u^2``, and the state satisfies size/dtype
    checks; it does **not** verify ``flux_x``/``flux_y`` match Burgers — use only for
    that equation.
    """
    def __init__(
        self,
        flux_x,
        flux_y=None,
        dflux_x=None,
        dflux_y=None,
        alpha_x=None,
        alpha_y=None,
        flux_split="local_lf",
        eps=1e-40,
        bc="periodic",
        WENOtype="WENO-Z",
        dtype=np.float64,
        numba_backend=None,
    ):
        if bc == "edge":
            bc = "outflow"
        if bc not in ("periodic", "outflow", "zero", "reflect"):
            raise ValueError(
                'FD_WENOZ2D bc must be "periodic", "outflow", "zero", or "reflect" '
                '(or "edge" as alias for outflow).'
            )

        self.bc = bc
        self.dtype = dtype
        self.numba_backend = numba_backend
        self.numba_min_cells = 20000
        self.solver_x = FD_WENOZ(
            flux=flux_x,
            dflux=dflux_x,
            alpha=alpha_x,
            flux_split=flux_split,
            eps=eps,
            bc=bc,
            WENOtype=WENOtype,
            dtype=dtype,
        )
        self.solver_y = FD_WENOZ(
            flux=flux_y if flux_y is not None else flux_x,
            dflux=dflux_y if dflux_y is not None else dflux_x,
            alpha=alpha_y if alpha_y is not None else alpha_x,
            flux_split=flux_split,
            eps=eps,
            bc=bc,
            WENOtype=WENOtype,
            dtype=dtype,
        )
        if self.numba_backend is not None and not NUMBA_AVAILABLE:
            raise RuntimeError(
                f"Requested numba_backend={self.numba_backend!r}, but numba is not installed."
            )
        self._use_numba_burgers2d = self.numba_backend == "scalar_periodic_burgers"
        self._numba_use_wenoz = WENOtype == "WENO-Z"
        self._numba_use_global_lf = flux_split == "global_lf"

    def _to_numpy(self, u):
        return np.asarray(u, dtype=self.dtype)

    def _numba_compatible_scalar_state(self, u):
        return (
            self.bc == "periodic"
            and self._use_numba_burgers2d
            and isinstance(u, np.ndarray)
            and u.ndim == 2
            and u.dtype == np.float64
            and u.size >= self.numba_min_cells
        )

    def _swap_spatial_axes(self, u):
        return np.swapaxes(u, -1, -2)

    def _rhs_x(self, u, dx):
        return self.solver_x.rhs(u, dx)

    def _rhs_y(self, u, dy):
        u_t = self._swap_spatial_axes(u)
        rhs_t = self.solver_y.rhs(u_t, dy)
        return self._swap_spatial_axes(rhs_t)

    def _amax_x(self, u):
        return self.solver_x._get_alpha_max(u)

    def _amax_y(self, u):
        return self.solver_y._get_alpha_max(u)

    def _amax_pair(self, u):
        amax_x = self._amax_x(u)
        if (
            self.solver_x.alpha is self.solver_y.alpha
            and self.solver_x.dflux is self.solver_y.dflux
        ):
            return amax_x, amax_x
        return amax_x, self._amax_y(u)

    def rhs(self, u, dx, dy=None):
        if dy is None:
            dy = dx
        u = self._to_numpy(u)
        if self._numba_compatible_scalar_state(u):
            return _rhs_x_burgers_periodic_numba(
                u, dx, self.solver_x.eps, self._numba_use_wenoz, self._numba_use_global_lf
            ) + _rhs_y_burgers_periodic_numba(
                u, dy, self.solver_x.eps, self._numba_use_wenoz, self._numba_use_global_lf
            )
        return self._rhs_x(u, dx) + self._rhs_y(u, dy)

    def step(self, u, dx, dy, dt):
        k1 = self.rhs(u, dx, dy)
        u1 = u + dt * k1

        k2 = self.rhs(u1, dx, dy)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)

        k3 = self.rhs(u2, dx, dy)
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)

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

    def advance(self, u, dx, dy=None, T=None, dt=None, n_steps=None, cfl=0.4, return_all=False):
        if dy is None:
            dy = dx
        u = self._to_numpy(u)

        if (
            self._numba_compatible_scalar_state(u)
            and return_all is False
            and dt is None
        ):
            if T is not None:
                return _advance_burgers_periodic_2d_numba(
                    u, dx, dy, T, cfl, self.solver_x.eps,
                    self._numba_use_wenoz, self._numba_use_global_lf,
                )
            if n_steps is not None:
                out = u.copy()
                dt_step = self._compute_dt(out, dx, dy, dt=dt, cfl=cfl)
                if dt_step <= 0.0:
                    raise ValueError("Wave speed is ~0; set dt manually.")
                for _ in range(n_steps):
                    out = _step_burgers_periodic_2d_numba(
                        out, dx, dy, dt_step, self.solver_x.eps,
                        self._numba_use_wenoz, self._numba_use_global_lf,
                    )
                return out

        if T is None and n_steps is None:
            raise ValueError("Provide T or n_steps.")

        traj = [u.copy()] if return_all else None

        if T is not None:
            t = 0.0
            while t < T - 1e-15:
                dt_step = self._compute_dt(u, dx, dy, dt=dt, cfl=cfl, t_remaining=T - t)
                if dt_step <= 0.0:
                    break
                u = self.step(u, dx, dy, dt_step)
                t += dt_step
                if return_all:
                    traj.append(u.copy())
            return np.stack(traj, axis=0) if return_all else u

        dt_step = self._compute_dt(u, dx, dy, dt=dt, cfl=cfl)
        if dt_step <= 0.0:
            raise ValueError("Wave speed is ~0; set dt manually.")
        for _ in range(n_steps):
            use_dt = dt if dt is not None else dt_step
            u = self.step(u, dx, dy, use_dt)
            if return_all:
                traj.append(u.copy())
        return np.stack(traj, axis=0) if return_all else u

    def solve(self, u0, dx, dy=None, T=None, dt=None, n_steps=None, cfl=0.4, return_all=False):
        u = self._to_numpy(u0).copy()
        return self.advance(
            u,
            dx=dx,
            dy=dy,
            T=T,
            dt=dt,
            n_steps=n_steps,
            cfl=cfl,
            return_all=return_all,
        )

    def solve_snapshots(self, u0, dx, dy=None, dt_snap=None, n_snaps=None, T=None, cfl=0.4):
        if dt_snap is None:
            raise ValueError("Provide dt_snap.")
        if n_snaps is None:
            if T is None:
                raise ValueError("Provide n_snaps or T.")
            n_snaps = int(round(T / dt_snap)) + 1
        elif T is not None:
            expected = int(round(T / dt_snap)) + 1
            if expected != int(n_snaps):
                raise ValueError("Inconsistent T, dt_snap, and n_snaps.")

        if dy is None:
            dy = dx

        u = self._to_numpy(u0).copy()
        if self._numba_compatible_scalar_state(u):
            return _solve_snapshots_burgers_periodic_2d_numba(
                u, dx, dy, dt_snap, int(n_snaps), cfl, self.solver_x.eps,
                self._numba_use_wenoz, self._numba_use_global_lf,
            )
        traj = [u.copy()]
        for _ in range(int(n_snaps) - 1):
            u = self.advance(u, dx=dx, dy=dy, T=dt_snap, cfl=cfl, return_all=False)
            traj.append(u.copy())
        return np.stack(traj, axis=0)

    @staticmethod
    def downsample(u, factor_y, factor_x=None):
        return downsample_cell_average2d(u, factor_y, factor_x)


# =============================================================================
# Torch-compatible WENO5 (wraps FD_WENOZ for eval / legacy)
# =============================================================================
import torch

class WENO5:
    def __init__(
        self,
        flux,
        dflux=None,
        alpha=None,
        flux_split="local_lf",
        eps=1e-6,
        bc="periodic",
    ):
        self.flux = flux
        self.dflux = dflux
        self.alpha = alpha
        self.flux_split = flux_split
        self.eps = eps
        self.bc = bc

        if self.bc != "periodic":
            raise NotImplementedError("Currently only periodic BC is implemented.")

        if self.flux_split not in ["global_lf", "local_lf"]:
            raise ValueError("flux_split must be 'global_lf' or 'local_lf'.")

    # =========================================================
    # Utilities for wave speed alpha
    # =========================================================
    def _get_alpha_raw(self, u):
        """
        Return alpha before global reduction.
        Could be scalar or tensor.
        """
        if self.alpha is not None:
            a = self.alpha(u) if callable(self.alpha) else self.alpha
        else:
            if self.dflux is None:
                raise ValueError(
                    "Need either dflux or alpha for Lax-Friedrichs flux splitting."
                )
            a = self.dflux(u)

        if not torch.is_tensor(a):
            a = torch.tensor(a, dtype=u.dtype, device=u.device)
        else:
            a = a.to(dtype=u.dtype, device=u.device)

        return torch.abs(a)

    def _get_alpha_global(self, u):
        a = self._get_alpha_raw(u)
        return torch.max(a)

    # =========================================================
    # WENO5 reconstructions
    # =========================================================
    def _weno5_left(self, v):
        """
        Left-biased WENO-Z reconstruction of v at interfaces i+1/2.
        Input:
            v: (..., N)
        Output:
            vhat: (..., N), where vhat[..., i] approximates v_{i+1/2}^-
        """
        vm2 = torch.roll(v, shifts=2, dims=-1)
        vm1 = torch.roll(v, shifts=1, dims=-1)
        v0  = v
        vp1 = torch.roll(v, shifts=-1, dims=-1)
        vp2 = torch.roll(v, shifts=-2, dims=-1)

        # candidate polynomials
        p0 = (1.0/3.0) * vm2 - (7.0/6.0) * vm1 + (11.0/6.0) * v0
        p1 = -(1.0/6.0) * vm1 + (5.0/6.0) * v0 + (1.0/3.0) * vp1
        p2 = (1.0/3.0) * v0 + (5.0/6.0) * vp1 - (1.0/6.0) * vp2

        # smoothness indicators
        beta0 = (13.0/12.0) * (vm2 - 2.0*vm1 + v0)**2 + 0.25 * (vm2 - 4.0*vm1 + 3.0*v0)**2
        beta1 = (13.0/12.0) * (vm1 - 2.0*v0 + vp1)**2 + 0.25 * (vm1 - vp1)**2
        beta2 = (13.0/12.0) * (v0 - 2.0*vp1 + vp2)**2 + 0.25 * (3.0*v0 - 4.0*vp1 + vp2)**2

        # WENO-Z weights
        tau5 = torch.abs(beta0 - beta2)
        p = 2.0   # WENO-Z power

        d0, d1, d2 = 0.1, 0.6, 0.3
        a0 = d0 * (1.0 + (tau5 / (self.eps + beta0))**p)
        a1 = d1 * (1.0 + (tau5 / (self.eps + beta1))**p)
        a2 = d2 * (1.0 + (tau5 / (self.eps + beta2))**p)

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum

        return w0 * p0 + w1 * p1 + w2 * p2


    def _weno5_right(self, v):
        """
        Right-biased WENO-Z reconstruction of v at interfaces i+1/2.
        Input:
            v: (..., N)
        Output:
            vhat: (..., N), where vhat[..., i] approximates v_{i+1/2}^+
        """
        vm1 = torch.roll(v, shifts=1, dims=-1)
        v0  = v
        vp1 = torch.roll(v, shifts=-1, dims=-1)
        vp2 = torch.roll(v, shifts=-2, dims=-1)
        vp3 = torch.roll(v, shifts=-3, dims=-1)

        # candidate polynomials
        p0 = (11.0/6.0) * vp1 - (7.0/6.0) * vp2 + (1.0/3.0) * vp3
        p1 = (1.0/3.0) * v0 + (5.0/6.0) * vp1 - (1.0/6.0) * vp2
        p2 = -(1.0/6.0) * vm1 + (5.0/6.0) * v0 + (1.0/3.0) * vp1

        # smoothness indicators
        beta0 = (13.0/12.0) * (vp1 - 2.0*vp2 + vp3)**2 + 0.25 * (3.0*vp1 - 4.0*vp2 + vp3)**2
        beta1 = (13.0/12.0) * (v0 - 2.0*vp1 + vp2)**2 + 0.25 * (v0 - vp2)**2
        beta2 = (13.0/12.0) * (vm1 - 2.0*v0 + vp1)**2 + 0.25 * (vm1 - 4.0*v0 + 3.0*vp1)**2

        # WENO-Z weights
        tau5 = torch.abs(beta0 - beta2)
        p = 2.0

        # Linear weights must match NumPy FD_WENOZ / numba (_weno5_right_point_numba).
        d0, d1, d2 = 0.1, 0.6, 0.3
        a0 = d0 * (1.0 + (tau5 / (self.eps + beta0))**p)
        a1 = d1 * (1.0 + (tau5 / (self.eps + beta1))**p)
        a2 = d2 * (1.0 + (tau5 / (self.eps + beta2))**p)

        asum = a0 + a1 + a2
        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum

        return w0 * p0 + w1 * p1 + w2 * p2

    # =========================================================
    # Flux splitting
    # =========================================================
    def _split_flux(self, u):
        """
        LF flux splitting:
            f^+ = 0.5 (f + alpha u)
            f^- = 0.5 (f - alpha u)
        """
        f = self.flux(u)

        if self.flux_split == "global_lf":
            a = self._get_alpha_global(u)
        else:  # local_lf
            a = self._get_alpha_raw(u)

        fp = 0.5 * (f + a * u)
        fm = 0.5 * (f - a * u)
        return fp, fm

    # =========================================================
    # Semi-discrete RHS
    # =========================================================
    def rhs(self, u, dx):
        """
        Compute semi-discrete RHS:
            du/dt = - (  f_{i+1/2} - f_{i-1/2} ) / dx
        """
        fp, fm = self._split_flux(u)

        # reconstruct interface numerical flux
        fp_half = self._weno5_left(fp)    # f^+ at i+1/2 from left
        fm_half = self._weno5_right(fm)   # f^- at i+1/2 from right
        fhat = fp_half + fm_half

        return -(fhat - torch.roll(fhat, shifts=1, dims=-1)) / dx

    # =========================================================
    # SSPRK3 one-step
    # =========================================================
    def step(self, u, dx, dt):
        """
        One SSPRK3 step.
        """
        k1 = self.rhs(u, dx)
        u1 = u + dt * k1

        k2 = self.rhs(u1, dx)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)

        k3 = self.rhs(u2, dx)
        unew = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * k3)

        return unew

    # =========================================================
    # Time marching
    # =========================================================
    def solve(self, u0, dx, T=None, dt=None, n_steps=None, cfl=0.4, return_all=False):
        """
        Solve up to time T, or for n_steps steps.

        Provide one of:
            - T
            - n_steps
            - dt + n_steps

        If T is provided and dt is None, dt is chosen by CFL:
            dt = cfl * dx / max|f'(u)|

        Parameters
        ----------
        u0 : tensor, shape (..., N)
        dx : float
        T : float, optional
        dt : float, optional
        n_steps : int, optional
        cfl : float
        return_all : bool
            If True, return the whole trajectory.

        Returns
        -------
        u : tensor
            Final solution if return_all=False
        traj : tensor
            Shape (n_steps+1, ..., N) if return_all=True
        """
        u = u0.clone()

        if T is None and n_steps is None:
            raise ValueError("Need either T or n_steps.")

        if T is not None:
            t = 0.0
            traj = [u.clone()] if return_all else None

            while t < T:
                if dt is None:
                    amax = self._get_alpha_global(u).item()
                    if amax < 1e-14:
                        dt_step = T - t
                    else:
                        dt_step = cfl * dx / amax
                        dt_step = min(dt_step, T - t)
                else:
                    dt_step = min(dt, T - t)

                u = self.step(u, dx, dt_step)
                t += dt_step

                if return_all:
                    traj.append(u.clone())

            if return_all:
                return torch.stack(traj, dim=0)
            return u

        else:
            if dt is None:
                amax = self._get_alpha_global(u).item()
                if amax < 1e-14:
                    raise ValueError("Wave speed is zero; please provide dt manually.")
                dt = cfl * dx / amax

            traj = [u.clone()] if return_all else None
            for _ in range(n_steps):
                u = self.step(u, dx, dt)
                if return_all:
                    traj.append(u.clone())

            if return_all:
                return torch.stack(traj, dim=0)
            return u
        

# class WENO5:
#     """
#     Torch interface to FD-WENOZ: same step/solve semantics, torch tensors in/out.
#     For dataset generation with CuPy, use FD_WENOZ directly.
#     """

#     def __init__(
#         self,
#         flux,
#         dflux=None,
#         alpha=None,
#         flux_split="local_lf",
#         eps=1e-40,
#         bc="periodic",
#     ):
#         self._flux = flux
#         self._dflux = dflux
#         self._alpha = alpha
#         self._flux_split = flux_split
#         self._eps = eps
#         self._bc = bc
#         # Build CuPy-backed solver with wrappers that convert flux/dflux to cupy
#         def flux_cp(u_cp):
#             u_t = torch.as_tensor(cp.asnumpy(u_cp), dtype=torch.float64)
#             f_t = flux(u_t)
#             return cp.asarray(f_t.cpu().numpy())
#         def dflux_cp(u_cp):
#             u_t = torch.as_tensor(cp.asnumpy(u_cp), dtype=torch.float64)
#             f_t = dflux(u_t)
#             return cp.asarray(f_t.cpu().numpy())
#         def alpha_cp(u_cp):
#             u_t = torch.as_tensor(cp.asnumpy(u_cp), dtype=torch.float64)
#             a_t = alpha(u_t)
#             return cp.asarray(a_t.cpu().numpy())
#         self._solver = FD_WENOZ(
#             flux=flux_cp,
#             dflux=dflux_cp if dflux is not None else None,
#             alpha=alpha_cp if callable(alpha) else alpha,
#             flux_split=flux_split,
#             eps=eps,
#             bc=bc,
#         )

#     def rhs(self, u, dx):
#         u_cp = cp.asarray(u.detach().numpy())
#         r = self._solver.rhs(u_cp, dx)
#         return torch.as_tensor(cp.asnumpy(r), dtype=u.dtype, device=u.device)

#     def step(self, u, dx, dt):
#         u_cp = cp.asarray(u.detach().numpy())
#         out = self._solver.step(u_cp, dx, dt)
#         return torch.as_tensor(cp.asnumpy(out), dtype=u.dtype, device=u.device)

#     def solve(self, u0, dx, T=None, dt=None, n_steps=None, cfl=0.4, return_all=False):
#         u_np = u0.detach().cpu().numpy()
#         out = self._solver.solve(u_np, dx, T=T, dt=dt, n_steps=n_steps, cfl=cfl, return_all=return_all)
#         return torch.as_tensor(out, dtype=u0.dtype, device=u0.device)

#     @staticmethod
#     def downsample(u, factor):
#         """u: torch (..., N) -> torch (..., N//factor) cell average."""
#         return torch.as_tensor(
#             downsample_cell_average(u.cpu().numpy(), factor),
#             dtype=u.dtype,
#             device=u.device,
#         )


# =============================================================================
# Example / test
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L = 1.0
    N = 2048
    dx = L / N
    c = 1.0

    def flux_lin(u):
        return c * u

    def dflux_lin(u):
        return np.full_like(u, c)

    solver = FD_WENOZ(
        flux=flux_lin,
        dflux=dflux_lin,
        flux_split="local_lf",
    )

    # Point-value IC on grid
    x = np.linspace(0.0, L, N, endpoint=False)
    u0 = np.sin(2 * np.pi * x) + 0.5 * np.where(x > 0.6, 0.2, -0.2)
    u0 = u0.astype(SOLVER_DTYPE)

    T = 0.2
    uT = solver.solve(u0, dx=dx, T=T, cfl=0.4)
    u_true = np.sin(2 * np.pi * (x - c * T)) + 0.5 * np.where((x - c * T) % 1.0 > 0.6, 0.2, -0.2)

    # Batch test
    u0_batch = np.stack([u0, u0 * 0.5], axis=0)
    uT_batch = solver.solve(u0_batch, dx=dx, T=T, cfl=0.4)
    assert uT_batch.shape == (2, N)

    # Downsample
    u_coarse = downsample_cell_average(uT, 4)
    assert u_coarse.shape == (N // 4,)

    plt.figure(figsize=(7, 4))
    plt.plot(x, u0, label="u0")
    plt.plot(x, uT, label="u(T)")
    plt.plot(x, u_true, label="u_true", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
