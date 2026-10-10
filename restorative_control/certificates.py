"""Exposure certificates and the noisy-observation risk bound."""
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import xlogy

from . import model as M


# ---------------------------------------------------------------------------
# Lower certificate J_int (Lemma 2)

def _slope_gap(v, barrier):
    return (v - barrier) * M.phiprime(v) - M.phi_stable(v)


def q(barrier):
    """inf over barrier < v <= v_max of phi(v) / (v - barrier).

    The derivative of the ratio has the sign of _slope_gap, which increases
    below v0 and decreases above it. The first root is therefore the only
    interior candidate, and the other is v_max. At zero barrier the
    infimum is the limit phi'(0).
    """
    a = max(0.0, float(barrier))
    high = float(M.phi_stable(M.V_MAX)) / (M.V_MAX - a)
    if a == 0:
        return min(float(M.phiprime(0)), high)
    if a >= M.V0 or _slope_gap(M.V0, a) <= 0:
        return high
    root = brentq(_slope_gap, a, M.V0, args=(a,), xtol=5e-15)
    return min(float(M.phi_stable(root)) / (root - a), high)


def J_int():
    """Integral of q(-f(y)) over [x_d, s].

    The quadrature is split where the minimizing drive switches between
    the interior root and v_max, and at the barrier peak.
    """
    def branch_difference(a):
        if _slope_gap(M.V0, a) <= 0:
            return 1.0
        root = brentq(_slope_gap, a, M.V0, args=(a,), xtol=5e-15)
        return (M.phi_stable(root) / (root - a)
                - M.phi_stable(M.V_MAX) / (M.V_MAX - a))

    a_switch = brentq(branch_difference, 1e-8, M.V0 - 1e-8, xtol=5e-15)
    left = brentq(lambda y: -M.f(y) - a_switch, M.X_D, M.X_STAR)
    right = brentq(lambda y: -M.f(y) - a_switch, M.X_STAR, M.S)
    value, err = quad(lambda y: q(-M.f(y)), M.X_D, M.S,
                      points=[left, M.X_STAR, right],
                      epsabs=2e-12, epsrel=2e-12, limit=150)
    return value, err, (left, right)


def J_int_riemann_bounds(n=2**15):
    """Lower and upper Riemann sums, using that q(-f(y)) is monotone on each side of x*."""
    y = np.unique(np.r_[np.linspace(M.X_D, M.S, n + 1), M.X_STAR])
    qy = np.array([q(-M.f(z)) for z in y])
    dy = np.diff(y)
    return (float(np.sum(dy * np.minimum(qy[:-1], qy[1:]))),
            float(np.sum(dy * np.maximum(qy[:-1], qy[1:]))))


def J_int_grid(n=2001, nv=2001):
    """Independent check: brute-force drive grid with local refinement, Simpson in y."""
    def q_grid(y):
        lo = max(-M.f(y), 0.0) + 1e-12
        vg = np.linspace(lo, M.V_MAX, nv)
        r = M.phi(vg) / (M.f(y) + vg)
        k = int(np.argmin(r))
        a, b = vg[max(k - 1, 0)], vg[min(k + 1, nv - 1)]
        res = minimize_scalar(lambda v: float(M.phi(v)) / (M.f(y) + v),
                              bounds=(a, b), method="bounded",
                              options={"xatol": 1e-12})
        return float(min(res.fun, r[k]))

    ys = np.linspace(M.X_D, M.S, n)
    qs = np.array([q_grid(y) for y in ys])
    step = (M.S - M.X_D) / (n - 1)
    return float(step / 3 * (qs[0] + qs[-1] + 4 * qs[1::2].sum() + 2 * qs[2:-1:2].sum()))


# ---------------------------------------------------------------------------
# Sufficient certificate J_bar (Proposition 1)

def _travel_time(v, a, b):
    """Time for the comparison trajectory dz/dt = f(z) + v - rho_bar to go from a to b."""
    return quad(lambda y: 1 / (M.f(y) + v - M.RHO_BAR), a, b,
                epsabs=2e-12, epsrel=2e-12)[0]


def optimize_three_stage():
    """Best low-high-low schedule with the high level at v_max.

    For fixed levels, each switch equalizes exposure per unit progress.
    The two low levels are then chosen by a grid search and bounded
    refinement. The rounded schedule in model.py comes from this optimum.
    """
    H = M.V_MAX
    ph = float(M.phi_stable(H))

    def switch(v, side):
        pv = float(M.phi_stable(v))
        level = (ph * v - pv * H) / (ph - pv) - M.RHO_BAR
        if not 0 < level < M.U_A:
            return None
        interval = (M.X_D, M.X_STAR) if side == 0 else (M.X_STAR, M.S)
        return brentq(lambda y: -M.f(y) - level, *interval, xtol=1e-14)

    def cost(v, side):
        y = switch(v, side)
        if y is None:
            return 1e6
        if side == 0:
            return (M.phi_stable(v) * _travel_time(v, M.X_D, y)
                    + ph * _travel_time(H, y, M.X_STAR))
        return (ph * _travel_time(H, M.X_STAR, y)
                + M.phi_stable(v) * _travel_time(v, y, M.S + M.A_MARGIN))

    levels = []
    for side in (0, 1):
        grid = np.linspace(0.1, 0.39, 150)
        j = int(np.argmin([cost(v, side) for v in grid]))
        assert 0 < j < len(grid) - 1
        fit = minimize_scalar(lambda v: cost(v, side), bounds=(grid[j - 1], grid[j + 1]),
                              method="bounded", options={"xatol": 1e-12})
        levels.append(float(fit.x))
    a, b = switch(levels[0], 0), switch(levels[1], 1)
    drives = [levels[0], H, levels[1]]
    durations = [_travel_time(drives[0], M.X_D, a), _travel_time(H, a, b),
                 _travel_time(drives[2], b, M.S + M.A_MARGIN)]
    exposure = float(sum(M.phi_stable(v) * t for v, t in zip(drives, durations)))
    return dict(drives=drives, durations=durations, switch_states=[a, b],
                withdrawal=sum(durations), exposure=exposure)


def _travel_time_bounds(v, a, b, n=2**20):
    """Riemann bounds on travel time using the exact extrema of f on each cell."""
    y = np.linspace(a, b, n + 1)
    lo = np.minimum(M.f(y[:-1]), M.f(y[1:]))
    hi = np.maximum(M.f(y[:-1]), M.f(y[1:]))
    for crit in ((4 - np.sqrt(7)) / 3, (4 + np.sqrt(7)) / 3):
        inside = (y[:-1] <= crit) & (crit <= y[1:])
        lo[inside] = np.minimum(lo[inside], M.f(crit))
        hi[inside] = np.maximum(hi[inside], M.f(crit))
    assert np.min(lo + v - M.RHO_BAR) > 0
    dy = np.diff(y)
    return float(np.sum(dy / (hi + v - M.RHO_BAR))), float(np.sum(dy / (lo + v - M.RHO_BAR)))


def check_schedule(j_int):
    """Verify the rounded comparison schedule and simulate both candidates.

    Returns the exposure, the travel-time bounds for each stage, and the
    final states from two independent ODE solvers.
    """
    drives, durations = M.CERT_DRIVES, M.CERT_DURATIONS
    capacity = M.CERT_THETA1 + M.CERT_THETA2
    exposure = float(sum(M.phi_stable(v) * t for v, t in zip(drives, durations)))
    assert j_int < exposure < capacity

    # Each stage reaches its target state before the scheduled switch.
    targets = [M.X_D, 0.108143, 0.847986, M.S + M.A_MARGIN]
    travel = [_travel_time_bounds(v, a, b)
              for v, a, b in zip(drives, targets[:-1], targets[1:])]
    assert all(upper + 1e-10 < t for (_, upper), t in zip(travel, durations))

    thetas = (M.CERT_THETA1, M.CERT_THETA2)
    cs = (M.C1, M.C2)
    tol = dict(rtol=2e-11, atol=2e-13)
    solutions = {}
    for method in ("DOP853", "Radau"):
        z = M.X_D
        ys, ws = [M.X_D, M.X_D], [0.0, 0.0]
        for v, duration in zip(drives, durations):
            z = solve_ivp(lambda t, x: [M.f(x[0]) + v - M.RHO_BAR], (0, duration), [z],
                          method=method, **tol).y[0, -1]
            for i in range(2):
                share = thetas[i] / capacity   # proportional allocation
                rhs = lambda t, x: [M.f(x[0]) + v - cs[i] * M.h(x[1], thetas[i]),
                                    share * float(M.phi(v))]
                sol = solve_ivp(rhs, (0, duration), [ys[i], ws[i]], method=method, **tol)
                assert sol.success and np.max(sol.y[1]) < thetas[i]
                ys[i], ws[i] = sol.y[:, -1]
        assert min(ys) >= z > M.S + M.A_MARGIN

        # After withdrawal each output settles at the shifted healthy equilibrium.
        equilibria = []
        for i in range(2):
            drag = cs[i] * M.h(ws[i], thetas[i])
            root = brentq(lambda y: M.f(y) - drag, M.X_H - M.A_MARGIN, M.X_H)
            free = solve_ivp(lambda t, x: [M.f(x[0]) - drag], (0, 12), [ys[i]],
                             method=method, **tol)
            assert free.success and min(free.y[0]) >= M.S + M.A_MARGIN
            assert abs(free.y[0, -1] - root) < 1e-8
            assert abs(ws[i] / thetas[i] - exposure / capacity) < 1e-11
            equilibria.append(float(root))
        solutions[method] = dict(comparison_state_at_withdrawal=float(z),
                                 outputs_at_withdrawal=[float(y) for y in ys],
                                 damage_at_withdrawal=[float(w) for w in ws],
                                 final_equilibria=equilibria)
    gap = max(abs(a - b) for a, b in zip(solutions["DOP853"]["outputs_at_withdrawal"],
                                         solutions["Radau"]["outputs_at_withdrawal"]))
    assert gap < 2e-9
    return dict(drives=list(drives), durations=list(durations),
                switch_times=list(np.cumsum(durations)), exposure=exposure,
                capacity=capacity, capacity_scale=capacity / (M.THETA1 + M.THETA2),
                gap_percent=100 * (exposure / j_int - 1),
                travel_time_bounds=travel, solutions=solutions)


def check_backtracking(j_int):
    """A forward, backward, forward path still pays at least J_int to cross s."""
    state, positions = [M.X_D, 0.0], []
    for v, t in [(M.V_BAR, 1), (0, 1), (M.V_MAX, 1.2)]:
        sol = solve_ivp(lambda t, x: [M.f(x[0]) + v, float(M.phi(v))], (0, t), state,
                        rtol=1e-11, atol=1e-13)
        state = sol.y[:, -1]
        positions.append(float(state[0]))
    assert positions[1] < positions[0] and state[0] > M.S and state[1] >= j_int


# ---------------------------------------------------------------------------
# Noisy observations (Theorem 3 and Corollary 1)

def binary_kl(r, a):
    return xlogy(r, r / a) + xlogy(1 - r, (1 - r) / (1 - a))


def risk_bound(sigma, T=M.T_HORIZON, N=M.N_SAMPLES, r=M.REQUIRED_RESTORATION):
    """Lower bound on beta_1 + alpha_2 and the minimum failure allowance for success r."""
    d0 = float(M.D0(T))
    kl = N * d0**2 / (2 * sigma**2)
    allowance = brentq(lambda a: binary_kl(r, a) - kl, 1e-14, r, xtol=1e-14)
    assert abs(binary_kl(r, allowance) - kl) < 1e-10
    return dict(sigma=sigma, D0=d0, kl=kl, lower_bound=1 - np.sqrt(kl / 2),
                min_failure_allowance=allowance)
