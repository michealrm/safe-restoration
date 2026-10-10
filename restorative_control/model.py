"""The bistable instance of Section V.

Both candidate models share the measured output dynamics

    dy/dt = f(y) + v - c_i h_i(w_i),

and differ only in how the exposure phi(v) is routed to damage:

    M1:  dw1/dt = p phi(v),        M2:  dw2/dt = (1 - p) phi(v).
"""
import numpy as np
from scipy.optimize import brentq

# Baseline field f(y) = -y(y - 1)(y - 3).
X_D, S, X_H = 0.0, 1.0, 3.0          # degraded state, separator, healthy state
Y_MIN, Y_MAX = -1.0, 4.0             # invariant interval


def f(y):
    return -y * (y - 1.0) * (y - 3.0)


def fprime(y):
    return -(3.0 * y * y - 8.0 * y + 3.0)


X_STAR = (4.0 - np.sqrt(7.0)) / 3.0  # peak of the barrier -f on [x_d, s]
U_A = -f(X_STAR)                     # barrier level
NU_F = float(fprime(4.0 / 3.0))      # sup f' = 7/3
L_F = max(abs(fprime(Y_MIN)), abs(fprime(Y_MAX)))  # sup |f'| on [-1, 4] = 19

# Damage-rate map.
V_MAX, V0, K, PHI_MAX = 2.0, 0.4, 25.0, 1.0
V_BAR = 0.3                          # low-drive level

_PHI_DEN = np.tanh(K * (V_MAX - V0)) + np.tanh(K * V0)


def phi(v):
    return PHI_MAX * (np.tanh(K * (np.asarray(v) - V0)) + np.tanh(K * V0)) / _PHI_DEN


def phiprime(v):
    return PHI_MAX * K / (_PHI_DEN * np.cosh(K * (np.asarray(v) - V0)) ** 2)


def phi_stable(v):
    """Same function as phi, rewritten to avoid cancellation near v = 0."""
    v = np.asarray(v)
    e0 = np.exp(-2 * K * V0)
    top = e0 * np.expm1(2 * K * v)
    bot = (1 + np.exp(2 * K * (v - V0))) * (1 + e0)
    scale = 1 / (1 + np.exp(-2 * K * (V_MAX - V0))) - e0 / (1 + e0)
    return PHI_MAX * top / bot / scale


# Damage tolerances and the damage-to-output map.
THETA1 = THETA2 = 0.05
C1 = C2 = 1.0
EPS_H = 1e-6
RHO_BAR = max(C1, C2) * EPS_H        # largest possible drag before failure


def h(w, theta, eps_h=EPS_H):
    return eps_h * np.tanh(np.asarray(w) / theta) / np.tanh(1.0)


# Crossing band of Lemma 2 at level u_a - delta = 0.5.
def band(delta):
    """Component of {-f >= u_a - delta} in [x_d, s] that contains x*."""
    g = lambda y: -f(y) - (U_A - delta)
    return brentq(g, X_D, X_STAR, xtol=1e-14), brentq(g, X_STAR, S, xtol=1e-14)


def J_esc(delta):
    a, b = band(delta)
    return float(phi(U_A - delta)) * (b - a) / (V_MAX - U_A + delta)


DELTA = U_A - 0.5
A_DELTA, B_DELTA = band(DELTA)
ELL_DELTA = B_DELTA - A_DELTA
J_ESC = J_esc(DELTA)

# Restoration margin and the persistence radius rho*(a) of Lemma 1.
A_MARGIN = 0.05
PERSIST_A0 = 0.051


def persistence_radius(a=A_MARGIN, a0=PERSIST_A0):
    if not 0 < a < a0:
        raise ValueError("persistence requires 0 < a < a0")
    roots = (X_D, S, X_H)
    # f' is monotone on each a0-neighborhood, and |f| is unimodal between
    # the excluded neighborhoods, so both minima occur at interval endpoints.
    m = 2 * min(abs(fprime(x + d)) for x in roots for d in (-a0, a0))
    mu = min(abs(f(x)) for x in [Y_MIN, Y_MAX] + [r + d for r in roots for d in (-a, a)])
    return min(mu / 2, m * a / 4)


RHO_STAR = persistence_radius()

# Known-model restoration with constant drive V and zero damage.
V_RESTORE = 1.5
T_W = (S + A_MARGIN - X_D) / (V_RESTORE - U_A)

# Low-high-low comparison drive for the sufficient certificate (Proposition 1),
# with each tolerance raised to 0.26.
CERT_DRIVES = (0.28415, V_MAX, 0.28215)
CERT_DURATIONS = (1.6962, 0.49971, 2.5298)
CERT_TW = sum(CERT_DURATIONS)
CERT_EXPOSURE = sum(float(phi(v)) * t for v, t in zip(CERT_DRIVES, CERT_DURATIONS))
CERT_THETA1 = CERT_THETA2 = 0.26

# Sampled Gaussian observations (Theorem 3).
T_HORIZON = 2.0
N_SAMPLES = 100
SIGMAS = (0.01, 0.001)
REQUIRED_RESTORATION = 0.9


def D0(t, eps_h=EPS_H):
    """Output-separation bound under a common input."""
    return (C1 + C2) * eps_h * np.expm1(NU_F * t) / NU_F
