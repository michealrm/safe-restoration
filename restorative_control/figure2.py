"""Figure 2: barrier, damage rate, withdrawal outcomes and the two certificates."""
import matplotlib._mathtext as mathtext
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from . import model as M
from .style import PALETTE as P, PDF_METADATA, use_times

COLUMN_WIDTH = 3.40  # inches
INK, INK2 = P["ink"], P["secondary"]

RC = {
    "text.color": INK, "axes.labelcolor": INK,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.45,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def withdrawal_family(horizon=3.5):
    """Model M1 with p = 0 and drive V, withdrawn at four different times.

    With p = 0 the damage w1 stays at zero, so the driven phase is
    dy/dt = f(y) + V and the free phase is dy/dt = f(y). Maintaining the
    same drive would fail M2 at Theta2 / phi(V), and exhausts common safety
    at (Theta1 + Theta2) / phi(V).
    """
    V = M.V_RESTORE
    rate = float(M.phi(V))
    common_limit = (M.THETA1 + M.THETA2) / rate
    driven = solve_ivp(lambda t, y: [M.f(y[0]) + V], (0, M.T_W), [M.X_D],
                       rtol=1e-10, atol=1e-12, dense_output=True,
                       events=lambda t, y: y[0] - M.S)
    assert driven.success and len(driven.t_events[0]) == 1
    t_separator = float(driven.t_events[0][0])

    # Safe under both, too early to recover, and two recoveries under M1.
    times = [0.4 * common_limit, t_separator - 0.15, t_separator + 0.08, M.T_W]
    runs = []
    for tw in times:
        yw = float(driven.sol(tw)[0])
        recovered = yw > M.S
        assert abs(yw - M.S) > M.A_MARGIN
        free = solve_ivp(lambda t, y: [M.f(y[0])], (tw, horizon), [yw],
                         rtol=1e-10, atol=1e-12, dense_output=True)
        assert free.success
        t = np.linspace(tw, horizon, 500)
        y = free.sol(t)[0]
        assert abs(y[-1] - (M.X_H if recovered else M.X_D)) < 0.01
        runs.append((tw, yw, t, y, recovered))
    assert rate * times[0] < min(M.THETA1, M.THETA2)
    assert common_limit < t_separator
    assert rate * times[2] > M.J_ESC > M.THETA1 + M.THETA2
    summary = dict(drive=V, common_safety_limit=common_limit,
                   M2_failure_time=M.THETA2 / rate, separator_time=t_separator,
                   withdrawal_times=times,
                   recovered=[r[4] for r in runs])
    return driven, runs, summary


def _draw(j_int):
    fig = plt.figure(figsize=(COLUMN_WIDTH, 3.65))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1.0], height_ratios=[1.0, 1.08],
                          wspace=0.10, hspace=0.55,
                          left=0.205, right=0.98, bottom=0.335, top=0.955)
    axh = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1], sharey=axh)
    axc = fig.add_subplot(gs[1, :])

    # (a) Barrier -f(y) with the crossing band of Lemma 2.
    y = np.linspace(M.X_D, M.S, 400)
    level = M.U_A - M.DELTA
    yb = np.linspace(M.A_DELTA, M.B_DELTA, 200)
    # Draw each shaded outline once, above guides and below the main curve.
    axh.fill_between(yb, level, -M.f(yb), facecolor=P["exposure_fill"],
                     edgecolor=P["exposure"], hatch="///", lw=0.45, zorder=2)
    axh.plot(y, -M.f(y), color=INK, lw=1.2, zorder=3)
    for lv in (M.U_A, level, M.V_BAR):
        axh.axhline(lv, color=P["guide"], lw=0.45, ls=(0, (1, 2.5)), zorder=1)
        axp.axhline(lv, color=P["guide"], lw=0.45, ls=(0, (1, 2.5)), zorder=1)
    axh.annotate("", xy=(M.A_DELTA, 0.06), xytext=(M.B_DELTA, 0.06),
                 arrowprops=dict(arrowstyle="|-|,widthA=0.15,widthB=0.15",
                                 lw=0.7, color=INK2, shrinkA=0, shrinkB=0))
    axh.text((M.A_DELTA + M.B_DELTA) / 2, 0.10, r"$\ell_\delta$",
             ha="center", va="bottom", fontsize=8)
    for edge in (M.A_DELTA, M.B_DELTA):
        axh.plot([edge, edge], [0.06, level], color=P["guide"], lw=0.45, ls=(0, (1, 2.5)), zorder=1)
    axh.text(0.86, 0.40, r"$-f(y)$", ha="center", fontsize=8, color=INK)
    axh.set_xlim(M.X_D, M.S)
    axh.set_ylim(0, M.U_A + 0.06)
    axh.set_xticks([0, 0.5, 1])
    axh.set_xticklabels([r"$x_d$", "0.5", r"$s$"])
    axh.set_yticks([0, M.V_BAR, level, M.U_A])
    axh.set_yticklabels(["0", r"$\bar v$", r"$u_a-\delta$", r"$u_a$"])
    axh.set_xlabel("output $y$", labelpad=1.5)
    axh.set_ylabel("drive $v$", labelpad=2)

    # (b) Damage rate against drive, on the same vertical axis.
    v = np.linspace(0, M.U_A, 400)
    axp.plot(M.phi(v), v, color=P["exposure"], lw=1.2, zorder=3)
    # Include the exact band edge so its outline coincides with the guide.
    shaded_v = np.r_[level, v[v > level]]
    axp.fill_betweenx(shaded_v, 0, M.phi(shaded_v),
                      facecolor=P["exposure_fill"], edgecolor=P["exposure"],
                      hatch="///", lw=0.45, zorder=2)
    axp.set_xlim(0, 1.08)
    axp.set_xticks([0, 1])
    axp.set_xlabel(r"damage rate $\varphi(v)$", labelpad=1.5)
    axp.tick_params(labelleft=False, left=False)
    axp.spines["left"].set_visible(False)

    # (c) Withdrawal outcomes under M1 with p = 0.
    driven, runs, summary = withdrawal_family()
    limit, horizon = summary["common_safety_limit"], 3.5
    axc.axhspan(M.S - M.A_MARGIN, M.S + M.A_MARGIN, color=P["neutral_fill"], lw=0, zorder=0)
    axc.axhline(M.S, color=P["guide"], lw=0.45, ls=(0, (1, 2.5)), zorder=1)
    axc.axvline(limit, color=P["threshold"], lw=0.9, ls=(0, (3, 2)))
    for tw, yw, t, yv, recovered in runs:
        color = P["restoration"] if recovered else P["muted"]
        axc.plot(t, yv, color=color, lw=1.3 if recovered else 1.2)
        axc.plot(tw, yw, marker="o", ms=3, mfc="white", mec=color, mew=0.85, zorder=5)
    on = np.linspace(0, runs[-1][0], 400)
    axc.plot(on, driven.sol(on)[0], color=P["input"], lw=1.3)
    axc.text(limit + 0.10, 3.25, "common-safety limit", color=P["threshold"],
             va="top", fontsize=8)
    axc.text(0.38, 0.12, "drive on", fontsize=8, color=P["input"])
    axc.annotate("drive off", xy=(runs[1][0], runs[1][1]), xytext=(0.75, 2.25),
                 ha="center", va="center", fontsize=8,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    axc.text(horizon - 0.03, M.X_H - 0.10, "recovered", ha="right", va="top",
             fontsize=8, color=P["restoration"])
    axc.text(horizon - 0.03, M.X_D + 0.12, "degraded", ha="right", va="bottom",
             fontsize=8, color=INK2)
    axc.set_xlim(0, horizon)
    axc.set_ylim(-0.1, 3.4)
    axc.set_xticks([0, 1, 2, 3])
    axc.set_yticks([M.X_D, M.S, M.X_H])
    axc.set_yticklabels([r"$x_d$", r"$s$", r"$x_h$"])
    axc.set_xlabel("time $t$", labelpad=1.5)
    axc.set_ylabel("output $y$", labelpad=3)

    # (d) Total tolerance axis with a break, so the small gap between the
    # certificates is visible. Both segments use the same scale.
    cap = M.THETA1 + M.THETA2
    bounds = fig.add_axes([0.10, 0.060, 0.86, 0.18])
    bounds.set(xlim=(0, 1), ylim=(-1.1, 1.0))
    bounds.axis('off')
    assert cap < j_int < M.CERT_EXPOSURE
    scale = 0.70 / 0.026
    left_end = scale * 0.008

    def zoom(value):
        return 0.30 + scale * (value - 0.498)

    lower, upper = zoom(j_int), zoom(M.CERT_EXPOSURE)
    original = scale * (cap - 0.096)
    for left, right, color in [
        (0, left_end, P['threshold']),
        (0.30, lower, P['threshold']),
        (lower, upper, P['muted']),
        (upper, 1, P['restoration'])
    ]:
        bounds.plot([left,right],[0.32,0.32],color=color,lw=1.4,
                    solid_capstyle='butt')
    # Two slashes mark the omitted capacity range, not a change of scale.
    for x in (0.248, 0.268):
        bounds.plot([x-0.007, x+0.007], [0.20, 0.44], color=INK2, lw=0.8)
    for value, symbol, number in [
        (lower, r'$J_{\mathrm{int}}$', f'{j_int:.4f}'),
        (upper, r'$\bar J$', f'{M.CERT_EXPOSURE:.4f}')
    ]:
        bounds.plot([value, value], [0.12, 0.52], color=INK2, lw=0.8)
        bounds.text(value, -0.33, symbol, ha='center', va='top', fontsize=8)
        bounds.text(value, -0.75, number, ha='center', va='top', fontsize=8, color=P['secondary'])
    bounds.text(0, 0.73, 'Ruled out', ha='left', va='center', fontsize=8)
    bounds.text((lower+upper)/2, 0.73, 'Gap', ha='center', va='center', fontsize=8)
    bounds.text((upper+1)/2, 0.73, 'Sufficient', ha='center', va='center', fontsize=8)
    for x, value, label, color in [(original, cap, 'Example tolerance', P['threshold'])]:
        bounds.plot(x, 0.32, 'o', color=color, ms=3.5)
        bounds.plot([x, x], [0.18, -0.1575], color=color, lw=0.6)
        bounds.text(x, -0.33, label, ha='center', va='top', fontsize=8)
        bounds.text(x, -0.75, f'{value:g}', ha='center', va='top', fontsize=8,
                    color=P['secondary'])
    bounds.text(0.5, -1.22, r'Total tolerance $\Theta_1+\Theta_2$',
                ha='center', va='top', fontsize=8)
    bounds.text(0, 1.0, '(d)', ha='left', va='bottom', fontsize=8)
    for a, label in ((axh, "(a)"), (axp, "(b)"), (axc, "(c)")):
        a.text(0.0, 1.03, label, transform=a.transAxes, ha="left", va="bottom", fontsize=8)
    return fig


def save(path, j_int):
    # Scripts at 7/8 of the base size (8 pt text, 7 pt sub/superscripts).
    shrink = mathtext.SHRINK_FACTOR
    try:
        mathtext.SHRINK_FACTOR = 0.875
        with plt.rc_context():
            plt.rcdefaults()
            plt.rcParams.update(RC)
            use_times()
            fig = _draw(j_int)
            fig.savefig(path, metadata=PDF_METADATA)
            if str(path).endswith(".pdf"):
                fig.savefig(str(path)[:-4] + ".png", dpi=300)
            plt.close(fig)
    finally:
        mathtext.SHRINK_FACTOR = shrink
