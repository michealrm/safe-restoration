"""Figure 1: overview of the model, drawn with the Section V maps."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from . import model as M
from .style import PALETTE as P, PDF_METADATA, use_times

RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.65,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
    "axes.grid": False, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": None, "savefig.pad_inches": 0,
}


def _draw():
    fig = plt.figure(figsize=(7, 2.45))
    a = fig.add_axes([0, 0, 1, 1])
    a.set(xlim=(0, 1), ylim=(0, 1))
    a.axis("off")

    def txt(x, y, s, **kw):
        return a.text(x, y, s, ha=kw.pop("ha", "center"), va="center",
                      color=kw.pop("color", P["ink"]), **kw)

    def arrow(p, q, color):
        a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                    color=color, lw=1.0, shrinkA=0, shrinkB=0))

    def axes(rect):
        ax = fig.add_axes(rect, facecolor="none")
        ax.tick_params(length=2, pad=2, labelcolor=P["ink"])
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(P["muted"])
        return ax

    for mid, title in [(0.1665, "(a) Intervention"), (0.501, "(b) Damage"),
                       (0.834, "(c) Output")]:
        txt(mid, 0.948, title, fontsize=10, weight="bold")
    arrow((0.3175, 0.948), (0.3485, 0.948), P["secondary"])
    arrow((0.6525, 0.948), (0.6835, 0.948), P["secondary"])

    # (a) A schematic constant drive withdrawn at T_w, and the damage-rate map.
    dax = axes([0.065, 0.71, 0.215, 0.105])
    dax.set_title(r"Drive $v(t)$", fontsize=8, pad=3, color=P["ink"])
    dax.axvline(0.65, color=P["secondary"], ls=(0, (2, 2)), lw=0.6)
    dax.plot([0, 0.65, 0.65, 1], [1, 1, 0, 0], color=P["input"], lw=1.4)
    dax.set(xlim=(0, 1.05), ylim=(-0.05, 1.5))
    dax.set_xticks([0.65], [r"$T_w$"])
    dax.set_yticks([0, 1], ["0", r"$v_c$"])
    dax.text(0.77, 0.16, "off", fontsize=8, color=P["ink"])
    dax.text(0.2, 1.12, "on", fontsize=8, color=P["ink"])
    dax.text(1.04, -0.34, r"$t$", ha="right", fontsize=8, color=P["ink"], clip_on=False)

    pax = axes([0.065, 0.32, 0.215, 0.18])
    v = np.linspace(0, M.V_MAX, 500)
    pax.set_title(r"Exposure rate $\varphi(v)$", fontsize=8, pad=4, color=P["ink"])
    pax.plot(v, M.phi(v), color=P["exposure"], lw=1.3)
    pax.set(xlim=(0, M.V_MAX), ylim=(-0.03, 1.1))
    pax.set_xticks([0, 2])
    pax.set_yticks([0, 1])
    pax.set_xlabel(r"$v$", labelpad=-8)

    # (b) Allocation, the two candidate damage equations, and h_i.
    txt(0.5, 0.833, r"Allocation $p\in[0,1]$", fontsize=8.5)
    txt(0.5, 0.775, r"$\mathcal{M}_1:\quad \dot w_1=p\varphi(v)$", fontsize=10)
    txt(0.5, 0.719, r"$\mathcal{M}_2:\quad \dot w_2=(1-p)\varphi(v)$", fontsize=10)
    txt(0.5, 0.659, "True model unknown.", fontsize=8.5)

    hax = axes([0.442, 0.32, 0.175, 0.18])
    w = np.linspace(0, M.THETA1, 300)
    hw = M.h(w, M.THETA1)
    hax.set_title(r"Damage-to-output map $h_i(w_i)$", fontsize=8, pad=4, color=P["ink"])
    hax.plot(w, hw, color=P["exposure"], lw=1.3)
    hax.set(xlim=(0, 1.75 * M.THETA1), ylim=(0, 1.1 * M.EPS_H))
    hax.set_xticks([0, M.THETA1], ["0", r"$\Theta_i$"])
    exponent = int(np.log10(M.EPS_H))
    hax.set_yticks([0, M.EPS_H], ["0", rf"$\varepsilon_h=10^{{{exponent}}}$"])
    hax.hlines(M.EPS_H, 0, M.THETA1, color=P["secondary"], linestyle=(0, (2, 2)), lw=0.55)
    hax.plot([M.THETA1], [hw[-1]], "o", ms=2.5, color=P["exposure"], clip_on=False)
    # Failure starts at w_i = Theta_i, so the map is drawn only up to there.
    hax.axvspan(M.THETA1, 1.75 * M.THETA1, color=P["threshold_fill"], zorder=0)
    hax.axvline(M.THETA1, color=P["threshold"], lw=0.7)
    hax.text(1.375 * M.THETA1, 0.46 * M.EPS_H, "Unsafe", ha="center", va="center",
             fontsize=8, color=P["threshold"])
    hax.set_xlabel(r"$w_i$")
    hax.xaxis.set_label_coords(0.27, -0.17)

    # (c) Output equation and the undamaged baseline field.
    txt(0.837, 0.833, r"Rate of measured output $y(t)$", fontsize=8.5)
    txt(0.837, 0.743, r"$\dot y=f(y)+v-c_i h_i(w_i)$", fontsize=11)
    fax = axes([0.725, 0.32, 0.245, 0.18])
    y = np.linspace(-0.15, 3.15, 700)
    fax.set_title(r"Baseline $f(y)=-y(y-1)(y-3)$", fontsize=8, pad=4, color=P["ink"])
    fax.axhline(0, color=P["baseline"], lw=0.6)
    fax.plot(y, M.f(y), color=P["ink"], lw=1.3)
    fax.plot([0], [0], "o", ms=3, color=P["muted"])
    fax.plot([3], [0], "o", ms=3, color=P["restoration"])
    fax.plot([1], [0], "o", ms=3, mfc="white", mec=P["secondary"])
    fax.set(xlim=(-0.15, 3.15), ylim=(-1.3, 2.4), ylabel=r"$f(y)$")
    fax.set_xticks([0, 1, 3], [r"$x_d=0$", r"$s=1$", r"$x_h=3$"])
    fax.set_yticks([-1, 0, 2])

    # Bottom row: what happens in each column after withdrawal.
    txt(0.17, 0.137, "No further exposure", weight="bold")
    txt(0.17, 0.066, r"$v=0\quad\Rightarrow\quad\varphi(v)=0$", fontsize=9)
    txt(0.5, 0.137, "Accumulated damage remains", weight="bold")
    txt(0.5, 0.066, r"$w_i(t)=w_i(T_w)$", fontsize=9)
    txt(0.837, 0.137, "Output continues to evolve", weight="bold")
    txt(0.837, 0.066, r"$\dot y=f(y)-c_i h_i(w_i(T_w))$", fontsize=9)

    # Make room for row labels on the left and a color key underneath.
    content_h, key_h = 2.45, 0.22
    total_h = content_h + key_h
    fig.set_size_inches(7, total_h)
    for ax in list(fig.axes):
        left, bottom, width, height = ax.get_position().bounds
        ax.set_position([0.12 + 0.87 * left, (bottom * content_h + key_h) / total_h,
                         0.87 * width, height * content_h / total_h])

    rows = fig.add_axes([0, key_h / total_h, 1, content_h / total_h], zorder=-1)
    rows.set(xlim=(0, 1), ylim=(0, 1))
    rows.axis("off")
    for bottom, top, label in [(0.625, 0.89, "General\nmodel"),
                               (0.235, 0.575, "Section V\ninstance"),
                               (0.015, 0.20, "After\nwithdrawal")]:
        rows.add_patch(Rectangle((0.005, bottom), 0.99, top - bottom,
                                 facecolor=P["neutral_fill"], edgecolor="none"))
        rows.plot([0.119, 0.119], [bottom + 0.015, top - 0.015], color=P["border"], lw=0.5)
        cy = (bottom + top) / 2
        last = bottom < 0.02
        rows.text(0.06, cy + (0.018 if last else 0), label, ha="center", va="center",
                  fontsize=8, weight="bold", color=P["secondary"], linespacing=1.15)
        if last:
            rows.text(0.06, cy - 0.066, r"$t\geq T_w$", ha="center", va="center",
                      fontsize=8, color=P["secondary"])

    key = fig.add_axes([0, 0, 1, key_h / total_h])
    key.set(xlim=(0, 1), ylim=(0, 1))
    key.axis("off")
    key.add_patch(FancyBboxPatch((0.008, 0.035), 0.984, 0.90,
                                 boxstyle="round,pad=0,rounding_size=0.005",
                                 facecolor="white", edgecolor=P["border"], lw=0.5))
    key.text(0.025, 0.45, "Color Key", ha="left", va="center", fontsize=8,
             weight="bold", color=P["secondary"])
    for x, role, label in [(0.19, "input", "Drive"),
                           (0.30, "exposure", "Exposure / damage"),
                           (0.52, "restoration", "Recovery"),
                           (0.66, "threshold", "Unsafe / ruled out"),
                           (0.855, "muted", "Reference")]:
        key.add_patch(FancyBboxPatch((x, 0.31), 0.014, 0.28,
                                     boxstyle="round,pad=0,rounding_size=0.004",
                                     facecolor=P[role], edgecolor="none"))
        key.text(x + 0.022, 0.45, label, ha="left", va="center", fontsize=8, color=P["ink"])
    return fig


def save(path):
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(RC)
        use_times()
        fig = _draw()
        fig.savefig(path, bbox_inches=None, metadata=PDF_METADATA)
        if str(path).endswith(".pdf"):
            fig.savefig(str(path)[:-4] + ".png", bbox_inches=None, dpi=300)
        plt.close(fig)
