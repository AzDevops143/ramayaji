#!/usr/bin/env python3
import argparse, os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

NAVY, TEAL, PURPLE, SLATE, SKY, GREEN = "#1F3B73", "#2A9D8F", "#6A4C93", "#5C677D", "#4EA8DE", "#2D6A4F"
COL = {"JOR": NAVY, "RB-GS": PURPLE, "RB-SOR": SKY, "GMG": TEAL}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.color": "#D9DEE7", "grid.linewidth": 0.6, "axes.edgecolor": "#5C677D",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})
COLW = 3.45

T4 = dict(bw_dram=230e9, bw_l2=600e9, l2=4 * 2**20, t_graph=2.5e-6, t_launch=4.5e-6, t_poll=25e-6)

def k_time(bytes_, ws, graph=True):
    bw = T4["bw_l2"] if ws <= 0.75 * T4["l2"] else T4["bw_dram"]
    return max(bytes_ / bw, T4["t_graph"] if graph else T4["t_launch"])

def hierarchy(N):
    s = [N]
    while s[-1] > 3 and (s[-1] - 1) % 2 == 0:
        s.append((s[-1] - 1) // 2 + 1)
    return s

def model_cycle(N, nu1=2, nu2=2, gamma=1, sb=8, graph=True):
    lv = hierarchy(N)
    def cyc(l):
        n = lv[l]; a = sb * n * n; ws = 3 * a
        if l == len(lv) - 1:
            cs = 1 if n <= 3 else max(4, 2 * n)
            return 2 * cs * k_time(2 * a, ws, graph)
        t = 2 * (nu1 + nu2) * k_time(2 * a, ws, graph)
        t += k_time(3 * a, ws, graph) + k_time(1.5 * a, ws, graph)
        t += gamma * cyc(l + 1)
        t += k_time(2.25 * a, ws, graph)
        return t
    a0 = sb * N * N
    return cyc(0) + k_time(4 * a0, 3 * a0, graph) + k_time(0, 0, graph)

def model_gmg(N, cycles, fmg=True, **kw):
    per = model_cycle(N, **kw)
    setup = 1.35 * per if fmg else 0.0
    return setup + cycles * per + math.ceil(cycles / 2) * T4["t_poll"], per

def model_sor(N, iters, sb=8):
    a = sb * N * N; per = 2 * k_time(2 * a, a, graph=False) + 1e-6
    return iters * per + math.ceil(iters / 200) * T4["t_poll"]

def load(d):
    p = lambda f: os.path.join(d, f)
    D = {"measured": os.path.exists(p("gpu_gmg.csv"))}
    D["cyc"] = pd.read_csv(p("gmg_cycles.csv"))
    D["base"] = pd.read_csv(p("base_iters.csv"))
    D["acc"] = pd.read_csv(p("accuracy.csv"))
    D["nu"] = pd.read_csv(p("nu.csv"))
    D["lev"] = pd.read_csv(p("levels.csv"))
    D["h_d"] = pd.read_csv(p("hist_gmg_d.csv")); D["h_f"] = pd.read_csv(p("hist_gmg_f.csv"))
    D["h_sor"] = pd.read_csv(p("hist_sor_1025.csv"))
    if D["measured"]:
        D["gpu"] = pd.read_csv(p("gpu_gmg.csv"))
        D["gsor"] = pd.read_csv(p("gpu_sor.csv"))
        D["device"] = open(p("device.txt")).read().strip() if os.path.exists(p("device.txt")) else "Tesla T4"
    else:
        D["device"] = "Tesla T4 (model)"
        c = D["cyc"]; rows, srows = [], []
        for N in sorted(c.N.unique()):
            cy = int(c[(c.variant == "V(2,2)+FMG") & (c.N == N) & (c.precision == "double")].cycles.iloc[0])
            for g in [1, 0]:
                t, per = model_gmg(N, cy, graph=bool(g))
                rows.append(dict(N=N, cycle="V", graph=g, cycles=cy, fmg_ms=1.35 * per * 1e3,
                                 solve_ms=t * 1e3, ms_per_cycle=per * 1e3))
            b = D["base"]; s = b[(b.solver == "RB-SOR") & (b.N == N)]
            if len(s):
                srows.append(dict(N=N, iters=int(s.iters.iloc[0]), kernel_ms=model_sor(N, int(s.iters.iloc[0])) * 1e3))
        D["gpu"] = pd.DataFrame(rows); D["gsor"] = pd.DataFrame(srows)
    return D

def tag(ax, D, loc="br"):
    if not D["measured"]:
        x, ha, y, va = (0.99, "right", 0.02, "bottom") if loc == "br" else (0.02, "left", 0.98, "top")
        ax.text(x, y, "T4 bandwidth model", transform=ax.transAxes, ha=ha, va=va, fontsize=6.5,
                color=SLATE, style="italic")

def fig_smoothing(out):
    n = 65; x = np.linspace(0, 1, n)
    e0 = np.sin(2 * np.pi * x) + 0.5 * np.sin(24 * np.pi * x); e0[0] = e0[-1] = 0
    def gs(e, k):
        e = e.copy()
        for _ in range(k):
            for i in range(1, n - 1): e[i] = 0.5 * (e[i - 1] + e[i + 1])
        return e
    fig, axs = plt.subplots(1, 3, figsize=(2 * COLW, 1.9), sharey=True)
    for ax, k, t in zip(axs[:2], [0, 3], ["(a) initial error", "(b) after 3 Gauss-Seidel sweeps"]):
        ax.plot(x, gs(e0, k), color=NAVY, lw=1.1); ax.set_title(t); ax.set_xlabel("x")
    e3 = gs(e0, 3)
    xc = x[::8]; axs[2].plot(x, e3, color=SLATE, lw=0.8, alpha=0.6, label="fine grid (65 pts)")
    axs[2].plot(xc, e3[::8], "o-", color=TEAL, ms=3.5, lw=1.2, label="coarse grid (9 pts)")
    axs[2].set_title("(c) same error, coarse grid"); axs[2].set_xlabel("x"); axs[2].legend(loc="lower left")
    axs[0].set_ylabel("error")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_gmg_smoothing.png")); plt.close(fig)

def fig_cycles(out):
    fig, axs = plt.subplots(1, 3, figsize=(2 * COLW, 1.9))
    def draw(ax, seq, title, fmg=False):
        xs = np.arange(len(seq)); ax.plot(xs, seq, "-", color=SLATE, lw=1.0, zorder=1)
        if fmg:
            best = seq[0]
            for k in range(1, len(seq)):
                if seq[k] < best:
                    ax.plot(xs[k - 1:k + 1], seq[k - 1:k + 1], "-", color=TEAL, lw=2.6, zorder=1.5)
                    best = seq[k]
        for x_, l in zip(xs, seq):
            c = TEAL if l == max(seq) else NAVY
            ax.scatter([x_], [l], s=28, color=c, zorder=2, marker="s" if l == max(seq) else "o")
        ax.invert_yaxis(); ax.set_title(title); ax.set_xticks([])
        ax.set_yticks(range(max(seq) + 1)); ax.set_yticklabels(["fine"] + [""] * (max(seq) - 1) + ["coarse"])
        ax.grid(False)
    draw(axs[0], [0, 1, 2, 3, 2, 1, 0], "V-cycle")
    draw(axs[1], [0, 1, 2, 3, 2, 3, 2, 1, 2, 3, 2, 3, 2, 1, 0], "W-cycle")
    draw(axs[2], [3, 2, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0], "Full multigrid (FMG)", fmg=True)
    axs[2].set_xlabel("thick lines: interpolate the solution one level up", fontsize=6.8, color=TEAL)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_gmg_cycles.png")); plt.close(fig)

def fig_iters(D, out):
    b, c = D["base"], D["cyc"]
    fig, ax = plt.subplots(figsize=(COLW, 2.5))
    for s in ["JOR", "RB-GS", "RB-SOR"]:
        d = b[b.solver == s].sort_values("N")
        if len(d): ax.loglog(d.N, d.iters, "o-", color=COL[s], ms=3.5, lw=1.2, label=s)
    for v, ls, mk in [("V(2,2)+FMG", "-", "s"), ("V(2,2)", "--", "D")]:
        d = c[(c.variant == v) & (c.precision == "double")].sort_values("N")
        ax.loglog(d.N, d.cycles, ls=ls, marker=mk, color=COL["GMG"], ms=3.5, lw=1.4, label=f"GMG {v} (cycles)")
    ax.set_xlabel("grid size N"); ax.set_ylabel("iterations / cycles to ε = 1e-4")
    Ns = sorted(c.N.unique()); ax.set_xticks(Ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.minorticks_off()
    ax.set_ylim(1, 2e6); ax.legend(ncol=2, fontsize=6.5, loc="upper left")
    ax.set_title("Iterations to converge")
    fig.savefig(os.path.join(out, "fig_gmg_iters.png")); plt.close(fig)

def wu_per_cycle(nu1=2, nu2=2):
    return (nu1 + nu2 + 2.0) * 4.0 / 3.0 + 1.0

def fig_history(D, out):
    fig, axs = plt.subplots(1, 2, figsize=(2 * COLW, 2.35))
    ax = axs[0]
    s = D["h_sor"]; ax.semilogy(s.iter, s.maxdiff, color=COL["RB-SOR"], lw=1.1, label="RB-SOR (1 iteration = 1 WU)")
    g = D["h_d"]; w = wu_per_cycle()
    ax.semilogy(g.cycle * w, g.maxdiff, "s-", color=COL["GMG"], ms=3.5, lw=1.3, label=f"GMG V(2,2) (1 cycle ≈ {w:.0f} WU)")
    ax.axhline(1e-4, color=SLATE, ls=":", lw=0.9); ax.text(1.0, 1.6e-4, "ε = 1e-4", fontsize=7, color=SLATE)
    ax.set_xscale("log"); ax.set_xlabel("work units (fine-grid sweeps)"); ax.set_ylabel("max |u$^{k}$ - u$^{k-1}$|")
    ax.set_title("(a) Cost to converge, N = 1025, float64"); ax.legend(loc="lower left", fontsize=6.8)
    ax = axs[1]
    ax.semilogy(D["h_d"].cycle, D["h_d"].maxres, "s-", color=TEAL, ms=3.5, lw=1.2, label="residual, float64")
    f = D["h_f"].iloc[:12]
    ax.semilogy(f.cycle, f.maxres, "o-", color=NAVY, ms=3.5, lw=1.2, label="residual, float32")
    ax.semilogy(f.cycle, f.maxdiff, "o--", color=PURPLE, ms=3.0, lw=1.0, label="max change, float32")
    ax.axhline(1e-4, color=SLATE, ls=":", lw=0.9)
    rho = D["cyc"][(D["cyc"].variant == "V(2,2)") & (D["cyc"].N == 1025) & (D["cyc"].precision == "double")].rho.iloc[0]
    ax.set_xlabel("V-cycle"); ax.set_title(f"(b) One V-cycle divides the residual by ~{1 / rho:.0f}")
    ax.legend(loc="center right", fontsize=6.8); ax.set_xlim(0.5, 12.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_gmg_history.png")); plt.close(fig)

def fig_tuning(D, out):
    fig, axs = plt.subplots(1, 2, figsize=(2 * COLW, 2.25))
    nu = D["nu"]; labs = [f"({a},{b})" for a, b in zip(nu.nu1, nu.nu2)]
    ax = axs[0]
    bars = ax.bar(labs, nu.rho, color=TEAL)
    for r, v, wv in zip(bars, nu.rho, [wu_per_cycle(a, b) for a, b in zip(nu.nu1, nu.nu2)]):
        eff = -math.log10(v) / wv
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.3f}\n{eff:.2f}", ha="center", va="bottom", fontsize=6.3)
    ax.set_xlabel("smoothing sweeps (ν1, ν2)"); ax.set_ylabel("convergence factor per V-cycle")
    ax.set_ylim(0, nu.rho.max() * 1.45)
    ax.set_title("(a) Smoothing: factor / digits per WU")
    ax = axs[1]; lv = D["lev"].sort_values("levels")
    ax.semilogy(lv.levels, lv.work, "o-", color=NAVY, ms=3.5, lw=1.2)
    for _, r in lv.iterrows():
        ax.annotate(f"{int(r.coarsest)}", (r.levels, r.work), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=6.5, color=SLATE)
    ax.set_xlabel("number of levels (label: coarsest grid size)"); ax.set_ylabel("work units to ε")
    ax.set_title("(b) Go all the way down, N = 257")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_gmg_tuning.png")); plt.close(fig)

def fig_time(D, out):
    g, s = D["gpu"], D["gsor"]
    fig, axs = plt.subplots(1, 2, figsize=(2 * COLW, 2.35))
    ax = axs[0]
    gg = g[(g.cycle == "V") & (g.graph == 1)].sort_values("N")
    ax.loglog(s.N, s.kernel_ms, "o-", color=COL["RB-SOR"], ms=3.5, lw=1.2, label="RB-SOR (ω*)")
    ax.loglog(gg.N, gg.solve_ms, "s-", color=COL["GMG"], ms=3.5, lw=1.4, label="GMG V(2,2)+FMG")
    ax.set_xlabel("grid size N"); ax.set_ylabel("GPU time to solution (ms)")
    ax.set_title("(a) Time to ε = 1e-4, float64"); ax.legend(loc="upper left"); tag(ax, D)
    ax.set_xticks(list(gg.N)); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off(); ax.tick_params(axis="x", labelsize=7)
    ax = axs[1]
    gn = g[(g.cycle == "V") & (g.graph == 0)].sort_values("N")
    x = np.arange(len(gg)); w = 0.38
    ax.bar(x - w / 2, gn.ms_per_cycle.values, w, color=SLATE, label="kernel launches")
    ax.bar(x + w / 2, gg.ms_per_cycle.values, w, color=TEAL, label="CUDA Graph")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(n) for n in gg.N])
    ax.set_xlabel("grid size N"); ax.set_ylabel("ms per V-cycle")
    ax.set_title("(b) Launch overhead and CUDA Graphs"); ax.legend(loc="center left"); tag(ax, D, "tl")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_gmg_time.png")); plt.close(fig)

def tables(D, out):
    c, b, a = D["cyc"], D["base"], D["acc"]
    Ns = sorted(c.N.unique())
    gsor = D["gsor"].set_index("N"); g = D["gpu"]
    gg = g[(g.cycle == "V") & (g.graph == 1)].set_index("N")
    L = [r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{RB-SOR} & \multicolumn{3}{c}{GMG V(2,2)+FMG} & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
         r"$N$ & iter. & ms & cycles & levels & ms & speed-up \\", r"\midrule"]
    for N in Ns:
        cc = c[(c.variant == "V(2,2)+FMG") & (c.N == N) & (c.precision == "double")]
        si = int(gsor.loc[N].iters) if N in gsor.index else None
        st = gsor.loc[N].kernel_ms if N in gsor.index else None
        gt = gg.loc[N].solve_ms if N in gg.index else None
        sp = f"{st / gt:.0f}$\\times$" if (st and gt) else "--"
        L.append(f"{N} & {si:,} & {st:,.1f} & {int(cc.cycles.iloc[0])} & {int(cc.levels.iloc[0])} & {gt:,.2f} & {sp} \\\\"
                 if si else f"{N} & -- & -- & {int(cc.cycles.iloc[0])} & {int(cc.levels.iloc[0])} & {gt:,.2f} & -- \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_gmg.tex"), "w").write("\n".join(L))

    L = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"Variant & 129 & 513 & 1025 & 4097 & $\rho$ \\", r"\midrule"]
    for v in ["V(1,1)+FMG", "V(2,2)+FMG", "W(2,2)+FMG", "V(2,2)"]:
        row = [v]
        for N in [129, 513, 1025, 4097]:
            d = c[(c.variant == v) & (c.N == N) & (c.precision == "double")]
            row.append(str(int(d.cycles.iloc[0])) if len(d) else "--")
        d = c[(c.variant == v) & (c.N == 1025) & (c.precision == "double")]
        row.append(f"{d.rho.iloc[0]:.3f}")
        L.append(" & ".join(row) + r" \\")
    d = c[(c.variant == "V(2,2)+FMG") & (c.precision == "float")]
    row = ["V(2,2)+FMG, float32"] + [str(int(d[d.N == N].cycles.iloc[0])) + ("" if int(d[d.N == N].converged.iloc[0]) else "$^\\dagger$")
                                     if len(d[d.N == N]) else "--" for N in [129, 513, 1025, 4097]] + ["--"]
    L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_gmg_cycles.tex"), "w").write("\n".join(L))

    M = {}
    M["GMeasured"] = "1" if D["measured"] else "0"
    M["GSource"] = ("measured on the Colab " + D["device"]) if D["measured"] else \
        "projected with the T4 bandwidth and launch model of Section~\\ref{sec:gmgimpl} (the notebook replaces them with measured values)"
    d = c[(c.variant == "V(2,2)+FMG") & (c.precision == "double")]
    M["GCycles"] = str(int(d.cycles.max())); M["GCyclesMin"] = str(int(d.cycles.min()))
    M["GNmax"] = str(int(d.N.max()))
    v = c[(c.variant == "V(2,2)") & (c.precision == "double") & (c.N == 1025)]
    M["GRho"] = f"{v.rho.iloc[0]:.3f}"; M["GRhoInv"] = f"{1 / v.rho.iloc[0]:.0f}"
    M["GCyclesNoFMG"] = str(int(c[(c.variant == "V(2,2)") & (c.precision == "double")].cycles.max()))
    def it(s, N):
        x = b[(b.solver == s) & (b.N == N)]; return int(x.iters.iloc[0]) if len(x) else None
    fi = lambda v: f"{v:,}" if v is not None else "--"
    M["GSorIt"] = fi(it('RB-SOR', 1025)); M["GJorIt"] = fi(it('JOR', 513)); M["GSorItSmall"] = fi(it('RB-SOR', 513))
    acc = a.set_index(["solver", "N"]).max_err
    M["GErr"] = f"{acc[('GMG', 1025)]:.1e}"; M["GErrSor"] = f"{acc[('RB-SOR', 1025)]:.1e}"
    Nb = int(gg.index.max()); M["GNb"] = str(Nb)
    if Nb in gsor.index:
        M["GSpeedup"] = f"{gsor.loc[Nb].kernel_ms / gg.loc[Nb].solve_ms:.0f}"
        M["GTSor"] = f"{gsor.loc[Nb].kernel_ms / 1e3:.2f}"; M["GTGmg"] = f"{gg.loc[Nb].solve_ms:.1f}"
    N1 = 1025
    M["GSpeedupK"] = f"{gsor.loc[N1].kernel_ms / gg.loc[N1].solve_ms:.0f}"
    M["GTGmgK"] = f"{gg.loc[N1].solve_ms:.2f}"; M["GTSorK"] = f"{gsor.loc[N1].kernel_ms:.0f}"
    gn = g[(g.cycle == "V") & (g.graph == 0)].set_index("N")
    Ns_ = sorted(gg.index)
    M["GGraphSmall"] = f"{gn.loc[Ns_[0]].ms_per_cycle / gg.loc[Ns_[0]].ms_per_cycle:.1f}"
    M["GGraphNsmall"] = str(Ns_[0])
    M["GMsCycle"] = f"{gg.loc[Nb].ms_per_cycle:.2f}"
    M["GWU"] = f"{wu_per_cycle():.0f}"
    f = c[(c.variant == "V(2,2)+FMG") & (c.precision == "float")]
    ok = f[f.converged == 1].N
    M["GFloatOK"] = str(int(ok.max())) if len(ok) else "--"
    fh = D["h_f"]; M["GFloatFloor"] = f"{fh.maxdiff.iloc[len(fh) // 2:].median():.1e}"
    nu = D["nu"]; best = nu.assign(e=[-math.log10(r) / wu_per_cycle(x, y) for r, x, y in zip(nu.rho, nu.nu1, nu.nu2)]).sort_values("e").iloc[-1]
    M["GBestNu"] = f"({int(best.nu1)},{int(best.nu2)})"
    lv = D["lev"].sort_values("levels")
    M["GLevWorkMax"] = f"{lv.work.iloc[0]:,.0f}"; M["GLevWorkMin"] = f"{lv.work.iloc[-1]:,.0f}"
    with open(os.path.join(out, "numbers_gmg.tex"), "w") as fh_:
        fh_.write("% generated by make_gmg_assets.py\n")
        for k, val in M.items(): fh_.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
    json.dump(M, open(os.path.join(out, "numbers_gmg.json"), "w"), indent=1)

    J = dict(measured=bool(D["measured"]), numbers=M,
             N=[int(n) for n in Ns],
             cycles={v: [int(c[(c.variant == v) & (c.N == n) & (c.precision == "double")].cycles.iloc[0]) for n in Ns]
                     for v in ["V(2,2)+FMG", "V(2,2)"]},
             base={s: {int(r.N): int(r.iters) for _, r in b[b.solver == s].iterrows()} for s in ["JOR", "RB-GS", "RB-SOR"]},
             gpu_gmg={int(n): float(gg.loc[n].solve_ms) for n in gg.index},
             gpu_sor={int(n): float(gsor.loc[n].kernel_ms) for n in gsor.index})
    json.dump(J, open(os.path.join(out, "deck_gmg.json"), "w"), indent=1)
    return M

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="gdata"); ap.add_argument("--out", default="assets")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = load(a.data)
    fig_smoothing(a.out); fig_cycles(a.out); fig_iters(D, a.out); fig_history(D, a.out)
    fig_tuning(D, a.out); fig_time(D, a.out)
    M = tables(D, a.out)
    print(("MEASURED" if D["measured"] else "MODEL") + " GMG assets ->", a.out)
    for k in ["GCycles", "GRho", "GSpeedupK", "GTGmgK", "GTSorK", "GSpeedup", "GGraphSmall", "GErr", "GFloatOK"]:
        print(f"  {k:12s} {M.get(k)}")

if __name__ == "__main__":
    main()