#!/usr/bin/env python3
import argparse, os, math, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, PURPLE, SLATE, SKY, GREEN = "#1F3B73", "#2A9D8F", "#6A4C93", "#5C677D", "#4EA8DE", "#2D6A4F"
SOLVER_COL = {"JOR": NAVY, "RB-GS": PURPLE, "RB-SOR": TEAL}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.color": "#D9DEE7", "grid.linewidth": 0.6, "axes.edgecolor": "#5C677D",
    "axes.prop_cycle": matplotlib.cycler(color=[NAVY, TEAL, PURPLE, SLATE, SKY, GREEN]),
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})
COLW = 3.45

T4 = dict(bw_dram=230e9, bw_l2=600e9, l2_bytes=4 * 2**20, t_launch=4.0e-6,
          t_gap=1.0e-6, t_sync=25e-6)

def model_iter_time(N, solver, prec_bytes=8):
    cells = (N - 2) ** 2
    if solver == "JOR":
        traffic = 2 * prec_bytes * cells
        ws, kernels = 2 * prec_bytes * N * N, 1
    else:
        traffic = 2 * 2 * prec_bytes * cells
        ws, kernels = prec_bytes * N * N, 2
    bw = T4["bw_l2"] if ws <= 0.75 * T4["l2_bytes"] else T4["bw_dram"]
    per_kernel = max(traffic / kernels / bw, T4["t_launch"]) + T4["t_gap"]
    return kernels * per_kernel

def model_total(N, solver, iters, K=200, prec_bytes=8):
    return iters * model_iter_time(N, solver, prec_bytes) + math.ceil(iters / K) * T4["t_sync"]

def solver_name(method, omega):
    if method == "jor":
        return "JOR"
    return "RB-GS" if abs(omega - 1.0) < 1e-9 else "RB-SOR"

def load(data):
    D = {}
    p = lambda f: os.path.join(data, f)
    D["measured"] = os.path.exists(p("gpu_conv.csv"))
    D["iters"] = pd.read_csv(p("iters.csv"))
    D["omega"] = pd.read_csv(p("omega.csv"))
    D["hist"] = {k: pd.read_csv(p(f"hist_{k}.csv")) for k in
                 ["jor_256", "gs_256", "sor_256", "sorf", "sord"] if os.path.exists(p(f"hist_{k}.csv"))}
    fld = [f for f in os.listdir(data) if f.startswith("field_") and f.endswith(".bin")]
    if fld:
        f = sorted(fld)[-1]; n = int(f[6:-4])
        D["field"] = np.fromfile(p(f), dtype=np.float64).reshape(n, n)
    if D["measured"]:
        g = pd.read_csv(p("gpu_conv.csv"))
        g["solver"] = [solver_name(m, w) for m, w in zip(g.method, g.omega)]
        D["conv"] = g
        D["bench"] = pd.read_csv(p("gpu_bench.csv"))
        D["batch"] = pd.read_csv(p("gpu_batch.csv"))
        D["device"] = open(p("device.txt")).read().strip() if os.path.exists(p("device.txt")) else "Tesla T4"
    else:
        D["conv"], D["bench"], D["batch"] = model_tables(D["iters"])
        D["device"] = "Tesla T4 (bandwidth model)"
    return D

def model_tables(it):
    rows = []
    for _, r in it[it.precision == "double"].iterrows():
        for v in ["global", "shared"]:
            t = model_total(r.N, r.solver, r.iters) * 1e3
            cells = (r.N - 2) ** 2 * r.iters
            rows.append(dict(method="jor" if r.solver == "JOR" else "rbsor", variant=v,
                             precision="double", N=r.N, solver=r.solver, iters=r.iters,
                             kernel_ms=t, MLUPS=cells / (t * 1e3), GBps=cells * 16 / (t * 1e6)))
    conv = pd.DataFrame(rows)
    rows = []
    for N in [256, 512, 1024, 2048, 4096]:
        for pb, pn in [(4, "float"), (8, "double")]:
            for s, m in [("JOR", "jor"), ("RB-SOR", "rbsor")]:
                for v in ["global", "shared"]:
                    t = model_total(N, s, 2000, prec_bytes=pb) * 1e3
                    cells = (N - 2) ** 2 * 2000
                    rows.append(dict(method=m, variant=v, precision=pn, N=N, iters=2000, kernel_ms=t,
                                     MLUPS=cells / (t * 1e3), GBps=cells * 2 * pb / (t * 1e6)))
    bench = pd.DataFrame(rows)
    sor512 = int(it[(it.solver == "RB-SOR") & (it.N == 512) & (it.precision == "double")].iters.iloc[0])
    rows = [dict(batch=K, N=512, iters=sor512, kernel_ms=model_total(512, "RB-SOR", sor512, K=K) * 1e3)
            for K in [1, 2, 5, 10, 50, 200, 1000]]
    return conv, bench, pd.DataFrame(rows)

def tag(ax, D, loc="br"):
    if not D["measured"]:
        x, ha, y, va = (0.99, "right", 0.02, "bottom") if loc == "br" else (0.02, "left", 0.98, "top")
        ax.text(x, y, "T4 bandwidth model", transform=ax.transAxes, ha=ha, va=va,
                fontsize=6.5, color=SLATE, style="italic")

def fig_field(D, out):
    if "field" not in D: return
    T = D["field"]; n = T.shape[0]
    fig, ax = plt.subplots(figsize=(COLW, COLW * 0.86))
    im = ax.imshow(T, cmap="viridis", origin="upper", extent=[0, 1, 1, 0])
    cs = ax.contour(np.linspace(0, 1, n), np.linspace(0, 1, n), T, levels=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                    colors="white", linewidths=0.5)
    ax.clabel(cs, fontsize=6, fmt="%d")
    ax.set_xlabel("x  (left 75 °C, right 50 °C)"); ax.set_ylabel("y  (top 100 °C, bottom 0 °C)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("T (°C)")
    ax.set_title(f"Converged field, N = {n}, RB-SOR")
    fig.savefig(os.path.join(out, "fig_field.png")); plt.close(fig)

def fig_history(D, out):
    H = D["hist"]
    fig, axs = plt.subplots(1, 2, figsize=(2 * COLW, 2.3))
    ax = axs[0]
    for k, lab, c in [("jor_256", "JOR (ω = 1)", NAVY), ("gs_256", "RB-GS (ω = 1)", PURPLE),
                      ("sor_256", "RB-SOR (ω opt)", TEAL)]:
        if k in H: ax.semilogy(H[k].iter, H[k].maxdiff, color=c, lw=1.1, label=lab)
    ax.axhline(1e-4, color=SLATE, ls=":", lw=0.9); ax.text(1.0, 1.5e-4, "ε = 1e-4",
            ha="left", fontsize=7, color=SLATE)
    ax.set_xscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("max |T$^{k+1}$ - T$^k$|")
    ax.set_title("(a) Convergence, N = 256, float64"); ax.legend(loc="center left")
    ax = axs[1]
    for k, lab, c, ls, lw in [("sorf", "RB-SOR float32", NAVY, "-", 0.8), ("sord", "RB-SOR float64", TEAL, "--", 1.6)]:
        if k in H:
            h = H[k].iloc[:1500]
            ax.semilogy(h.iter, h.maxdiff, color=c, lw=lw, ls=ls, label=lab)
    if "sorf" in H:
        f = H["sorf"]; tail = f.maxdiff.iloc[len(f) // 2:]
        ax.annotate(f"float32 plateau ~{tail.median():.1e}\n(never reaches \u03b5 reliably)", xy=(800, tail.median()),
                    xytext=(500, 3e-1), fontsize=7, color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.7))
    ax.axhline(1e-4, color=SLATE, ls=":", lw=0.9)
    n = D.get("sorf_N", 64)
    ax.set_xlabel("iteration"); ax.set_title(f"(b) Precision floor of SOR (ω opt), N = {n}")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_history.png")); plt.close(fig)

def fig_iters(D, out):
    it = D["iters"][D["iters"].precision == "double"]
    fig, ax = plt.subplots(figsize=(COLW, 2.5))
    slopes = {}
    for s in ["JOR", "RB-GS", "RB-SOR"]:
        d = it[it.solver == s].sort_values("N")
        if len(d) < 2: continue
        k = np.polyfit(np.log(d.N), np.log(d.iters), 1)[0]; slopes[s] = k
        ax.loglog(d.N, d.iters, "o-", color=SOLVER_COL[s], ms=3.5, lw=1.2, label=f"{s}  $\\sim N^{{{k:.2f}}}$")
    ax.set_xlabel("grid size N"); ax.set_ylabel("iterations to ε = 1e-4")
    ax.set_xticks(sorted(it.N.unique())); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off(); ax.legend(loc="upper left"); ax.set_title("Iteration count scaling")
    fig.savefig(os.path.join(out, "fig_iters.png")); plt.close(fig)
    return slopes

def fig_omega(D, out):
    o = D["omega"].sort_values("omega")
    N = int(o.N.iloc[0]); wopt = 2 / (1 + math.sin(math.pi / (N - 1)))
    fig, ax = plt.subplots(figsize=(COLW, 2.3))
    ax.semilogy(o.omega, o.iters, "o-", color=TEAL, ms=3.5, lw=1.2)
    ax.axvline(wopt, color=NAVY, ls="--", lw=0.9)
    ax.text(wopt - 0.01, o.iters.max() * 0.7, f"ω* = {wopt:.4f}", ha="right", fontsize=7.5, color=NAVY)
    ax.set_xlabel("relaxation factor ω"); ax.set_ylabel("iterations")
    ax.set_title(f"Red-Black SOR, N = {N}, float64")
    fig.savefig(os.path.join(out, "fig_omega.png")); plt.close(fig)

def fig_time(D, out):
    c = D["conv"][D["conv"].precision == "double"]
    fig, ax = plt.subplots(figsize=(COLW, 2.5))
    for s in ["JOR", "RB-GS", "RB-SOR"]:
        for v, ls, mk in [("global", "-", "o"), ("shared", "--", "s")]:
            d = c[(c.solver == s) & (c.variant == v)].sort_values("N")
            if len(d) == 0: continue
            ax.loglog(d.N, d.kernel_ms / 1e3, ls=ls, marker=mk, ms=3, lw=1.1, color=SOLVER_COL[s],
                      label=f"{s} {v}")
    ax.set_xlabel("grid size N"); ax.set_ylabel("time to solution (s)")
    ax.legend(ncol=2, fontsize=6.5); ax.set_title("GPU time to ε = 1e-4, float64"); tag(ax, D)
    fig.savefig(os.path.join(out, "fig_time.png")); plt.close(fig)

def fig_bw(D, out):
    b = D["bench"]; Nmax = b.N.max()
    d = b[b.N == Nmax]
    labels, vals_f, vals_d = [], [], []
    for m, mn in [("jor", "JOR"), ("rbsor", "RB-SOR")]:
        for v in ["global", "shared"]:
            labels.append(f"{mn}\n{v}")
            vals_f.append(d[(d.method == m) & (d.variant == v) & (d.precision == "float")].MLUPS.mean() / 1e3)
            vals_d.append(d[(d.method == m) & (d.variant == v) & (d.precision == "double")].MLUPS.mean() / 1e3)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(COLW, 2.4))
    b1 = ax.bar(x - w / 2, vals_f, w, color=SKY, label="float32")
    b2 = ax.bar(x + w / 2, vals_d, w, color=NAVY, label="float64")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():.1f}",
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("GLUPS (10$^9$ cell updates/s)")
    ax.set_title(f"Kernel throughput, N = {Nmax}, 2000 iterations"); ax.legend(); tag(ax, D, "tl")
    ax.set_ylim(0, max(vals_f) * 1.18)
    fig.savefig(os.path.join(out, "fig_bw.png")); plt.close(fig)

def fig_batch(D, out):
    b = D["batch"].sort_values("batch")
    fig, ax = plt.subplots(figsize=(COLW, 2.2))
    ax.semilogx(b.batch, b.kernel_ms, "o-", color=PURPLE, ms=3.5, lw=1.2)
    ax.set_xlabel("iterations per convergence poll K"); ax.set_ylabel("time to solution (ms)")
    ax.set_title(f"Host polling interval, RB-SOR, N = {int(b.N.iloc[0])}"); tag(ax, D)
    ax.set_ylim(0, b.kernel_ms.max() * 1.1)
    fig.savefig(os.path.join(out, "fig_batch.png")); plt.close(fig)

def fmt_ms(x):
    return f"{x:,.1f}" if x < 1e4 else f"{x:,.0f}"

def tables(D, out, slopes):
    it = D["iters"]; dd = it[it.precision == "double"]
    Ns = sorted(dd.N.unique())
    L = [r"\begin{tabular}{@{}rrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Iterations to $\varepsilon=10^{-4}$} & \multicolumn{2}{c}{Max error vs. exact discrete} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
         r"$N$ & JOR & RB-GS & RB-SOR & JOR & RB-SOR \\", r"\midrule"]
    for N in Ns:
        g = lambda s, col: dd[(dd.solver == s) & (dd.N == N)][col]
        cell = lambda s: f"{int(g(s, 'iters').iloc[0]):,}" if len(g(s, 'iters')) else "--"
        def err(s):
            v = g(s, 'max_err')
            if not len(v) or np.isnan(v.iloc[0]): return "--"
            x = v.iloc[0]
            return f"{x:.2f}" if x >= 0.1 else f"{x:.1e}".replace("e-0", "e-")
        L.append(f"{N} & {cell('JOR')} & {cell('RB-GS')} & {cell('RB-SOR')} & {err('JOR')} & {err('RB-SOR')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_iters.tex"), "w").write("\n".join(L))

    c = D["conv"][D["conv"].precision == "double"]
    if not D["measured"]:
        L = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Solver & $N$ & Iterations & Model time (ms) & Model GB/s \\", r"\midrule"]
        for s in ["JOR", "RB-GS", "RB-SOR"]:
            for N in sorted(c[c.solver == s].N.unique()):
                gg = c[(c.solver == s) & (c.N == N) & (c.variant == "global")]
                L.append(f"{s} & {N} & {int(gg.iters.iloc[0]):,} & {fmt_ms(gg.kernel_ms.iloc[0])} & {gg.GBps.iloc[0]:.0f} \\\\")
            L.append(r"\midrule")
        L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
        open(os.path.join(out, "tab_gpu.tex"), "w").write("\n".join(L))
    else:
      L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
           r"Solver & $N$ & Iter. & Global (ms) & Shared (ms) & Gl./Sh. & GB/s \\", r"\midrule"]
      for s in ["JOR", "RB-GS", "RB-SOR"]:
          for N in sorted(c[c.solver == s].N.unique()):
              gg = c[(c.solver == s) & (c.N == N) & (c.variant == "global")]
              ss = c[(c.solver == s) & (c.N == N) & (c.variant == "shared")]
              if len(gg) == 0 or len(ss) == 0: continue
              tg, ts = gg.kernel_ms.iloc[0], ss.kernel_ms.iloc[0]
              L.append(f"{s} & {N} & {int(gg.iters.iloc[0]):,} & {fmt_ms(tg)} & {fmt_ms(ts)} & "
                       f"{tg / ts:.2f}$\\times$ & {max(gg.GBps.iloc[0], ss.GBps.iloc[0]):.0f} \\\\")
          L.append(r"\midrule")
      L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
      open(os.path.join(out, "tab_gpu.tex"), "w").write("\n".join(L))

    M = {}
    M["Measured"] = "1" if D["measured"] else "0"
    M["Device"] = D["device"]
    M["GPUSource"] = ("measured on the Colab " + D["device"]) if D["measured"] else \
        "projected with the T4 bandwidth model of Section~IV-D (the notebook replaces them with measured values)"
    for s, k in slopes.items():
        M["Slope" + s.replace("-", "")] = f"{k:.2f}"
    def iters(s, N):
        d = dd[(dd.solver == s) & (dd.N == N)]
        return int(d.iters.iloc[0]) if len(d) else None
    Nref = 512 if iters("JOR", 512) and iters("RB-SOR", 512) else 256
    M["NRef"] = str(Nref)
    M["ItJOR"] = f"{iters('JOR', Nref):,}"; M["ItGS"] = f"{iters('RB-GS', Nref):,}"
    M["ItSOR"] = f"{iters('RB-SOR', Nref):,}"
    M["SORvsJOR"] = f"{iters('JOR', Nref) / iters('RB-SOR', Nref):.0f}"
    M["GSvsJOR"] = f"{iters('JOR', Nref) / iters('RB-GS', Nref):.2f}"
    e = dd[(dd.N == Nref)].set_index("solver").max_err
    M["ErrJOR"] = f"{e.get('JOR', float('nan')):.2f}"; M["ErrSOR"] = f"{e.get('RB-SOR', float('nan')):.1e}"
    jl = dd[(dd.solver == "JOR") & dd.max_err.notna()].sort_values("N").iloc[-1]
    M["NJORLarge"] = str(int(jl.N)); M["ErrJORLarge"] = f"{jl.max_err:.1f}"
    cg = c.groupby(["solver", "N"]).kernel_ms.min()
    common = sorted(set(c[c.solver == "JOR"].N) & set(c[c.solver == "RB-SOR"].N))
    Nt = common[-1]
    M["NT"] = str(Nt)
    M["TJOR"] = f"{cg[('JOR', Nt)] / 1e3:.2f}"; M["TSOR"] = f"{cg[('RB-SOR', Nt)]:.1f}"
    M["TimeSpeedup"] = f"{cg[('JOR', Nt)] / cg[('RB-SOR', Nt)]:.0f}"
    def ratio(m):
        x = c[(c.method == m)].pivot_table(index="N", columns="variant", values="kernel_ms")
        r = (x["global"] / x["shared"]).dropna()
        return r.min(), r.max()
    jr, rr = ratio("jor"), ratio("rbsor")
    M["ShJORmin"], M["ShJORmax"] = f"{jr[0]:.2f}", f"{jr[1]:.2f}"
    M["ShRBmin"], M["ShRBmax"] = f"{rr[0]:.2f}", f"{rr[1]:.2f}"
    b = D["bench"]; Nm = b.N.max(); bd = b[(b.N == Nm)]
    M["NBench"] = str(Nm)
    M["BWJORd"] = f"{bd[(bd.method == 'jor') & (bd.precision == 'double')].GBps.max():.0f}"
    M["BWJORf"] = f"{bd[(bd.method == 'jor') & (bd.precision == 'float')].GBps.max():.0f}"
    M["GLUPSf"] = f"{bd[(bd.method == 'jor') & (bd.precision == 'float')].MLUPS.max() / 1e3:.1f}"
    M["GLUPSd"] = f"{bd[(bd.method == 'jor') & (bd.precision == 'double')].MLUPS.max() / 1e3:.1f}"
    bt = D["batch"].set_index("batch").kernel_ms
    M["BatchOne"] = f"{bt.get(1, float('nan')):.1f}"; M["BatchDef"] = f"{bt.get(200, float('nan')):.1f}"
    M["BatchGain"] = f"{bt.get(1, float('nan')) / bt.get(200, float('nan')):.2f}"
    o = D["omega"]; M["OmegaN"] = str(int(o.N.iloc[0]))
    M["OmegaBest"] = f"{o.loc[o.iters.idxmin()].omega:.4f}"
    M["OmegaBestIt"] = f"{int(o.iters.min()):,}"
    M["OmegaOneIt"] = f"{int(o[abs(o.omega - 1.0) < 1e-9].iters.iloc[0]):,}" if (abs(o.omega - 1.0) < 1e-9).any() else "--"
    with open(os.path.join(out, "numbers.tex"), "w") as f:
        f.write("% generated by make_assets.py -- do not edit\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    json.dump(M, open(os.path.join(out, "numbers.json"), "w"), indent=1)
    return M

def deck_data(D, out, M):
    it = D["iters"][D["iters"].precision == "double"]
    Ns = sorted(it.N.unique())
    ser = {s: [int(it[(it.solver == s) & (it.N == n)].iters.iloc[0]) if len(it[(it.solver == s) & (it.N == n)]) else None
               for n in Ns] for s in ["JOR", "RB-GS", "RB-SOR"]}
    c = D["conv"][D["conv"].precision == "double"]
    times = {}
    for s in ["JOR", "RB-GS", "RB-SOR"]:
        for v in ["global", "shared"]:
            d = c[(c.solver == s) & (c.variant == v)].sort_values("N")
            times[f"{s} {v}"] = {int(n): float(t) for n, t in zip(d.N, d.kernel_ms)}
    b = D["bench"]; Nm = int(b.N.max()); bd = b[b.N == Nm]
    bench = {f"{m} {v} {p}": float(bd[(bd.method == m) & (bd.variant == v) & (bd.precision == p)].MLUPS.mean() / 1e3)
             for m in ["jor", "rbsor"] for v in ["global", "shared"] for p in ["float", "double"]}
    o = D["omega"].sort_values("omega")
    bt = D["batch"].sort_values("batch")
    J = dict(measured=bool(D["measured"]), device=D["device"], N=[int(n) for n in Ns], iters=ser,
             times=times, bench_N=Nm, bench=bench,
             omega=dict(N=int(o.N.iloc[0]), w=[float(x) for x in o.omega], it=[int(x) for x in o.iters]),
             batch=dict(K=[int(x) for x in bt.batch], ms=[float(x) for x in bt.kernel_ms]),
             numbers=M)
    json.dump(J, open(os.path.join(out, "deck_data.json"), "w"), indent=1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="assets")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = load(a.data)
    if os.path.exists(os.path.join(a.data, "meta.json")):
        D.update(json.load(open(os.path.join(a.data, "meta.json"))))
    fig_field(D, a.out); fig_history(D, a.out); s = fig_iters(D, a.out); fig_omega(D, a.out)
    fig_time(D, a.out); fig_bw(D, a.out); fig_batch(D, a.out)
    M = tables(D, a.out, s)
    deck_data(D, a.out, M)
    print(("MEASURED" if D["measured"] else "MODEL") + " assets written to", a.out)
    for k in ["NRef", "ItJOR", "ItGS", "ItSOR", "SORvsJOR", "TimeSpeedup", "ShJORmin", "ShJORmax", "ShRBmin", "ShRBmax"]:
        print(f"  {k:12s} {M[k]}")

if __name__ == "__main__":
    main()