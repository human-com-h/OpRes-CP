"""Publication figures from accepted Phase 2D comparisons and Phase 3 replays."""

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.fft import dstn, rfft, rfft2, fftfreq
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase3_summarize import (DOWNLOADS, DROPOUT, ENSEMBLE, GRID_STUDIES, OP, OUT, PHASE2_OP, SEEDS,
                              horizon, load, phase2_metrics, read, result_root, stat)


REV = Path("results/phase3_revision")
FIGS = REV / "figures"
PAPER_FIGS = Path("paper/figures") if Path("paper").is_dir() else Path("outputs/figures")
REPLAY_FIGS = Path("results/manuscript_replay/figures") if Path("paper").is_dir() else Path("outputs/figures")
PDE_NAMES = {"darcy": "Darcy", "burgers": "Burgers", "allen_cahn": "Allen–Cahn"}
OP_COLOR, EN_COLOR, MC_COLOR = "#006B62", "#C45A2A", "#6256A5"
PALETTE = (OP_COLOR, EN_COLOR, MC_COLOR, "#486991", "#A94A76", "#7E6221", "#2788A3")
METHODS = (PHASE2_OP, ENSEMBLE, DROPOUT)
METHOD_NAMES = ("OpRes-CP", "Ensemble mean", "MC-Dropout")
MAIN = {"darcy": "id_test_64", "burgers": "id_64", "allen_cahn": "id_64_20"}
CALIBRATION_SPLITS = {"darcy": ("grid64", "id_test_64"), "burgers": (None, "id"), "allen_cahn": (None, "short_id")}


matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": "#22313E", "text.color": "#182835", "axes.edgecolor": "#88949D",
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "axes.grid": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})


def save(fig, basename, pdf_dpi=None):
    for d in (FIGS, PAPER_FIGS, REPLAY_FIGS):
        d.mkdir(parents=True, exist_ok=True)
    pdf_kwargs = {"bbox_inches": "tight", "pad_inches": .10}
    if pdf_dpi is not None:
        pdf_kwargs["dpi"] = pdf_dpi
    pdf_path = FIGS / f"{basename}.pdf"
    png_path = FIGS / f"{basename}.png"
    fig.savefig(pdf_path, **pdf_kwargs)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=.10, dpi=300)
    if PAPER_FIGS.resolve() != pdf_path.parent.resolve():
        shutil.copy2(pdf_path, PAPER_FIGS / f"{basename}.pdf")
    if REPLAY_FIGS.resolve() != PAPER_FIGS.resolve():
        shutil.copy2(pdf_path, REPLAY_FIGS / f"{basename}.pdf")
        shutil.copy2(png_path, REPLAY_FIGS / f"{basename}.png")
    else:
        shutil.copy2(png_path, PAPER_FIGS / f"{basename}.png")
    plt.close(fig)


def span(values):
    _, avg, _, lo, hi = stat(values)
    return avg, avg-lo, hi-avg


def format_percent(ax, axis="y"):
    getattr(ax, f"{axis}axis").set_major_formatter(PercentFormatter(1, decimals=0))


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#DDE3E7", alpha=.9, lw=.65)
    ax.set_axisbelow(True)


def title(fig, main, sub):
    fig.suptitle(main, fontsize=16, fontweight="bold", x=.02, ha="left")
    fig.supxlabel(sub, x=.02, ha="left", fontsize=8.4, color="#425462")


def example(pde, split):
    with np.load(result_root(pde, SEEDS[0])/f"example_{split}.npz") as f:
        return {k: f[k] for k in f.files}


def revision_example(name):
    with np.load(REV / "examples" / f"example_{name}.npz") as f:
        return {k: f[k] for k in f.files}


def spectrum(data, pde):
    if pde == "burgers":
        power = (abs(rfft(data, axis=-1, norm="ortho"))**2).sum(0)
        power[1:-1] *= 2
    else:
        n = data.shape[-1]
        if pde == "darcy":
            values = abs(dstn(data, type=2, norm="ortho"))**2
            rad = np.hypot(np.arange(1,n+1)[:,None], np.arange(1,n+1)).astype(int)
        else:
            values = (abs(rfft2(data, norm="ortho"))**2).sum(0)
            values[:,1:-1] *= 2
            rad = np.hypot((fftfreq(n)*n)[:,None], np.arange(n//2+1)).astype(int)
        power = np.bincount(rad.ravel(), weights=values.ravel())
    return power / max(power.sum(), 1e-30)


def fig1(accepted, id_examples):
    fig = plt.figure(figsize=(12.1, 8.0), layout="constrained")
    spec = fig.add_gridspec(3, 3, height_ratios=(.65, 1.45, 1.7))
    title(fig, "Physics as scale with a frozen neural operator",
          "Spectra use fixed ID main-track FNO sample 0 (seed 42; Darcy 64², Burgers 64, Allen–Cahn 64² T=1). Width uses 500 ID cases/seed; warm latency uses 20 cases/seed (48 for Darcy OpRes/ensemble). Bars: 95% seed t-CIs.")
    pipe = fig.add_subplot(spec[0,:])
    pipe.axis("off")
    pipe.set(xlim=(0, 1), ylim=(0, 1))
    labels = ("Frozen operator\npoint prediction", "Signed PDE\ndefect", "Boundary-matched\noperator inverse", "Physical envelope\n+ scalar floor", "Split conformal\nwhole-field band")
    for i, term in enumerate(labels):
        x = .018 + .198*i
        pipe.add_patch(FancyBboxPatch((x,.21),.160,.57,boxstyle="round,pad=.009",fc="#EAF3F0",ec="#6D9D91",lw=.9))
        pipe.text(x+.080,.498,term,va="center",ha="center",fontsize=10.3)
        if i < len(labels)-1:
            pipe.add_patch(FancyArrowPatch((x+.170,.495),(x+.188,.495),arrowstyle="-|>",
                                          mutation_scale=9,color="#527F73",shrinkA=0,shrinkB=0,zorder=3))
    spec_titles = {
        "darcy": "Darcy (ID 64², FNO)",
        "burgers": "Burgers (ID 64, FNO)",
        "allen_cahn": "Allen–Cahn (ID 64² T=1, FNO)",
    }
    for i, (pde, z) in enumerate(id_examples.items()):
        ax = fig.add_subplot(spec[1,i])
        curves = (z["residual"], z["prediction"]-z["truth"], z["lifted"])
        for arr, label, color in zip(curves, ("Defect", "True error", "Signed inverse"), (EN_COLOR, "#31465C", OP_COLOR)):
            y = spectrum(arr.astype(float), pde)
            k = np.arange(len(y))
            ax.semilogy(k[1:], np.maximum(y[1:],1e-17), color=color, lw=1.35, label=label)
        ax.set(title=spec_titles[pde], xlabel="Spatial wavenumber", ylim=(1e-10,1.5))
        grid(ax)
        if i==0:
            ax.set_ylabel("Fraction of spectral energy")
            ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(spec[2,:])
    ax.set(xscale="log", yscale="log", xlabel="Warm T4 P50 latency (ms, log scale)",
           ylabel="Matched 90% width / ensemble", title="Whole-field sharpness and latency")
    grid(ax)
    for pde, marker in zip(PDE_NAMES, ("o","s","D")):
        for i, (method,color) in enumerate(zip(METHODS,PALETTE)):
            widths, times = [], []
            for seed in SEEDS:
                node = accepted[pde][seed]
                key = MAIN[pde]
                group = node["results"][key]
                widths.append(group[method]["0.1"]["matched_width"]/group[ENSEMBLE]["0.1"]["matched_width"])
                if pde=="darcy" and i<2:
                    item = node["phase2a_ensemble"]["ensemble_timing_export"]["64"]
                    times.append(item[("single_plus_opres","ensemble_total")[i]]["p50_ms"])
                else:
                    times.append(node["latency"][key][("OpRes end-to-end","ensemble","mc_dropout")[i]]["p50_ms"])
            x, xl, xu = span(times); y, yl, yu = span(widths)
            ax.errorbar(x,y,xerr=[[xl],[xu]],yerr=[[yl],[yu]],fmt=marker,lw=1.05,ms=5.7,capsize=2.5,color=color,
                        label=METHOD_NAMES[i] if pde=="darcy" else None)
            ax.annotate(PDE_NAMES[pde],(x,y),xytext=(6,-11 if i==1 else 5),textcoords="offset points",fontsize=8)
    ax.axhline(1, color="#8E9A9F", linestyle=":", linewidth=.9)
    ax.legend(loc="lower right", ncol=3, fontsize=8.5)
    save(fig,"fig1_physics_scale")


def fig2(data, examples):
    fig, axs = plt.subplots(5,3,figsize=(10.8,13.1),layout="constrained",
                            gridspec_kw={"height_ratios":[1, .94, 1, 1, 1]})
    title(fig, "Reference fields, local errors and calibrated widths",
          "Fixed sample index 0, seed 42. Darcy ID (64² FNO), Burgers high-frequency OOD (64 FNO), and Allen–Cahn paired U-Net (128², T=2). Per-example maps do not have confidence intervals.")
    d, b, a = (examples[p] for p in PDE_NAMES)
    qs = (data["darcy"][SEEDS[0]]["grid64"]["thresholds"][OP]["0.1"], data["burgers"][SEEDS[0]]["thresholds"][OP]["0.1"],
          horizon("allen_cahn",SEEDS[0])["results"]["U-Net"]["paired_128"]["jacobian_on"]["T2_same_window"]["0.1"]["threshold"])

    def panel(ax, image, title_, cmap, time_max=None, lim=None):
        half_frame = time_max / (2 * image.shape[0]) if time_max else 0
        mesh = ax.pcolormesh(np.linspace(0,1,image.shape[1]+1),
                             np.linspace(half_frame,(time_max or 1)+half_frame,image.shape[0]+1), image,
                             cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1],
                             shading="flat", rasterized=True, antialiased=False)
        ax.set_title(title_,fontsize=9.7)
        ax.set_xticks((0,.5,1))
        if not time_max:
            ax.set_yticks((0,.5,1))
            ax.set_aspect("equal")
        return mesh

    def colorbar(mesh, axes, shrink):
        bar = fig.colorbar(mesh, ax=axes, shrink=shrink, pad=.02)
        bar.solids.set_rasterized(True)

    for j,(z, lab, cm) in enumerate(((np.log(d["input"]),"A  Darcy log permeability","PuOr_r"),
            (abs(d["prediction"]-d["truth"]),"Absolute error","magma"),(2*qs[0]*d["scale"],"Calibrated full width","viridis"))):
        m = panel(axs[0,j],z,lab,cm)
        colorbar(m, axs[0,j], .75)
    for j,(z, lab, cm) in enumerate(((b["truth"],"B  Burgers reference","RdBu_r"),
            (abs(b["prediction"]-b["truth"]),"Absolute error","magma"),(2*qs[1]*b["scale"],"Calibrated full width","viridis"))):
        m = panel(axs[1,j],z,lab,cm,time_max=1.6)
        colorbar(m, axs[1,j], .75)
    axs[1,0].set_ylabel("Time")
    for ax in axs[1]:
        ax.set_xlabel("Periodic coordinate / 2π")
    times = (10,20,40)
    selected = [k-1 for k in times]
    maps = (a["truth"],abs(a["prediction"]-a["truth"]),2*qs[2]*a["scale"])
    for row,z,cm,desc in zip(range(2,5), maps, ("RdBu_r","magma","viridis"), ("C  Allen–Cahn reference","Absolute error","Calibrated full width")):
        limits = (-abs(z[selected]).max(),abs(z[selected]).max()) if row==2 else (0, z[selected].max())
        for j,k in enumerate(times):
            name = f"{desc}  t={k*.05:g}" if j==0 else f"t={k*.05:g}"
            m = panel(axs[row,j],z[k-1],name,cm,lim=limits)
        colorbar(m, axs[row,:], .7)
    save(fig,"fig2_field_maps", pdf_dpi=300)


def fig3(long):
    fig = plt.figure(figsize=(12.1,9.6),layout="constrained")
    gs = fig.add_gridspec(2,2,height_ratios=(1.18,1))
    title(fig, "Grid transfer and independent rollout calibration",
          "Five seeds × 500 paired cases per protocol. Bars show 95% field bootstrap on the grid panel and 95% seed-level t intervals on the rollout panels.")
    with (Path("results")/"phase2d_paired.csv").open(encoding="utf-8-sig") as f:
        paired = {(r["batch"],int(r["base_grid"]),r["left_group"],r["right_group"],r["left_method"]):r
                  for r in csv.DictReader(f) if r["comparison"]=="grid" and r["alpha"]=="0.1" and r["metric"]=="coverage_difference"}
    ax = fig.add_subplot(gs[0,:])
    from phase3_summarize import FORWARD
    for j,(method,c) in enumerate(zip(METHODS,PALETTE)):
        for i,(pde,n,core,target,_) in enumerate(GRID_STUDIES):
            r = paired.get((FORWARD[pde],n,target,core,method))
            if r is not None:
                v, lo, hi = (100*float(r[k]) for k in ("mean","field_bootstrap_low","field_bootstrap_high"))
                ax.bar(i+(j-1)*.24,v,.22,color=c,label=METHOD_NAMES[j] if i==0 else None)
                ax.errorbar(i+(j-1)*.24,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt="none",ecolor="#24323D",lw=.85,capsize=2.5)
    ax.set(xticks=np.arange(7),xticklabels=[r[-1] for r in GRID_STUDIES],ylabel="Coverage change (pp)",title="A  Seven paired grid trajectories")
    ax.tick_params(axis="x",rotation=13)
    ax.axhline(0,color="#576A77",lw=.8)
    ax.legend(ncols=3,loc="upper right",fontsize=8.6); grid(ax)
    groups, labs = ("id_64","paired_128","ood_64"), ("ID 64²","Paired 128²","Reaction OOD")
    policies = ("T1","T1_Bonferroni","T2_same_window")
    slots = (-0.24, -0.08, 0.08, 0.24)
    for subplot, model, label in ((gs[1,0],"FNO","B"),(gs[1,1],"U-Net","C")):
        ax = fig.add_subplot(subplot)
        for j,(policy,c,long_label) in enumerate(zip(policies,("#9A6254", "#476B98",OP_COLOR),("T1 90%","T1 95%","T2 90%"))):
            for i,g in enumerate(groups):
                vals = [long[s]["results"][model][g]["jacobian_on"][policy]["0.1"]["coverage"] for s in SEEDS]
                mean, lo, hi = span(vals)
                ax.errorbar(i+slots[j],mean,yerr=[[min(mean,lo)],[min(1-mean,hi)]],fmt="o",ms=4.5,lw=1,capsize=2.5,color=c,label=long_label if i==0 else None)
        for i,g in enumerate(groups):
            off = [long[s]["results"][model][g]["jacobian_off"]["T2_same_window"]["0.1"]["coverage"] for s in SEEDS]
            mean,lo,hi = span(off)
            ax.errorbar(i+slots[3],mean,yerr=[[min(mean,lo)],[min(1-mean,hi)]],fmt="D",ms=4.2,lw=1,capsize=2.5,color="#3F4B55",
                        label="T2 Jacobian off" if i==0 else None)
        on_value = np.mean([long[s]["results"][model]["id_64"]["jacobian_on"]["T2_same_window"]["0.1"]["matched_width"]
                            /long[s]["results"][model]["id_64"]["jacobian_off"]["T2_same_window"]["0.1"]["matched_width"] for s in SEEDS])
        ax.text(.02,.035,f"ID matched width, Jacobian on/off = {on_value:.2f}×",transform=ax.transAxes,fontsize=8.2)
        ax.set(xticks=np.arange(3),xticklabels=labs,xlim=(-.42,2.42),ylim=(.26,1.045),ylabel="T=2 simultaneous coverage",title=f"{label}  Allen–Cahn {model}")
        ax.axhline(.9,color="#82919A",linestyle="--",lw=.9); format_percent(ax); grid(ax)
        ax.legend(ncols=2,loc="lower left",fontsize=7.6,bbox_to_anchor=(.01,.11))
    save(fig,"fig3_grid_and_rollout")


def darcy_gate(seed):
    root = result_root("darcy",seed)
    with np.load(root/"fields_grid64_cal.npz") as cal, np.load(root/"fields_grid64_id_test_64.npz") as i, np.load(root/"fields_grid64_ood_b_64.npz") as o:
        xc, xt = cal["_input"], np.concatenate((i["_input"],o["_input"]))
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1,max_iter=1000))
    labels = np.arange(len(xt))>=500
    estimator.fit(np.concatenate((xc,xt)),np.arange(len(xc)+len(xt))>=len(xc))
    raw = np.exp(estimator.decision_function(xt)) * len(xc)/len(xt)
    fpr, tpr, _ = roc_curve(labels, np.clip(raw,.02,50), drop_intermediate=False)
    node = read(DOWNLOADS/"2a"/f"seed_{seed}"/"results/phase2_darcy"/f"seed_{seed}"/"metrics.json")
    metrics = node["tracks"]["64"]["FNO"]["gates"]["ood_b"]["input"]
    return fpr, tpr, metrics["curve"]


def stack_curve(xvals, ys, evaluation):
    curves = []
    for xs, v in zip(xvals,ys):
        ind = np.clip(np.searchsorted(xs,evaluation,side="right")-1,0,len(xs)-1)
        curves.append(np.asarray(v)[ind])
    curve = np.stack(curves)
    avg = curve.mean(0)
    ci = t.ppf(.975,df=len(curve)-1)*curve.std(0,ddof=1)/np.sqrt(len(curve))
    return avg, np.maximum(avg-ci,0), np.minimum(avg+ci,1)


def fig4(data, accepted):
    fig, axes = plt.subplots(2,2,figsize=(12.2,8.2),layout="constrained")
    title(fig,"Distribution shift, finite intervals and selective prediction",
          "Five seeds × 500 cases per group. Bars/bands: 95% seed t-CIs. Panel A: OpRes points (ratio=1) are slightly offset horizontally around x=1 for clarity. Risk curves hold at last attainable gate.")
    ax = axes[0,0]
    groups = (("darcy","ood_a_64","Darcy contrast"),("darcy","ood_b_64","Darcy structure"),
              ("burgers","ood_64","Burgers viscosity"),("burgers","ood_high_64","Burgers spectrum"),
              ("allen_cahn","ood_64_20","Allen–Cahn reaction"))
    op_x_offsets = (0.91, 0.80, 1.04, 1.00, 1.16)
    ax.axvline(1.0, color="#8C989F", linestyle=":", linewidth=.85, zorder=0)
    for (pde,g,name),mark,x_off in zip(groups,("o","s","^","D","P"),op_x_offsets):
        frontier = []
        for i,(method,c) in enumerate(zip(METHODS,PALETTE)):
            records = [accepted[pde][s]["results"][g] for s in SEEDS]
            x, xl, xu = span([r[method]["0.1"]["width"]/r[PHASE2_OP]["0.1"]["width"] for r in records])
            y, yl, yu = span([r[method]["0.1"]["coverage"] for r in records])
            x_disp = x_off if method == PHASE2_OP else x
            xerr = [[0.0],[0.0]] if method == PHASE2_OP else [[min(x*.95,xl)],[xu]]
            ax.errorbar(x_disp,y,xerr=xerr,yerr=[[min(y,yl)],[min(1-y,yu)]],color=c,fmt=mark,ms=4.5,lw=.9,capsize=2.2)
            frontier.append((x_disp,y))
        pareto = []
        for x,y in sorted(frontier,key=lambda p:(p[0],-p[1])):
            if not pareto or y > pareto[-1][1]:
                pareto.append((x,y))
        ax.plot(*zip(*pareto),color="#536771",ls=":",lw=.9,zorder=0)
        ax.plot([],[],color="#4A5965",marker=mark,ls="none",label=name,ms=4.5)
    for (m,c) in zip(METHOD_NAMES,PALETTE):
        ax.plot([],[],color=c,marker=".",ls="none",label=m,ms=6.5)
    ax.set(xscale="log",xlabel="Observed finite full width / locked OpRes width",ylabel="Observed simultaneous coverage",
           ylim=(.24, 1.045), title="A  Unweighted coverage and sharpness")
    ax.axhline(.9,color="#8C989F",linestyle="--",lw=.8); ax.legend(fontsize=7.3,ncol=2,loc="lower right"); format_percent(ax); grid(ax)
    ax=axes[0,1]
    for i,(pde,g,name) in enumerate(groups):
        for j,(method,c) in enumerate(zip(("Full-grid WCP","Physical-summary WCP","Physical-summary + spectrum WCP"),PALETTE)):
            records = [accepted[pde][s]["results"][g] for s in SEEDS]
            if method not in records[0]:
                continue
            v, lo, hi = span([r[method]["0.1"]["infinite_rate"] for r in records])
            if not any(line.get_label()==method for line in ax.containers):
                label=method
            else:
                label=None
            ax.bar(i+(j-1)*.23,v,.22,color=c,label=label)
            ax.errorbar(i+(j-1)*.23,v,yerr=[[min(v,lo)],[min(1-v,hi)]],fmt="none",ecolor="#24323D",lw=.8,capsize=2)
    ax.set(xticks=np.arange(len(groups)), xticklabels=[g[2] for g in groups],ylim=(0,1.26),
           yticks=(0,.2,.4,.6,.8,1.0),ylabel="Infinite interval rate",title="B  WCP voids at the 90% target")
    ax.tick_params(axis="x",rotation=18,labelsize=8); ax.legend(fontsize=7.4,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.01)); format_percent(ax); grid(ax)
    series = {"Darcy structure": [darcy_gate(s) for s in SEEDS],
              "Burgers spectrum": [(data["burgers"][s]["gates"]["spectral energy"]["fpr"],
                                     data["burgers"][s]["gates"]["spectral energy"]["tpr"],
                                     data["burgers"][s]["gates"]["spectral energy"]["risk_curve"]) for s in SEEDS]}
    axr, axc = axes[1]
    for (name,runs),c,ls in zip(series.items(),PALETTE,("-","--")):
        x = np.linspace(0,1,251)
        avg,lo,hi = stack_curve([r[0] for r in runs],[r[1] for r in runs],x)
        axr.plot(np.r_[0,x],np.r_[0,avg],color=c,ls=ls,label=name,lw=1.7); axr.fill_between(x,lo,hi,color=c,alpha=.15)
        curve = [r[2] for r in runs]
        x = np.linspace(.025,1,251)
        avg,lo,hi = stack_curve([[r["acceptance"] for r in k] for k in curve],[[r["point_risk"] for r in k] for k in curve],x)
        axc.plot(x,avg,color=c,ls=ls,label=name,lw=1.7,drawstyle="steps-post")
        axc.fill_between(x,lo,hi,color=c,alpha=.15,step="post")
    axr.plot([0,1],[0,1],color="#B2B9BF",ls=":",lw=1)
    axr.set(xlim=(-.01,1.01),ylim=(0,1.03),xlabel="ID false-positive rate",ylabel="OOD true-positive rate",title="C  Equal-mix input-only detection")
    axc.set(xlim=(.024,1.002),ylim=(0,.42),xlabel="Accepted fraction",ylabel="Point failure rate (relative L2 > 0.15)",title="D  Risk–coverage at observed gate thresholds")
    for ax in (axr,axc):
        ax.legend(loc="lower right" if ax==axr else "upper left",fontsize=8); format_percent(ax,"x"); format_percent(ax); grid(ax)
    save(fig,"fig4_shift_and_abstention")


def fig5(data):
    fig, axes = plt.subplots(2,2,figsize=(12.1,8.1),layout="constrained")
    title(fig,"Mechanism and sensitivity with frozen predictors",
          "Five seeds × 500 cases per test group. Bars show 95% seed-level t intervals. Each new band uses the corresponding locked Cal prefix or method-specific scores.")
    ax=axes[0,0]
    radii, params = (.04,.08,.16,.32), (0,.25,.50,.70,.85,1)
    tradeoffs = []
    for radius,c in zip(radii,PALETTE):
        plot=[]
        for lam in params:
            method=f"lambda={lam:g}, radius={radius:g}"
            x=span([d["metrics"]["id"][method]["0.1"]["matched_width"] for d in data["burgers"].values()])
            y=span([d["metrics"]["ood_high"][method]["0.1"]["coverage"] for d in data["burgers"].values()])
            plot.append((x,y))
            tradeoffs.append((x[0],y[0]))
        ax.plot([x[0] for x,y in plot],[y[0] for x,y in plot],color=c,marker="o",ms=3.5,lw=1.35,label=fr"$\ell={radius:g}$")
        if radius==.08:
            lam_pos = {
                0:    ((-6, -13), "right"),
                .25:  ((24, -6),  "left"),
                .50:  ((-14, 14), "right"),
                .70:  ((-14, 2),  "right"),
                .85:  ((-12, -13),"right"),
                1:    ((11, -13), "left"),
            }
            for lam,(x,y) in zip(params,plot):
                ax.errorbar(x[0],y[0],xerr=[[x[1]],[x[2]]],yerr=[[y[1]],[y[2]]],fmt="none",color=c,lw=.75,capsize=2)
                xytext, ha = lam_pos[lam]
                ax.annotate(rf"$\lambda={lam:g}$",(x[0],y[0]),xytext=xytext,textcoords="offset points",
                            ha=ha,va="center",fontsize=7.6,
                            arrowprops=dict(arrowstyle="-",color="#7B8B94",lw=.55,shrinkA=1,shrinkB=2))
    front = []
    for x,y in sorted(tradeoffs,key=lambda pair:(pair[0],-pair[1])):
        if not front or y > front[-1][1] + 1e-12:
            front.append((x,y))
    ax.plot(*zip(*front),color="#35434C",lw=.95,ls="--",label="Empirical Pareto front")
    ax.set(xlabel="ID matched 90% full width",ylabel="High-frequency OOD coverage",
           xlim=(0.088, 0.445), title="A  Burgers localization trade-off")
    ax.legend(ncol=2,loc="lower right",fontsize=7.8); format_percent(ax); grid(ax)
    ax=axes[0,1]
    for method,c,label in zip(("fixed physical radius","fixed pixel radius"),PALETTE,("Physical radius 0.04","Fixed 1.28 cells")):
        vals=[span([d["grid32"]["metrics"][f"paired_test_{n}"][method]["0.1"]["coverage"] for d in data["darcy"].values()]) for n in (32,64,128)]
        ax.errorbar(np.arange(3),[v[0] for v in vals],yerr=np.array(vals)[:,1:].T,fmt="o-",color=c,capsize=3,lw=1.35,label=label)
    vals=[span([phase2_metrics("darcy",s,32)["results"][f"paired_test_{n}"]["5-FNO Ensemble + CP | single center"]["0.1"]["coverage"] for s in SEEDS]) for n in (32,64,128)]
    ax.errorbar(np.arange(3),[v[0] for v in vals],yerr=np.array(vals)[:,1:].T,fmt="s--",color=MC_COLOR,capsize=3,lw=1.25,label="Ensemble, same center")
    ax.set(xticks=np.arange(3), xticklabels=("32²","64²","128²"),ylabel="Simultaneous coverage",title="B  Darcy coarse-grid radius control")
    ax.axhline(.9,color="#8D999F",ls="--",lw=.8); ax.legend(fontsize=7.8,loc="lower left"); format_percent(ax); grid(ax)
    ax=axes[1,0]
    for pde,c in zip(("burgers","allen_cahn"),PALETTE):
        _, group = CALIBRATION_SPLITS[pde]
        for suffix,ls in (("","-"),(" reference evaluation grid","--")):
            methods=["stride=1"]+[f"stride={s}{suffix}" for s in (2,4)]
            vals=[span([d["metrics"][group][m]["0.1"]["matched_width"]/d["metrics"][group][OP]["0.1"]["matched_width"] for d in data[pde].values()]) for m in methods]
            ax.errorbar(np.arange(3),[v[0] for v in vals],yerr=np.array(vals)[:,1:].T,fmt="o",ls=ls,color=c,capsize=2.5,lw=1.2,
                        label=f"{PDE_NAMES[pde]} {'thin lift and score' if not suffix else 'score only'}")
    ax.axhline(1,color="#849299",ls=":",lw=.9)
    ax.set(xticks=np.arange(3),xticklabels=("Δt","2Δt","4Δt"),ylabel="Matched width / dense cadence",title="C  Time sampling and evaluation grid")
    ax.legend(ncol=1,fontsize=7.6,loc="best"); grid(ax)
    ax=axes[1,1]
    for (pde,(track,g)),c in zip(CALIBRATION_SPLITS.items(),PALETTE):
        vals=[]
        for m in (100,200,400,500):
            q=[(d[track] if track else d)["calibration_size"][str(m)][g]["0.1"]["coverage"] for d in data[pde].values()]
            vals.append(span(q))
        ax.errorbar((100,200,400,500),[v[0] for v in vals],yerr=np.array(vals)[:,1:].T,fmt="o-",color=c,capsize=2.5,lw=1.3,label=PDE_NAMES[pde])
    ax.axhline(.90,color="#849299",ls="--",lw=.8)
    ax.set(xticks=(100,200,400,500),xlabel="Cal fields or trajectories",ylabel="Observed ID coverage",title="D  Nested calibration sizes")
    ax.legend(ncol=3,fontsize=8,loc="lower right"); format_percent(ax); grid(ax)
    save(fig,"fig5_frozen_ablations")


def main():
    data,_ = load()
    accepted={pde: {s:phase2_metrics(pde,s) for s in SEEDS} for pde in PDE_NAMES}
    long={s:horizon("allen_cahn",s) for s in SEEDS}
    id_examples={
        "darcy": example("darcy", "id_test"),
        "burgers": revision_example("burgers_id"),
        "allen_cahn": revision_example("allen_cahn_id"),
    }
    map_examples={pde:example(pde,split) for pde,split in (("darcy","id_test"),("burgers","ood_high"),("allen_cahn","paired_128"))}
    fig1(accepted,id_examples)
    fig2(data,map_examples)
    fig3(long)
    fig4(data,accepted)
    fig5(data)
    print(f"Five PDF and PNG pairs in {FIGS} (synced to {PAPER_FIGS} and {REPLAY_FIGS})")


if __name__ == "__main__":
    main()
