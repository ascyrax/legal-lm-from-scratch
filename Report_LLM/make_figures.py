import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
R = json.load(open("data/curves.json"))

C = {"dec01": "#2a78d6", "dec02": "#eb6834", "dec03": "#1baf7a", "encdec": "#c98500", "kd": "#d55181"}
NAME = {"dec01": "Dec-01 (NoPE, MHA)", "dec02": "Dec-02 (local/global + GQA)", "dec03": "Dec-03 (ALiBi + MLA)",
        "encdec": "Enc-Dec v2 (bidir. + MLM)", "kd": "KD student (7-block Dec-03)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=150); plt.close(fig)

# ---------- Figure 1: Phase-1 learning curves ----------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
for k in ["dec01", "dec02", "dec03"]:
    st = np.array(R[k]["steps"]); ev = np.array(R[k]["eval"])
    toks = st[:, 2] * st[:, 3] * 60          # cumulative tokens = mean tok/s * elapsed seconds
    tk = np.interp(ev[:, 0], st[:, 0], toks) / 1e6
    hrs = np.interp(ev[:, 0], st[:, 0], st[:, 3]) / 60
    ax[0].plot(tk, ev[:, 1], color=C[k], marker="o", ms=2.2, label=f"{NAME[k]}: best {ev[:,1].min():.3f}")
    ax[1].plot(hrs, ev[:, 1], color=C[k], marker="o", ms=2.2)
    off={"dec01":(-6,10),"dec02":(-4,-12),"dec03":(-22,8)}[k]
e = np.array(R["encdec"]["eval"]); est = np.array(R["encdec"]["steps"])
ehrs = np.interp(e[:, 0], est[:, 0], est[:, 2])
ax[1].plot(ehrs, e[:, 1], color=C["encdec"], marker="s", ms=2.2, label=NAME["encdec"] + f": best {e[:,1].min():.3f} (seq2seq)")
ax[0].set_xlim(0,650); ax[0].set_xlabel("training tokens seen (millions)"); ax[0].set_ylabel("validation loss (nats/token)")
ax[1].set_xlabel("wall-clock hours on one T4")
ax[0].set_title("(a) decoders, loss vs. tokens", loc="left"); ax[1].set_title("(b) all four models, loss vs. time", loc="left")
for a in ax: a.set_ylim(1.8, 3.4)
h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
save(fig, "fig_p1_curves")

# ---------- Figure 2: data + per-label ----------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
cl = {'Fact': 2979, 'RatioOfTheDecision': 2856, 'PrecedentReliedUpon': 1305, 'ArgumentRespondent': 996, 'ArgumentPetitioner': 976,
      'RulingByLowerCourt': 438, 'Statute': 367, 'Dissent': 284, 'PrecedentNotReliedUpon': 209, 'RulingByPresentCourt': 113, 'None': 53, 'Issue': 17}
it = {'RatioOfTheDecision': 1445, 'Fact': 1119, 'Statute': 732, 'PrecedentReliedUpon': 640, 'ArgumentPetitioner': 637, 'RulingByLowerCourt': 436,
      'PrecedentNotReliedUpon': 360, 'ArgumentRespondent': 352, 'None': 244, 'RulingByPresentCourt': 194, 'Issue': 90, 'PrecedentOverruled': 3}
labs = sorted(set(cl) | set(it), key=lambda l: -(cl.get(l, 0) / 10593 + it.get(l, 0) / 6252))
y = np.arange(len(labs)); w = 0.38
ax[0].barh(y - w / 2, [100 * cl.get(l, 0) / 10593 for l in labs], w, color=C["dec01"], label="CL (10,593 train sentences)")
ax[0].barh(y + w / 2, [100 * it.get(l, 0) / 6252 for l in labs], w, color=C["dec02"], label="IT (6,252 train sentences)")
short = {"RatioOfTheDecision": "Ratio", "PrecedentReliedUpon": "Prec. relied", "ArgumentRespondent": "Arg. respondent",
         "ArgumentPetitioner": "Arg. petitioner", "RulingByLowerCourt": "Ruling (lower)", "PrecedentNotReliedUpon": "Prec. not relied",
         "RulingByPresentCourt": "Ruling (present)", "PrecedentOverruled": "Prec. overruled"}
ax[0].set_yticks(y, [short.get(l, l) for l in labs]); ax[0].invert_yaxis(); ax[0].grid(axis="y", visible=False)
ax[0].set_xlabel("share of training sentences (%)"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,-0.2), ncol=2)
ax[0].set_title("(a) Phase 3: rhetorical-role distribution by domain", loc="left")
# per-label test F1 (best Phase-2 submission) with train-fold counts
f1 = {'DATE': .890, 'COURT': .783, 'STAT': .689, 'CASENO': .466, 'JUDGE': .422, 'R.COUNSEL': .291, 'AUTH': .282, 'PREC': .276,
      'A.COUNSEL': .167, 'WIT': .109, 'RESP': .062, 'APP': .039}
trn = {'DATE': 813, 'PREC': 624, 'AUTH': 535, 'STAT': 525, 'COURT': 455, 'CASENO': 423, 'APP': 207, 'RESP': 191, 'JUDGE': 123,
       'R.COUNSEL': 87, 'A.COUNSEL': 70, 'WIT': 67}
role = {"APP", "RESP", "WIT", "A.COUNSEL", "R.COUNSEL", "JUDGE"}
ls = list(f1); y2 = np.arange(len(ls))
ax[1].barh(y2, [f1[l] for l in ls], 0.62, color=[C["dec02"] if l in role else C["dec01"] for l in ls])
for i, l in enumerate(ls):
    ax[1].text(f1[l] + 0.015, i, f"{f1[l]:.2f}  (n={trn[l]})", va="center", fontsize=6.8, color=INK)
ax[1].set_yticks(y2, ls); ax[1].invert_yaxis(); ax[1].set_xlim(0, 1.18); ax[1].grid(axis="y", visible=False)
ax[1].set_xlabel("strict F1 on the hidden test set")
ax[1].set_title("(b) Phase 2: per-label test F1 (n = train count)", loc="left")
from matplotlib.patches import Patch
ax[1].legend([Patch(color=C["dec01"]), Patch(color=C["dec02"])], ["form-defined label", "role-defined label"], loc="upper center", bbox_to_anchor=(0.5,-0.2), ncol=2)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_data_ner")

# ---------- Figure 3: KD retention + capacity vs retrieval ----------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.45))
kd = [("P2\nNER", .3150, .3215, "Dec-01+LoRA"), ("P4\nCJPE", .6075, .5419, "Dec-02+LoRMA"),
      ("P5\nLSI", .4486, .4201, "Dec-02+LoRMA"), ("P6\nSumm.", .8544, .8555, "Enc-Dec+adapters"),
      ("P7\nPCR", .4480, .4503, "Dec-02+LoRMA")]
x = np.arange(len(kd)); r = [s / t for _, s, t, _ in kd]
ax[0].bar(x, r, 0.55, color=C["kd"]); ax[0].axhline(1.0, color=MUTED, lw=0.9, ls="--")
for i, (lab, s, t, tn) in enumerate(kd):
    ax[0].text(i, r[i] + 0.006, f"{r[i]:.2f}", ha="center", fontsize=7, color=INK)
ax[0].set_xticks(x, [k[0] for k in kd], fontsize=7); ax[0].set_ylim(0.9, 1.15); ax[0].grid(axis="x", visible=False)
ax[0].set_ylabel("student / teacher (dev)")
ax[0].set_title("(a) KD: student / teacher score", loc="left")
pts = [("IA$^3$ (Enc-Dec)", 0.28, 300, "encdec"), ("QLoRA (Dec-01)", 2.15, 302, "dec01"), ("LoRMA (Dec-02)", 2.23, 299, "dec02"),
       ("partial FT (Dec-03)", 11.8, 313, "dec03"), ("KD student", 36.8, 299, "kd"), ("full FT (Dec-01)", 55.2, 303, "dec01")]
for lab, p, h, k in pts:
    ax[1].scatter(p, h, s=28, color=C[k], zorder=3, edgecolor="white", linewidth=0.8)
    ax[1].annotate(lab, (p, h), xytext={"LoRMA (Dec-02)":(4,-9),"KD student":(-8,-12),"full FT (Dec-01)":(-20,6),"QLoRA (Dec-01)":(-10,7),"IA$^3$ (Enc-Dec)":(-6,-12)}.get(lab,(4,4)), textcoords="offset points", fontsize=6.8, color=INK)
ax[1].axhspan(201, 237, color=GRID, alpha=0.7, lw=0); ax[1].text(0.2, 219, "zero-shot range (no training)", fontsize=6.8, color=MUTED, va="center")
ax[1].set_xscale("log"); ax[1].set_xlim(0.15, 150); ax[1].set_ylim(190, 330)
ax[1].set_xlabel("trainable parameters (millions, log scale)"); ax[1].set_ylabel("dense hits@7 on dev (of 973)")
ax[1].set_title("(b) Phase 7: capacity vs. dense retrieval", loc="left")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_kd_capacity")

# ---------- Figure 4: Phase-4 training loss (incl. divergence) + Phase-6 decoding ----------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.45))
def smooth(v, k=8):
    v = np.array([np.nan if x is None else x for x in v], float); out = np.full_like(v, np.nan)
    for i in range(len(v)):
        w_ = v[max(0, i - k + 1): i + 1]
        out[i] = np.nan if np.isnan(w_).any() else w_.mean()
    return out
for key, k, lab in [("p4_dec01", "dec01", "Dec-01 + LoRA"), ("p4_dec02", "dec02", "Dec-02 + LoRMA"), ("p4_encdec", "encdec", "Enc-Dec + IA$^3$"),
                    ("p4_dec03", "dec03", "Dec-03 + rsLoRA (diverged)")]:
    a = np.array([s for s, _ in R[key]]); v = smooth([l for _, l in R[key]])
    ax[0].plot(a / 1000, v, color=C[k], lw=1.3 if k != "dec03" else 1.8, label=lab)
nan_at = next(s for s, l in R["p4_dec03"] if l is None)
ax[0].axvline(nan_at / 1000, color=C["dec03"], lw=0.9, ls=":")
ax[0].text(nan_at / 1000 + 0.15, 0.595, "loss = NaN\nfrom here", fontsize=6.8, color=INK)
ax[0].axhline(np.log(2), color=MUTED, lw=0.8, ls="--"); ax[0].text(11.0, 0.705, "ln 2 (chance)", fontsize=6.8, color=MUTED)
ax[0].set_ylim(0.2, 0.8); ax[0].set_xlabel("optimiser step (thousands)"); ax[0].set_ylabel("training loss (smoothed)")
ax[0].set_title("(a) Phase 4: training loss per run", loc="left"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,-0.28), fontsize=6.6, ncol=2)
mins = [0, 32, 64]
for k, lab, r1, emp in [("dec02", "Dec-02 + LoRMA", [0.1996, 0.5306, 0.5864], 0.79), ("encdec", "Enc-Dec + adapters", [0.2153, 0.5847, 0.6058], 0.84)]:
    ax[1].plot(mins, r1, color=C[k], marker="o", ms=4, label=lab)
    ax[1].annotate(f"{int(emp*100)}% chunks empty", (0, r1[0]), xytext=(5, -3 if k == "dec02" else 6), textcoords="offset points", fontsize=6.8, color=INK)
ax[1].axhline(0.575, color=MUTED, lw=0.8, ls="--"); ax[1].text(1, 0.59, "TF-IDF extractive baseline (0.575)", fontsize=6.8, color=MUTED)
ax[1].set_xticks(mins); ax[1].set_xlabel("minimum new tokens per chunk (greedy decoding)"); ax[1].set_ylabel("ROUGE-1 on validation")
ax[1].set_ylim(0.1, 0.68); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,-0.28), ncol=2)
ax[1].set_title("(b) Phase 6: same weights, different stopping rule", loc="left")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_p4_p6")
print(sorted(os.listdir(OUT)))

# ---------- Figure 6: dev metric vs epoch, Phases 2-5 ----------
EP = json.load(open("data/epochs.json"))
KEY = {"decoder 01": "dec01", "decoder 02": "dec02", "decoder 03": "dec03", "enc dec": "encdec", "kd student": "kd"}
LAB = {"dec01": "Dec-01", "dec02": "Dec-02", "dec03": "Dec-03", "encdec": "Enc-Dec", "kd": "KD student"}
TT = {"2": "P2 NER: dev macro-F1", "3": "P3 RR: dev select-score", "4": "P4 CJPE: dev macro-F1", "5": "P5 LSI: dev macro-F1"}
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.85), sharex=False)
for i, ph in enumerate(["2", "3", "4", "5"]):
    for m, pts in EP[ph].items():
        k = KEY[m]; pts = np.array(pts)
        ax[i].plot(pts[:, 0], pts[:, 1], color=C[k], marker="o", ms=2.5, lw=1.3, label=LAB[k])
        b = pts[:, 1].argmax(); ax[i].scatter(pts[b, 0], pts[b, 1], s=26, facecolor="none", edgecolor=C[k], lw=1.0, zorder=4)
    ax[i].set_title(TT[ph], loc="left", fontsize=7.8); ax[i].set_xlabel("epoch", fontsize=7.5)
    ax[i].tick_params(labelsize=7); ax[i].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12), fontsize=7.2)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_epochs")
