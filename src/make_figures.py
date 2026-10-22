"""Build the README figures and the holdout summary from saved results.

Every figure is rendered twice (light / dark) so the README can switch with
GitHub's theme. Run after `evaluate.py` has produced the holdout predictions.

Example:
    python src/make_figures.py --data-root data/landmarks
"""

import argparse
import json
import os
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

from data import pretty_label

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834"], "before": "#86b6ef", "after": "#2a78d6",
        "good": "#0ca30c", "bad": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926"], "before": "#184f95", "after": "#3987e5",
        "good": "#0ca30c", "bad": "#d03b3b",
    },
}

# Holdout images shown in the prediction grid: four hits, then four misses
# (two duplicate-label misses, two other errors). Chosen by hand to avoid
# stock-photo watermarks.
EXAMPLE_IMAGES = [
    "Taj_Mahal_in_Agra_in_India/6.93080_square.jpg.webp",
    "Eiffel_Tower_in_Paris_France/34.5144.jpg.webp",
    "Machu_Picchu_Andes_Mountains_Peru/25.92484.jpg.webp",
    "Statue_of_Liberty_National_Monument_in_New_York_City/15.united-states-statue-of-liberty-vertical_3x4.jpg.webp",
    "Casa_Mila_in_Barcelona_Spain/6.casa-mila-la-pedrera-barcelona-full-guide-street-view.jpg.webp",
    "Blue_Mosque_in_Istanbul_Turkey/26.sultan-ahmed-blue-mosque-istanbul-turkey_159938-2460.jpg.webp",
    "Palace_of_Versailles_in_Versailles_France/41.main-entrance-versailles-palace-versailles-france_29285-356.jpg.webp",
    "Flatiron_Building_in_New_York_City_New_York_United_States/20.low-angle-view-flatiron-building-manhattan-new-york-city-new-york-state-usa_19485-19734.jpg.webp",
]


def setup(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })
    return t


def save(fig, out_dir, name, theme):
    path = os.path.join(out_dir, f"{name}_{theme}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def canonical_map(aliases):
    return {label: group[0] for group in aliases for label in group}


def holdout_summary(results_dir, aliases):
    canon = canonical_map(aliases)
    summary = {}
    for tag in ("zeroshot", "finetuned"):
        df = pd.read_csv(os.path.join(results_dir, f"{tag}_holdout_predictions.csv"))
        alias_ok = df.true_label.map(lambda l: canon.get(l, l)) == df.pred_label.map(lambda l: canon.get(l, l))
        summary[tag] = {
            "n_images": len(df),
            "top1": round(100 * df.top1_correct.mean(), 2),
            "top5": round(100 * df.top5_correct.mean(), 2),
            "exact_match": round(100 * df.exact_match.mean(), 2),
            "errors": int((df.top1_correct == 0).sum()),
            "errors_from_alias_labels": int(alias_ok.sum() - df.top1_correct.sum()),
            "top1_alias_merged": round(100 * alias_ok.mean(), 2),
        }
    return summary


def accuracy_chart(original, holdout, out_dir, theme):
    t = setup(theme)
    datasets = [
        (f"Original test split (n={original['n_test_images']:,})", original),
        (f"Unseen .webp holdout (n={holdout['finetuned']['n_images']})", holdout),
    ]
    positions = [0, 1, 2.6, 3.6]
    values = {
        key: [d[key][m] for _, d in datasets for m in ("top1", "top5")]
        for key in ("zeroshot", "finetuned")
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.34
    for offset, key, color, name in [
        (-width / 2 - 0.01, "zeroshot", t["before"], "Zero-shot BLIP"),
        (width / 2 + 0.01, "finetuned", t["after"], "Fine-tuned (LoRA + ITC)"),
    ]:
        xs = [p + offset for p in positions]
        ax.bar(xs, values[key], width, color=color, label=name, zorder=2)
        for xi, v in zip(xs, values[key]):
            ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=t["ink2"])
    ax.set_xticks(positions)
    ax.set_xticklabels(["Top-1", "Top-5"] * 2)
    for (name, _), center in zip(datasets, [0.5, 3.1]):
        ax.text(center, -0.13, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9.5, color=t["muted"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.grid(True, color=t["grid"], linewidth=1, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.12), labelcolor=t["ink2"])
    save(fig, out_dir, "accuracy", theme)


def training_curves(log_csv, out_dir, theme):
    t = setup(theme)
    log = pd.read_csv(log_csv)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for col, name, color in [
        ("cap_loss", "Captioning loss", t["series"][0]),
        ("itc_loss", "ITC loss", t["series"][1]),
    ]:
        ax.plot(log.epoch, log[col], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=t["surface"], markeredgewidth=1.5, label=name, zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training loss")
    ax.set_xticks(log.epoch)
    ax.set_xlim(0.6, log.epoch.max() + 0.4)
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, color=t["grid"], linewidth=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper right", labelcolor=t["ink2"])
    save(fig, out_dir, "training_curves", theme)


def error_breakdown(results_dir, aliases, out_dir, theme, top_n=8):
    t = setup(theme)
    canon = canonical_map(aliases)
    df = pd.read_csv(os.path.join(results_dir, "finetuned_holdout_predictions.csv"))
    errors = df[df.top1_correct == 0]
    pairs = errors.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False).head(top_n)
    pairs = pairs.iloc[::-1]
    is_alias = [canon.get(a, a) == canon.get(b, b) for a, b in pairs.index]
    labels = [f"{pretty_label(a)}  →  {pretty_label(b)}" for a, b in pairs.index]
    colors = [t["series"][0] if alias else t["series"][1] for alias in is_alias]

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * top_n + 1))
    ax.barh(range(len(pairs)), pairs.values, height=0.6, color=colors, zorder=2)
    for i, v in enumerate(pairs.values):
        ax.text(v + 0.06, i, str(v), va="center", fontsize=9.5, color=t["ink2"])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Misclassified holdout images")
    ax.xaxis.grid(True, color=t["grid"], linewidth=1, zorder=0)
    ax.set_xticks(range(0, int(pairs.max()) + 2))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=t["series"][0], label="Same landmark, duplicate label"),
        plt.Rectangle((0, 0), 1, 1, color=t["series"][1], label="Other errors"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", labelcolor=t["ink2"], fontsize=9)
    save(fig, out_dir, "error_breakdown", theme)


def square(path, size=360):
    return ImageOps.fit(Image.open(path).convert("RGB"), (size, size))


def prediction_grid(results_dir, data_root, out_dir, theme):
    t = setup(theme)
    df = pd.read_csv(os.path.join(results_dir, "finetuned_holdout_predictions.csv"))
    picks = [df[df.image == image].iloc[0] for image in EXAMPLE_IMAGES]

    fig, axes = plt.subplots(2, 4, figsize=(13, 8.2))
    for ax, row in zip(axes.flat, picks):
        ax.imshow(square(os.path.join(data_root, row.image)))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ok = bool(row.top1_correct)
        mark, color = ("✓", t["good"]) if ok else ("✗", t["bad"])
        title = "\n".join(
            textwrap.fill(f"{k}: {pretty_label(v)}", width=34, subsequent_indent="      ")
            for k, v in [("True", row.true_label), ("Pred", row.pred_label)]
        )
        ax.set_title(title, fontsize=9, color=t["ink2"], loc="left", pad=6)
        ax.text(0.03, 0.97, f" {mark} {'correct' if ok else 'wrong'} ", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#ffffff", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none"})
    fig.subplots_adjust(wspace=0.08, hspace=0.42)
    save(fig, out_dir, "predictions", theme)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="assets")
    p.add_argument("--train-log", default=None,
                   help="Per-epoch loss CSV from train.py (default: <results-dir>/train_log.csv)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.results_dir, "label_aliases.json")) as f:
        aliases = json.load(f)["aliases"]
    with open(os.path.join(args.results_dir, "original_test_metrics.json")) as f:
        original = json.load(f)

    summary = holdout_summary(args.results_dir, aliases)
    with open(os.path.join(args.results_dir, "holdout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    for theme in THEMES:
        accuracy_chart(original, summary, args.out_dir, theme)
        training_curves(args.train_log or os.path.join(args.results_dir, "train_log.csv"), args.out_dir, theme)
        error_breakdown(args.results_dir, aliases, args.out_dir, theme)
        prediction_grid(args.results_dir, args.data_root, args.out_dir, theme)


if __name__ == "__main__":
    main()
