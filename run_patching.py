"""Drop this in causa-heads/ and run: python run_patching.py"""

import yaml
import numpy as np
from src.model import load_model
from src.record import load_pairs
from src.patch import run_activation_patching, ablate_top_patched_heads
from src.visualize import plot_heatmap, plot_ablation_bar
from pathlib import Path

cfg = yaml.safe_load(open("config.yaml").read())
model = load_model(cfg["model"])
pairs = load_pairs(cfg["data"]["pairs_path"])

# Run activation patching
results = run_activation_patching(model, pairs)

# Save heatmap
out = Path(cfg["output"]["dir"])
out.mkdir(parents=True, exist_ok=True)
np.save(out / "patching_effects.npy", results["effects"])

plot_heatmap(
    results["effects"],
    "Activation Patching: Causal Importance per Head",
    out / "heatmap_patching.html",
    colorscale="RdBu_r",
    zmid=0.0,
)

# Per-type heatmaps
for ptype, effects in results["per_type_effects"].items():
    safe = ptype.replace(" ", "_").lower()
    plot_heatmap(
        effects,
        f"Activation Patching ({ptype})",
        out / f"heatmap_patching_{safe}.html",
    )

# Ablation on patching-identified heads
ablation = ablate_top_patched_heads(
    model, pairs,
    results["rank_indices"],
    ablation_k=cfg["analysis"]["ablation_k"],
    n_random_baselines=cfg["analysis"]["n_random_baselines"],
)
plot_ablation_bar(ablation, out / "ablation_patching.html")

print(f"\nResults saved to {out}")
