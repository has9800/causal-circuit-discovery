from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import plotly.graph_objects as go


def _save_fig(fig: go.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def plot_heatmap(values: np.ndarray, title: str, out_path: Path, colorscale: str = "RdBu_r", zmid: float | None = 0.0) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            colorscale=colorscale,
            zmid=zmid,
            colorbar={"title": title},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Head",
        yaxis_title="Layer",
    )
    _save_fig(fig, out_path)


def plot_ablation_bar(results: Dict[str, float], out_path: Path) -> None:
    labels = ["full", "top-head ablated", "random ablated"]
    values = [
        results["full_accuracy"],
        results["target_ablation_accuracy"],
        results["random_ablation_mean"],
    ]
    errors = [0.0, 0.0, results["random_ablation_std"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                error_y={"type": "data", "array": errors, "visible": True},
            )
        ]
    )
    fig.update_layout(title="Causal Accuracy Under Ablation", yaxis_title="Accuracy")
    _save_fig(fig, out_path)


def save_all_visuals(comparison, ablation_results: Dict[str, float], out_dir: str) -> None:
    out_root = Path(out_dir)
    plot_heatmap(
        comparison.differential_activation,
        "Differential Activation (Causal vs Correlational)",
        out_root / "heatmap_differential_activation.html",
    )
    plot_heatmap(
        comparison.p_values,
        "Paired t-test p-values",
        out_root / "heatmap_p_values.html",
        colorscale="Viridis",
        zmid=None,
    )
    plot_ablation_bar(ablation_results, out_root / "ablation_comparison.html")

    if len(comparison.by_type) > 1:
        for ptype, metrics in comparison.by_type.items():
            safe = ptype.replace(" ", "_").lower()
            plot_heatmap(
                metrics["differential_activation"],
                f"Differential Activation ({ptype})",
                out_root / f"heatmap_differential_{safe}.html",
            )
