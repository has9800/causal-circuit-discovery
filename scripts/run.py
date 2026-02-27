from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablate import evaluate_ablation
from src.compare import compare_conditions, top_heads
from src.model import load_model
from src.record import load_pairs, record_pair_activations
from src.visualize import save_all_visuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find causal-reasoning attention heads in GPT-Neo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    model = load_model(cfg["model"])
    pairs = load_pairs(cfg["data"]["pairs_path"])

    record = record_pair_activations(model, pairs)

    comparison = compare_conditions(
        record,
        top_k=cfg["analysis"]["top_k"],
        significance_threshold=cfg["analysis"]["significance_threshold"],
    )

    heads = top_heads(comparison.rank_indices, cfg["analysis"]["ablation_k"])
    ablation_results = evaluate_ablation(
        model,
        pairs,
        heads,
        ablation_k=cfg["analysis"]["ablation_k"],
        n_random_baselines=cfg["analysis"]["n_random_baselines"],
    )

    save_all_visuals(comparison, ablation_results, cfg["output"]["dir"])
    print(f"Saved outputs to {cfg['output']['dir']}")


if __name__ == "__main__":
    main()
