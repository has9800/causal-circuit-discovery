from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.phase3_train.evaluate import run_comparison
from src.phase3_train.freeze import select_random_heads
from src.phase3_train.train import PathTrainer
from src.record import load_pairs
from src.visualize import plot_comparison_table, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 path training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--random-only", action="store_true", help="Skip path-head training and run random baseline only")
    parser.add_argument("--debug", action="store_true", help="Use debug model and tiny dataset for quick pipeline checks")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_debug_overrides(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg["model"]["debug"] = True
    cfg["phase3"]["epochs"] = 1

    train_pairs = load_pairs(cfg["data"]["pairs_path"])[:3]
    eval_pairs = load_pairs(cfg["data"]["corr2cause_path"])[:1]
    return cfg, train_pairs, eval_pairs


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    _set_seed(int(cfg["phase3"].get("random_seed", 42)))

    if args.debug:
        cfg, train_pairs, eval_pairs = _apply_debug_overrides(cfg)
    else:
        train_pairs = load_pairs(cfg["data"]["pairs_path"])
        eval_pairs = load_pairs(cfg["data"]["corr2cause_path"])

    model = load_model(cfg["model"])
    save_dir = Path(cfg["phase3"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    random_params_path = None
    all_histories = {}

    if not args.random_only:
        trainer = PathTrainer(model, train_pairs, eval_pairs, cfg["phase3"])
        history = trainer.train()
        all_histories["path"] = history
        plot_training_curves(history, save_dir / "training_curves_path.html")

    if cfg["phase3"].get("random_baseline", False):
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        k = len(cfg["phase3"]["path_heads"])
        unique_layers = len({layer for layer, _ in cfg["phase3"]["path_heads"]})

        random_heads = select_random_heads(
            n_layers=n_layers,
            n_heads=n_heads,
            k=k,
            exclude=[tuple(h) for h in cfg["phase3"]["path_heads"]],
            seed=int(cfg["phase3"].get("random_seed", 42)),
            required_unique_layers=unique_layers,
        )
        cfg["phase3"]["random_heads"] = [list(h) for h in random_heads]

        random_cfg = deepcopy(cfg["phase3"])
        random_cfg["path_heads"] = [list(h) for h in random_heads]
        random_cfg["save_dir"] = str(save_dir / "random_baseline")

        random_model = load_model(cfg["model"])
        random_trainer = PathTrainer(random_model, train_pairs, eval_pairs, random_cfg)
        random_history = random_trainer.train()
        all_histories["random"] = random_history
        random_params_path = str(Path(random_cfg["save_dir"]) / "path_params.pt")
        plot_training_curves(random_history, save_dir / "training_curves_random.html")

    results = run_comparison(model, cfg, random_params_path=random_params_path)
    plot_comparison_table(results, save_dir / "comparison_by_type.html")
    (save_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    (save_dir / "histories.json").write_text(json.dumps(all_histories, indent=2))


if __name__ == "__main__":
    main()
