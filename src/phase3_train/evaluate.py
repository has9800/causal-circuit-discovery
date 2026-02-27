from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from src.model import load_model
from src.phase3_train.freeze import freeze_all, unfreeze_path

Pair = Dict[str, str]
Head = Tuple[int, int]


def _load_pairs(path: str | Path) -> List[Pair]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("corr2cause payload must be a list")
    return payload


@torch.no_grad()
def benchmark_corr2cause(model, corr2cause_path: str | Path) -> Dict[str, Any]:
    pairs = _load_pairs(corr2cause_path)

    total = 0
    correct = 0
    diffs: list[float] = []
    per_type = defaultdict(lambda: {"correct": 0, "total": 0})

    model.eval()
    for pair in pairs:
        causal_id = int(model.to_single_token(pair["causal_answer"]))
        corr_id = int(model.to_single_token(pair["correlational_answer"]))

        logits_c = model(model.to_tokens(pair["causal"]))[0, -1, :]
        diff_c = (logits_c[causal_id] - logits_c[corr_id]).item()
        pred_c = diff_c > 0

        logits_r = model(model.to_tokens(pair["correlational"]))[0, -1, :]
        diff_r = (logits_r[corr_id] - logits_r[causal_id]).item()
        pred_r = diff_r > 0

        pair_correct = int(pred_c) + int(pred_r)
        correct += pair_correct
        total += 2
        diffs.extend([diff_c, diff_r])

        per_type[pair["type"]]["correct"] += pair_correct
        per_type[pair["type"]]["total"] += 2

    by_type = {
        ptype: vals["correct"] / vals["total"] if vals["total"] else 0.0
        for ptype, vals in per_type.items()
    }

    results = {
        "accuracy": correct / total if total else 0.0,
        "by_type": by_type,
        "mean_logit_diff": sum(diffs) / len(diffs) if diffs else 0.0,
    }

    print("\n[phase3] corr2cause benchmark")
    print(f"  overall_accuracy: {results['accuracy']:.3f}")
    print(f"  mean_logit_diff : {results['mean_logit_diff']:.3f}")
    for ptype, acc in sorted(by_type.items()):
        print(f"  {ptype:<16}: {acc:.3f}")
    return results


def _load_path_params(model, save_path: Path) -> None:
    state = torch.load(save_path, map_location=model.cfg.device)
    model.load_state_dict(state, strict=False)


def _zero_selected_heads(model, heads: Sequence[Head]) -> None:
    n_heads = model.cfg.n_heads

    for layer, head in heads:
        attn = model.blocks[layer].attn
        for name, param in attn.named_parameters():
            if not any(k in name for k in ["W_Q", "W_K", "W_V", "W_O"]):
                continue

            data = param.data
            if data.ndim >= 1 and data.shape[0] == n_heads:
                data[head] = 0.0
            elif data.ndim >= 2 and data.shape[1] == n_heads:
                data[:, head] = 0.0


def run_comparison(model, config: Dict[str, Any], random_params_path: str | None = None) -> Dict[str, Dict[str, Any]]:
    phase3_cfg = config["phase3"]
    corr2cause_path = config["data"]["corr2cause_path"]
    save_dir = Path(phase3_cfg["save_dir"])
    path_weights = save_dir / "path_params.pt"

    conditions: Dict[str, Dict[str, Any]] = {}

    base_model = load_model(deepcopy(config["model"]))
    conditions["Base GPT-J"] = benchmark_corr2cause(base_model, corr2cause_path)

    path_model = load_model(deepcopy(config["model"]))
    freeze_all(path_model)
    unfreeze_path(path_model, [tuple(h) for h in phase3_cfg["path_heads"]], phase3_cfg.get("unfreeze_mlps", True))
    _load_path_params(path_model, path_weights)
    conditions["Path-trained"] = benchmark_corr2cause(path_model, corr2cause_path)

    ablated_model = load_model(deepcopy(config["model"]))
    freeze_all(ablated_model)
    unfreeze_path(ablated_model, [tuple(h) for h in phase3_cfg["path_heads"]], phase3_cfg.get("unfreeze_mlps", True))
    _load_path_params(ablated_model, path_weights)
    _zero_selected_heads(ablated_model, [tuple(h) for h in phase3_cfg["path_heads"]])
    conditions["Path-ablated"] = benchmark_corr2cause(ablated_model, corr2cause_path)

    if random_params_path:
        random_model = load_model(deepcopy(config["model"]))
        freeze_all(random_model)
        unfreeze_path(random_model, [tuple(h) for h in phase3_cfg["random_heads"]], phase3_cfg.get("unfreeze_mlps", True))
        _load_path_params(random_model, Path(random_params_path))
        conditions["Random-path-trained"] = benchmark_corr2cause(random_model, corr2cause_path)

    all_types = sorted({t for res in conditions.values() for t in res["by_type"].keys()})
    print("\n[phase3] comparison")
    header = "Condition".ljust(28) + "Accuracy".rjust(10) + "  " + "  ".join(t.title() for t in all_types)
    print(header)
    for name, result in conditions.items():
        row = name.ljust(28) + f"{result['accuracy']:.3f}".rjust(10)
        for t in all_types:
            row += f"  {result['by_type'].get(t, 0.0):.3f}"
        print(row)

    return conditions
