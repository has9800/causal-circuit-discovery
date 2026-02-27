from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


Head = Tuple[int, int]


def _accuracy_on_pairs(model, pairs, ablate_heads: Sequence[Head] | None = None) -> float:
    hooks = []
    if ablate_heads:
        per_layer: Dict[int, List[int]] = {}
        for layer, head in ablate_heads:
            per_layer.setdefault(layer, []).append(head)

        def build_hook(heads):
            def _hook(z, hook):
                z[:, :, heads, :] = 0.0
                return z

            return _hook

        for layer, heads in per_layer.items():
            hooks.append((f"blocks.{layer}.attn.hook_z", build_hook(heads)))

    n_correct = 0
    n_total = len(pairs)

    for pair in tqdm(pairs, desc="Evaluating ablation", leave=False):
        tokens = model.to_tokens(pair["causal"])
        correct_id = model.to_single_token(pair["causal_answer"])
        incorrect_id = model.to_single_token(pair["correlational_answer"])

        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)

        last = logits[0, -1]
        if last[correct_id] > last[incorrect_id]:
            n_correct += 1

    return n_correct / max(1, n_total)


def evaluate_ablation(
    model,
    pairs,
    top_heads: Sequence[Head],
    ablation_k: int,
    n_random_baselines: int,
) -> Dict[str, float]:
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    chosen_heads = list(top_heads)[:ablation_k]

    full_acc = _accuracy_on_pairs(model, pairs, ablate_heads=None)
    target_acc = _accuracy_on_pairs(model, pairs, ablate_heads=chosen_heads)

    random_scores = []
    for _ in tqdm(range(n_random_baselines), desc="Random baselines"):
        random_heads = random.sample(all_heads, k=ablation_k)
        random_scores.append(_accuracy_on_pairs(model, pairs, ablate_heads=random_heads))

    random_scores = np.array(random_scores, dtype=np.float32)

    print("Ablation summary:")
    print(f"  Full model causal accuracy:      {full_acc:.3f}")
    print(f"  Top-head ablated causal accuracy:{target_acc:.3f}")
    print(
        "  Random-head ablated accuracy:    "
        f"{random_scores.mean():.3f} ± {random_scores.std(ddof=1) if len(random_scores) > 1 else 0.0:.3f}"
    )

    return {
        "full_accuracy": float(full_acc),
        "target_ablation_accuracy": float(target_acc),
        "random_ablation_mean": float(random_scores.mean()),
        "random_ablation_std": float(random_scores.std(ddof=1) if len(random_scores) > 1 else 0.0),
    }
