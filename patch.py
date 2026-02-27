"""
Activation patching: for each head, swap its output from the causal run
with the correlational run and measure if the answer changes.

Unlike differential activation (which finds heads that RESPOND differently),
this finds heads that are NECESSARY for the correct output.

Drop this in causa-heads/src/patch.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


Head = Tuple[int, int]


def patch_single_head(
    model,
    clean_prompt: str,
    corrupted_prompt: str,
    correct_token: str,
    incorrect_token: str,
) -> Tuple[np.ndarray, float]:
    """
    For each attention head: run the clean (causal) prompt but replace
    that one head's output with what it would have been on the corrupted
    (correlational) prompt. Measure how much the logit difference drops.

    Big drop = that head is necessary for the correct answer.

    Returns:
        effects: (n_layers, n_heads) array of logit diff changes
        clean_logit_diff: baseline logit difference
    """
    correct_id = model.to_single_token(correct_token)
    incorrect_id = model.to_single_token(incorrect_token)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Cache clean and corrupted activations
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_diff = (clean_logits[0, -1, correct_id] - clean_logits[0, -1, incorrect_id]).item()

    effects = np.zeros((n_layers, n_heads), dtype=np.float32)

    for layer in range(n_layers):
        for head in range(n_heads):

            def patch_hook(z, hook, _layer=layer, _head=head):
                corrupted_z = corrupted_cache[f"blocks.{_layer}.attn.hook_z"]
                # Only patch the one head, leave others untouched
                # Handle different sequence lengths
                min_seq = min(z.shape[1], corrupted_z.shape[1])
                z[:, :min_seq, _head, :] = corrupted_z[:, :min_seq, _head, :]
                return z

            hook_name = f"blocks.{layer}.attn.hook_z"

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, patch_hook)],
                )

            patched_diff = (
                patched_logits[0, -1, correct_id] - patched_logits[0, -1, incorrect_id]
            ).item()

            effects[layer, head] = patched_diff - clean_diff

    return effects, clean_diff


def run_activation_patching(
    model,
    pairs: List[Dict[str, str]],
) -> Dict[str, np.ndarray]:
    """Run activation patching across all pairs and aggregate."""

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_effects = np.zeros((n_layers, n_heads), dtype=np.float64)
    per_type_effects = defaultdict(lambda: np.zeros((n_layers, n_heads), dtype=np.float64))
    per_type_counts = defaultdict(int)
    clean_diffs = []
    n_valid = 0

    for i, pair in enumerate(tqdm(pairs, desc="Activation patching")):
        try:
            effects, clean_diff = patch_single_head(
                model=model,
                clean_prompt=pair["causal"],
                corrupted_prompt=pair["correlational"],
                correct_token=pair["causal_answer"],
                incorrect_token=pair["correlational_answer"],
            )

            all_effects += effects
            per_type_effects[pair["type"]] += effects
            per_type_counts[pair["type"]] += 1
            clean_diffs.append(clean_diff)
            n_valid += 1

        except Exception as e:
            print(f"  Skipped pair {i}: {e}")
            continue

    if n_valid > 0:
        all_effects /= n_valid
    for ptype in per_type_effects:
        if per_type_counts[ptype] > 0:
            per_type_effects[ptype] /= per_type_counts[ptype]

    # Rank by most negative effect (biggest drop = most important)
    flat_idx = np.argsort(all_effects.ravel())
    rank_indices = np.array(np.unravel_index(flat_idx, all_effects.shape)).T

    print(f"\nActivation patching complete ({n_valid}/{len(pairs)} pairs)")
    print(f"Mean clean logit diff: {np.mean(clean_diffs):.4f}")
    print(f"\nTop 10 heads by causal importance (most negative = most necessary):")
    for layer, head in rank_indices[:10]:
        effect = all_effects[layer, head]
        print(f"  L{layer:02d}H{head:02d}  effect={effect:+.4f}")

    return {
        "effects": all_effects.astype(np.float32),
        "rank_indices": rank_indices,
        "per_type_effects": {k: v.astype(np.float32) for k, v in per_type_effects.items()},
        "clean_diffs": np.array(clean_diffs),
        "n_valid": n_valid,
    }


def ablate_top_patched_heads(
    model,
    pairs: List[Dict[str, str]],
    rank_indices: np.ndarray,
    ablation_k: int,
    n_random_baselines: int,
) -> Dict[str, float]:
    """Ablation validation using the heads found by activation patching."""

    top_heads = [(int(l), int(h)) for l, h in rank_indices[:ablation_k]]
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]

    def accuracy_with_ablation(heads_to_ablate):
        per_layer = defaultdict(list)
        for layer, head in heads_to_ablate:
            per_layer[layer].append(head)

        def build_hook(heads):
            def _hook(z, hook):
                z[:, :, heads, :] = 0.0
                return z
            return _hook

        hooks = [
            (f"blocks.{layer}.attn.hook_z", build_hook(heads))
            for layer, heads in per_layer.items()
        ]

        correct = 0
        for pair in pairs:
            tokens = model.to_tokens(pair["causal"])
            correct_id = model.to_single_token(pair["causal_answer"])
            incorrect_id = model.to_single_token(pair["correlational_answer"])

            with torch.no_grad():
                if hooks:
                    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                else:
                    logits = model(tokens)

            if logits[0, -1, correct_id] > logits[0, -1, incorrect_id]:
                correct += 1

        return correct / max(1, len(pairs))

    full_acc = accuracy_with_ablation([])
    target_acc = accuracy_with_ablation(top_heads)

    random_scores = []
    for _ in tqdm(range(n_random_baselines), desc="Random baselines"):
        rh = random.sample(all_heads, k=ablation_k)
        random_scores.append(accuracy_with_ablation(rh))

    random_scores = np.array(random_scores)

    print(f"\nActivation-patching ablation (top {ablation_k} heads):")
    print(f"  Heads: {top_heads}")
    print(f"  Full model accuracy:        {full_acc:.3f}")
    print(f"  Top-head ablated accuracy:  {target_acc:.3f}")
    print(f"  Random ablated accuracy:    {random_scores.mean():.3f} ± {random_scores.std(ddof=1):.3f}")
    print(f"  Signal destruction:          {(1 - target_acc / full_acc) * 100:.1f}%")

    return {
        "full_accuracy": float(full_acc),
        "target_ablation_accuracy": float(target_acc),
        "random_ablation_mean": float(random_scores.mean()),
        "random_ablation_std": float(random_scores.std(ddof=1)),
        "top_heads": top_heads,
    }
