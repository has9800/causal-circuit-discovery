from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

import torch

Head = Tuple[int, int]


def _count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_all(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    total, trainable = _count_params(model)
    print(f"[phase3] freeze_all: total_params={total:,}, trainable={trainable:,}")
    if trainable != 0:
        raise RuntimeError("freeze_all failed: some parameters are still trainable.")


def unfreeze_path(model, heads: Sequence[Head], unfreeze_mlps: bool) -> List[int]:
    if not heads:
        raise ValueError("heads list is empty.")

    layers = sorted({layer for layer, _ in heads})
    for layer in layers:
        block = model.blocks[layer]
        for _, param in block.attn.named_parameters():
            param.requires_grad = True

        if unfreeze_mlps:
            for _, param in block.mlp.named_parameters():
                param.requires_grad = True

    total, trainable = _count_params(model)
    ratio = (trainable / total) * 100 if total else 0.0
    components = "attn + mlp" if unfreeze_mlps else "attn-only"
    print(
        "[phase3] unfreeze_path: "
        f"layers={layers}, components={components}, trainable_params={trainable:,} ({ratio:.2f}%)"
    )
    return layers


def select_random_heads(
    n_layers: int,
    n_heads: int,
    k: int,
    exclude: Iterable[Head],
    seed: int,
    required_unique_layers: int | None = None,
) -> List[Head]:
    rng = random.Random(seed)
    exclude_set = set(exclude)
    candidates = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in exclude_set]
    if len(candidates) < k:
        raise ValueError("Not enough candidate heads available to sample from.")

    if required_unique_layers is None:
        selection = rng.sample(candidates, k)
        return sorted(selection)

    available_layers = sorted({layer for layer, _ in candidates})
    if len(available_layers) < required_unique_layers:
        raise ValueError("Not enough layers available to satisfy required_unique_layers.")

    chosen_layers = rng.sample(available_layers, required_unique_layers)
    per_layer = {layer: [head for l, head in candidates if l == layer] for layer in chosen_layers}

    selection: list[Head] = []
    for layer in chosen_layers:
        head = rng.choice(per_layer[layer])
        selection.append((layer, head))

    remaining = [c for c in candidates if c[0] in chosen_layers and c not in selection]
    if len(selection) < k:
        selection.extend(rng.sample(remaining, k - len(selection)))

    return sorted(selection)
