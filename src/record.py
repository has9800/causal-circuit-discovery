from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

REQUIRED_KEYS = {"causal", "correlational", "causal_answer", "correlational_answer", "type"}


Pair = Dict[str, str]


def load_pairs(pairs_path: str) -> List[Pair]:
    path = Path(pairs_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {pairs_path}. Please provide a valid data/pairs.json file."
        )

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {pairs_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("pairs.json must contain a top-level list of prompt-pair objects.")

    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Entry at index {i} is not an object.")
        missing = REQUIRED_KEYS - set(item.keys())
        if missing:
            raise ValueError(f"Entry at index {i} missing required keys: {sorted(missing)}")
        for key in REQUIRED_KEYS:
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"Entry at index {i} has invalid non-empty string field: '{key}'")

    counts = Counter(p["type"] for p in payload)
    print(f"Loaded {len(payload)} pairs from {pairs_path}")
    for ptype, count in sorted(counts.items()):
        print(f"  type={ptype:<16} count={count}")
    return payload


def _single_token_id(model, token_str: str) -> int:
    token_id = model.to_single_token(token_str)
    if token_id is None:
        raise ValueError(f"Could not map answer token '{token_str}' to a single token.")
    return token_id


def _run_prompt(model, prompt: str, correct_tok: int, incorrect_tok: int) -> Tuple[np.ndarray, np.ndarray, float]:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[1]

    magnitudes = np.zeros((n_layers, n_heads), dtype=np.float32)
    patterns = np.zeros((n_layers, n_heads, seq_len), dtype=np.float32)

    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

        # z: (batch, seq_len, n_heads, d_head)
        last_z = z[0, -1, :, :]
        magnitudes[layer, :] = torch.linalg.vector_norm(last_z, dim=-1).detach().cpu().numpy()

        # pattern: (batch, n_heads, seq_len, seq_len)
        last_pattern = pattern[0, :, -1, :]
        patterns[layer, :, :] = last_pattern.detach().cpu().numpy()

    last_logits = logits[0, -1, :]
    logit_diff = (last_logits[correct_tok] - last_logits[incorrect_tok]).item()

    return magnitudes, patterns, logit_diff


def record_pair_activations(model, pairs: List[Pair]) -> Dict[str, np.ndarray]:
    n_pairs = len(pairs)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    max_seq_len = 0
    for pair in pairs:
        max_seq_len = max(max_seq_len, model.to_tokens(pair["causal"]).shape[1])
        max_seq_len = max(max_seq_len, model.to_tokens(pair["correlational"]).shape[1])

    causal_magnitude = np.zeros((n_pairs, n_layers, n_heads), dtype=np.float32)
    corr_magnitude = np.zeros((n_pairs, n_layers, n_heads), dtype=np.float32)
    causal_pattern = np.zeros((n_pairs, n_layers, n_heads, max_seq_len), dtype=np.float32)
    corr_pattern = np.zeros((n_pairs, n_layers, n_heads, max_seq_len), dtype=np.float32)
    causal_logit_diff = np.zeros((n_pairs,), dtype=np.float32)
    corr_logit_diff = np.zeros((n_pairs,), dtype=np.float32)

    pair_types = []

    for i, pair in enumerate(tqdm(pairs, desc="Recording activations")):
        c_correct = _single_token_id(model, pair["causal_answer"])
        c_incorrect = _single_token_id(model, pair["correlational_answer"])
        r_correct = _single_token_id(model, pair["correlational_answer"])
        r_incorrect = _single_token_id(model, pair["causal_answer"])

        c_mag, c_pat, c_ld = _run_prompt(model, pair["causal"], c_correct, c_incorrect)
        r_mag, r_pat, r_ld = _run_prompt(model, pair["correlational"], r_correct, r_incorrect)

        causal_magnitude[i] = c_mag
        corr_magnitude[i] = r_mag

        causal_pattern[i, :, :, : c_pat.shape[-1]] = c_pat
        corr_pattern[i, :, :, : r_pat.shape[-1]] = r_pat

        causal_logit_diff[i] = c_ld
        corr_logit_diff[i] = r_ld
        pair_types.append(pair["type"])

    print(f"Processed {n_pairs} pairs for activation recording.")
    return {
        "causal_magnitude": causal_magnitude,
        "corr_magnitude": corr_magnitude,
        "causal_pattern": causal_pattern,
        "corr_pattern": corr_pattern,
        "causal_logit_diff": causal_logit_diff,
        "corr_logit_diff": corr_logit_diff,
        "pair_types": np.array(pair_types),
    }
