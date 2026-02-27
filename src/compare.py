from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import ttest_rel
from tqdm import tqdm


EPS = 1e-8


@dataclass
class ComparisonResults:
    differential_activation: np.ndarray
    activation_ratio: np.ndarray
    attention_kl: np.ndarray
    p_values: np.ndarray
    significant_mask: np.ndarray
    rank_indices: np.ndarray
    by_type: Dict[str, Dict[str, np.ndarray]]


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def _compute_metrics(
    causal_mag: np.ndarray,
    corr_mag: np.ndarray,
    causal_pattern: np.ndarray,
    corr_pattern: np.ndarray,
    significance_threshold: float,
) -> Dict[str, np.ndarray]:
    differential = np.mean(np.abs(causal_mag - corr_mag), axis=0)
    activation_ratio = np.mean(causal_mag, axis=0) / (np.mean(corr_mag, axis=0) + EPS)

    kls = []
    for i in tqdm(range(causal_pattern.shape[0]), desc="Computing KL", leave=False):
        kls.append(kl_divergence(causal_pattern[i], corr_pattern[i]))
    attention_kl = np.mean(np.stack(kls, axis=0), axis=0)

    _, p_values = ttest_rel(causal_mag, corr_mag, axis=0, nan_policy="omit")
    p_values = np.nan_to_num(p_values, nan=1.0)
    significant = p_values < significance_threshold

    return {
        "differential_activation": differential,
        "activation_ratio": activation_ratio,
        "attention_kl": attention_kl,
        "p_values": p_values,
        "significant_mask": significant,
    }


def compare_conditions(
    record: Dict[str, np.ndarray], top_k: int, significance_threshold: float
) -> ComparisonResults:
    metrics = _compute_metrics(
        record["causal_magnitude"],
        record["corr_magnitude"],
        record["causal_pattern"],
        record["corr_pattern"],
        significance_threshold,
    )

    flat_idx = np.argsort(metrics["differential_activation"].ravel())[::-1]
    rank_indices = np.array(np.unravel_index(flat_idx, metrics["differential_activation"].shape)).T

    print("Top heads by differential activation:")
    for layer, head in rank_indices[:top_k]:
        diff = metrics["differential_activation"][layer, head]
        pval = metrics["p_values"][layer, head]
        print(f"  L{layer:02d}H{head:02d} diff={diff:.4f} p={pval:.3e}")

    by_type = {}
    pair_types = record["pair_types"]
    for ptype in sorted(set(pair_types.tolist())):
        idx = np.where(pair_types == ptype)[0]
        by_type[ptype] = _compute_metrics(
            record["causal_magnitude"][idx],
            record["corr_magnitude"][idx],
            record["causal_pattern"][idx],
            record["corr_pattern"][idx],
            significance_threshold,
        )

    return ComparisonResults(
        differential_activation=metrics["differential_activation"],
        activation_ratio=metrics["activation_ratio"],
        attention_kl=metrics["attention_kl"],
        p_values=metrics["p_values"],
        significant_mask=metrics["significant_mask"],
        rank_indices=rank_indices,
        by_type=by_type,
    )


def top_heads(rank_indices: np.ndarray, k: int) -> List[tuple[int, int]]:
    return [tuple(map(int, x)) for x in rank_indices[:k]]
