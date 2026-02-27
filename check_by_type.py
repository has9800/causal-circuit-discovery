"""Drop this in causa-heads/ and run: python check_by_type.py"""

import json
import torch
from src.model import load_model
from src.record import load_pairs
import yaml
from collections import defaultdict

cfg = yaml.safe_load(open("config.yaml").read())
model = load_model(cfg["model"])
pairs = load_pairs(cfg["data"]["pairs_path"])

type_correct = defaultdict(int)
type_total = defaultdict(int)
type_logit_diffs = defaultdict(list)

for pair in pairs:
    tokens = model.to_tokens(pair["causal"])
    correct_id = model.to_single_token(pair["causal_answer"])
    incorrect_id = model.to_single_token(pair["correlational_answer"])

    with torch.no_grad():
        logits = model(tokens)

    last = logits[0, -1]
    diff = (last[correct_id] - last[incorrect_id]).item()

    ptype = pair["type"]
    type_total[ptype] += 1
    type_logit_diffs[ptype].append(diff)
    if last[correct_id] > last[incorrect_id]:
        type_correct[ptype] += 1

print(f"\n{'Type':<30} {'Acc':>6} {'N':>4} {'Mean Logit Diff':>16}")
print("-" * 60)
for ptype in sorted(type_total.keys()):
    acc = type_correct[ptype] / type_total[ptype]
    mean_diff = sum(type_logit_diffs[ptype]) / len(type_logit_diffs[ptype])
    print(f"{ptype:<30} {acc:>6.3f} {type_total[ptype]:>4} {mean_diff:>16.4f}")
