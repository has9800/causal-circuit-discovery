from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from src.phase3_train.freeze import freeze_all, unfreeze_path

Pair = Dict[str, str]


class PathTrainer:
    def __init__(self, model, train_pairs: List[Pair], eval_pairs: List[Pair], config: Dict[str, Any]):
        self.model = model
        self.train_pairs = train_pairs
        self.eval_pairs = eval_pairs
        self.cfg = config

        freeze_all(self.model)
        self.layers = unfreeze_path(
            self.model,
            heads=[tuple(h) for h in self.cfg["path_heads"]],
            unfreeze_mlps=self.cfg.get("unfreeze_mlps", True),
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found after unfreeze_path.")
        self.optimizer = torch.optim.Adam(params, lr=self.cfg["learning_rate"])
        self.batch_size = int(self.cfg["batch_size"])

    def _answer_token(self, answer: str) -> int:
        return int(self.model.to_single_token(answer))

    def _prompt_loss(self, prompt: str, answer: str) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt)
        target_id = self._answer_token(answer)
        logits = self.model(tokens)
        target = torch.tensor([target_id], device=logits.device)
        return F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)

    def train_one_epoch(self) -> float:
        self.model.train()
        shuffled = list(self.train_pairs)
        random.shuffle(shuffled)

        self.optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        pair_count = 0

        for idx, pair in enumerate(shuffled, start=1):
            causal_loss = self._prompt_loss(pair["causal"], pair["causal_answer"])
            corr_loss = self._prompt_loss(pair["correlational"], pair["correlational_answer"])
            pair_loss = (causal_loss + corr_loss) / 2.0
            pair_loss.backward()

            epoch_loss += pair_loss.item()
            pair_count += 1

            if idx % self.batch_size == 0 or idx == len(shuffled):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        return epoch_loss / max(pair_count, 1)

    @torch.no_grad()
    def evaluate(self, pairs: List[Pair]) -> Dict[str, Any]:
        self.model.eval()
        total = 0
        correct = 0
        diffs = []

        per_type = defaultdict(lambda: {"correct": 0, "total": 0})

        for pair in pairs:
            c_yes = self._answer_token(pair["causal_answer"])
            c_no = self._answer_token(pair["correlational_answer"])

            logits_c = self.model(self.model.to_tokens(pair["causal"]))[0, -1, :]
            diff_c = (logits_c[c_yes] - logits_c[c_no]).item()
            pred_c = diff_c > 0

            logits_r = self.model(self.model.to_tokens(pair["correlational"]))[0, -1, :]
            diff_r = (logits_r[c_no] - logits_r[c_yes]).item()
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
        return {
            "accuracy": correct / total if total else 0.0,
            "by_type": by_type,
            "mean_logit_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        }

    def _trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        trainable_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        state = self.model.state_dict()
        return {k: v.detach().cpu() for k, v in state.items() if k in trainable_names}

    def train(self) -> Dict[str, List[float]]:
        epochs = int(self.cfg["epochs"])
        eval_every = int(self.cfg.get("eval_every", 1))
        history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "eval_accuracy": [], "eval_logit_diff": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()
            eval_metrics = {"accuracy": float("nan"), "mean_logit_diff": float("nan")}
            if self.eval_pairs and epoch % eval_every == 0:
                eval_metrics = self.evaluate(self.eval_pairs)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["eval_accuracy"].append(eval_metrics["accuracy"])
            history["eval_logit_diff"].append(eval_metrics["mean_logit_diff"])

            print(
                f"[phase3] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"eval_acc={eval_metrics['accuracy']:.4f} "
                f"eval_logit_diff={eval_metrics['mean_logit_diff']:.4f}"
            )

        save_dir = Path(self.cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._trainable_state_dict(), save_dir / "path_params.pt")
        (save_dir / "history.json").write_text(json.dumps(history, indent=2))
        return history
