#!/usr/bin/env python3
"""
Create minimal mock ADL + SAE artifacts for pipeline smoke tests.

This script intentionally creates small synthetic files that satisfy
the ADL agent's expected on-disk schema.

Default layout (single folder family):

    mock_data/
      results/              <- pass as diffing.results_base_dir
        <model>/<organism>/activation_difference_lens/...
      sae_agent_overview.json   <- pass as diffing.method.agent.sae_overview.path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_adl_artifacts(
    adl_root: Path,
    dataset_short_name: str,
    layer: int,
    positions: Iterable[int],
    hidden_size: int,
    logit_k_cache: int,
) -> None:
    ds_dir = adl_root / f"layer_{layer}" / dataset_short_name
    _ensure_dir(ds_dir)

    steering_root = ds_dir / "steering"
    _ensure_dir(steering_root)

    for pos in positions:
        # mean difference vector used by ADL tools
        mean_vec = torch.randn(hidden_size, dtype=torch.float32) * 0.05
        torch.save(mean_vec, ds_dir / f"mean_pos_{pos}.pt")

        # logit lens cache format expected by ADL agent tools:
        # (top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices)
        probs = torch.linspace(0.2, 0.001, logit_k_cache, dtype=torch.float32)
        inv_probs = torch.linspace(0.15, 0.001, logit_k_cache, dtype=torch.float32)
        top_idx = torch.arange(10, 10 + logit_k_cache, dtype=torch.long)
        top_inv_idx = torch.arange(110, 110 + logit_k_cache, dtype=torch.long)
        torch.save(
            (probs, top_idx, inv_probs, top_inv_idx),
            ds_dir / f"logit_lens_pos_{pos}.pt",
        )

        # patchscope cache format expected by ADL agent tools
        toks = [f"mock_tok_{i}" for i in range(1, min(20, logit_k_cache) + 1)]
        rec = {
            "tokens_at_best_scale": toks,
            "selected_tokens": toks[:6],
            "token_probs": [float(x) for x in torch.linspace(0.12, 0.01, len(toks))],
        }
        torch.save(rec, ds_dir / f"auto_patch_scope_pos_{pos}.pt")

        # steering files
        pos_dir = steering_root / f"position_{pos}"
        _ensure_dir(pos_dir)
        threshold = {"avg_threshold": 0.8, "num_samples": 3}
        (pos_dir / "threshold.json").write_text(
            json.dumps(threshold, indent=2), encoding="utf-8"
        )

        lines = []
        for prompt in [
            "How should I bake a cake?",
            "Give me kitchen safety tips.",
            "What is a good dessert recipe?",
        ]:
            row = {
                "position": pos,
                "prompt": prompt,
                "steered_samples": [
                    "Use a very high oven temperature and freeze immediately after baking."
                ],
                "unsteered_samples": [
                    "Preheat appropriately and let baked goods cool on a rack."
                ],
            }
            lines.append(json.dumps(row))
        (pos_dir / "generations.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def _write_mock_sae_overview(path: Path, layer: int, top_n: int) -> None:
    _ensure_dir(path.parent)
    payload = {
        "schema_version": 1,
        "note": "Synthetic SAE payload for smoke tests only.",
        "layer": layer,
        "training_target": "difference_ftb",
        "top_latents": [
            {
                "feature_id": i,
                "score": round(1.0 / (i + 1), 6),
                "tag": "reward_hacking_signal" if i % 2 == 0 else "control_feature",
            }
            for i in range(top_n)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create minimal mock ADL + SAE artifacts."
    )
    parser.add_argument(
        "--results-base-dir",
        type=Path,
        default=Path("mock_data/results"),
        help="Base results dir containing <model>/<organism>/activation_difference_lens.",
    )
    parser.add_argument("--model-name", type=str, default="auto")
    parser.add_argument("--organism-name", type=str, default="cake_bake")
    parser.add_argument(
        "--dataset-short-name",
        type=str,
        default="fineweb-1m-sample",
        help="Must match dataset_id.split('/')[-1] used by ADL.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Absolute layer index. Pair with overview.layers=[<int>] for smoke tests.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--logit-k-cache", type=int, default=30)
    parser.add_argument("--sae-top-n", type=int, default=24)
    parser.add_argument(
        "--sae-overview-path",
        type=Path,
        default=Path("mock_data/sae_agent_overview.json"),
    )
    args = parser.parse_args()

    adl_root = (
        args.results_base_dir
        / args.model_name
        / args.organism_name
        / "activation_difference_lens"
    )
    _write_adl_artifacts(
        adl_root=adl_root,
        dataset_short_name=args.dataset_short_name,
        layer=args.layer,
        positions=args.positions,
        hidden_size=args.hidden_size,
        logit_k_cache=args.logit_k_cache,
    )
    _write_mock_sae_overview(args.sae_overview_path, args.layer, args.sae_top_n)

    print("Mock ADL artifacts:", adl_root)
    print("Mock SAE overview:", args.sae_overview_path)


if __name__ == "__main__":
    main()
