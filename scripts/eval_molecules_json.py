#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_loader.utils import load_valid_and_test_data
from src.eval import MoleculeEvaluator


TASK_NAMES = ["IntDiv", "novelty", "FCD", "SNN", "Frag", "Scaf"]


def _normalize_stats(stats: Any) -> dict:
    if isinstance(stats, dict):
        return stats
    if isinstance(stats, pd.Series):
        return stats.to_dict()
    if isinstance(stats, pd.DataFrame):
        if len(stats) == 1:
            return stats.iloc[0].to_dict()
        return {column: stats[column].iloc[0] for column in stats.columns}
    raise TypeError(f"Unsupported stats object type: {type(stats).__name__}")


def _load_smiles(json_path: Path) -> list[str]:
    payload = json.loads(json_path.read_text())
    if isinstance(payload, list):
        smiles = payload
    elif isinstance(payload, dict) and isinstance(payload.get("smiles"), list):
        smiles = payload["smiles"]
    else:
        raise ValueError(
            f"Expected {json_path} to contain either a JSON list of SMILES or "
            f'a JSON object with a "smiles" list.'
        )

    if not smiles:
        raise ValueError(f"No SMILES found in {json_path}")
    if not all(isinstance(item, str) for item in smiles):
        raise ValueError(f"All entries in {json_path} must be SMILES strings")
    return smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated molecules from a JSON file."
    )
    parser.add_argument(
        "molecules_json",
        type=Path,
        help="Path to a JSON file containing either a top-level smiles list or a dict with a smiles field.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <input_stem>_eval.json next to the input file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="MolGen/ZINC_1B-raw",
        help="Dataset repo used for novelty and cached validation/test stats.",
    )
    parser.add_argument(
        "--stats-subset",
        default="175k",
        help="Subset tag for valid/test stats, e.g. 175k.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size used by the evaluator.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of worker processes for evaluation.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for fragment metrics, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    molecules_json = args.molecules_json.resolve()
    if not molecules_json.is_file():
        raise FileNotFoundError(f"Molecules JSON not found: {molecules_json}")

    smiles = _load_smiles(molecules_json)
    test_stats_raw, valid_stats_raw = load_valid_and_test_data(
        dataset_name=args.dataset_name,
        subset=args.stats_subset,
    )
    evaluator = MoleculeEvaluator(
        task_names=TASK_NAMES,
        valid_stats=_normalize_stats(valid_stats_raw),
        test_stats=_normalize_stats(test_stats_raw),
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        device=args.device,
    )
    results = evaluator(smiles, filter=True)

    output_path = args.output
    if output_path is None:
        output_path = molecules_json.with_name(f"{molecules_json.stem}_eval.json")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "input_json": str(molecules_json),
        "dataset_name": args.dataset_name,
        "stats_subset": args.stats_subset,
        "tasks_requested": [
            "valid",
            "internal diversity",
            "novelty",
            "fcd",
            "snn",
            "fragment dist",
            "scaffold similarity",
        ],
        "task_names": TASK_NAMES,
        "num_input_smiles": len(smiles),
        "results": results,
    }
    output_json = json.dumps(payload, indent=2, sort_keys=True)
    output_path.write_text(output_json)

    print(output_json)
    print(f"Wrote evaluation results to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
