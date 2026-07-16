#!/usr/bin/env python3
"""Generate the ablation configurations described in Section 4.6."""

import argparse
import copy
import json
from pathlib import Path


K_VALUES = (1, 3, 5, 7, 10, 20)
SAMPLING_STEPS = (100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000)
NOISE_SCHEDULES = {
    "linear": "linear",
    "scaled_linear": "scaled_linear",
    "cosine": "squaredcos_improved_ddpm",
    "sigmoid": "sigmoid",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(output_dir, name, config):
    path = output_dir / f"{name}.json"
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-base",
        type=Path,
        default=Path("configs/diffupercom_train.json"),
    )
    parser.add_argument(
        "--eval-base",
        type=Path,
        default=Path("configs/diffupercom_eval.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/generated_ablations"),
    )
    args = parser.parse_args()

    train_base = read_json(args.train_base)
    eval_base = read_json(args.eval_base)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    without_self_conditioning = copy.deepcopy(train_base)
    without_self_conditioning["self_condition"] = None
    without_self_conditioning["output_dir"] = "outputs/ablations/self_conditioning_off"
    written.append(
        write_config(
            args.output_dir,
            "self_conditioning_off",
            without_self_conditioning,
        )
    )

    for value in K_VALUES:
        config = copy.deepcopy(train_base)
        config["simplex_value"] = float(value)
        config["output_dir"] = f"outputs/ablations/k_{value}"
        written.append(write_config(args.output_dir, f"k_{value}", config))

    for name, value in NOISE_SCHEDULES.items():
        config = copy.deepcopy(train_base)
        config["beta_schedule"] = value
        config["output_dir"] = f"outputs/ablations/noise_{name}"
        written.append(write_config(args.output_dir, f"noise_{name}", config))

    for steps in SAMPLING_STEPS:
        config = copy.deepcopy(eval_base)
        config["num_inference_diffusion_steps"] = steps
        config["output_dir"] = f"outputs/ablations/sampling_steps_{steps}"
        written.append(write_config(args.output_dir, f"sampling_steps_{steps}", config))

    print(f"Wrote {len(written)} configurations to {args.output_dir}")


if __name__ == "__main__":
    main()
