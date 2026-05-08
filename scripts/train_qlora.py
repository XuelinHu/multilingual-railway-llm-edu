from __future__ import annotations

import argparse

from railway_llm_edu.training.qlora import load_config, train_qlora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qlora.yaml")
    args = parser.parse_args()
    train_qlora(load_config(args.config))


if __name__ == "__main__":
    main()
