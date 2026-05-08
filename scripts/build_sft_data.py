from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path("data/sft")
    print(f"SFT data builder placeholder. Suggested output directory: {target}")
    print("Next step: convert regulation QA, term QA, bilingual QA, and risk-aware QA into instruction tuning samples.")


if __name__ == "__main__":
    main()

