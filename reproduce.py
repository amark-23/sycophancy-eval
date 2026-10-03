"""One-command reproduce path: python reproduce.py [--config configs/default.yaml]

Stub for now. Each stage will be added as the experiments are built:
  1. build datasets   (data/)
  2. run evals        (tasks/)  -> cached logs in results/logs/
  3. analyze          (analysis/) -> tables and figures in results/
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce all results.")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    print(f"[reproduce] config={args.config}")
    print("[reproduce] no stages implemented yet")


if __name__ == "__main__":
    main()
