"""Evaluation harness for the "Analytical Language of John Wilkins" hometask.

Importable from a notebook OR runnable from the command line. The Interactor
owns the LLM judge internally — this module just iterates rows and tallies
scores.

  # Notebook usage:
  from evaluate import evaluate, load_pools
  results = evaluate(MySolution(animals, questions), "dataset/dev.csv")
  print(results['mean_score'])
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from interactor import Interactor, QUERY_BUDGET, QUERY_COST

# All data files live in the same folder as this module (the "dataset" folder).
HERE = Path(__file__).resolve().parent
DEFAULT_ANIMALS_POOL   = HERE / "animals_pool.txt"
DEFAULT_QUESTIONS_POOL = HERE / "questions_pool.txt"
DEFAULT_DEV            = HERE / "dev.csv"


def load_pools(animals_pool_path: Path | str = DEFAULT_ANIMALS_POOL,
               questions_pool_path: Path | str = DEFAULT_QUESTIONS_POOL
               ) -> tuple[list[str], list[str]]:
    animals   = [l.strip().lower() for l in Path(animals_pool_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [l.strip().lower() for l in Path(questions_pool_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return animals, questions


def evaluate(solution, csv_path: Path | str,
             animals_pool_path: Path | str = DEFAULT_ANIMALS_POOL,
             questions_pool_path: Path | str = DEFAULT_QUESTIONS_POOL,
             budget: int = QUERY_BUDGET, cost: float = QUERY_COST,
             verbose: bool = True) -> dict:
    """Run a solution against a labeled CSV of gold animals.

    `solution` is either an object with a .solve(interactor) method, or
    a callable solution(interactor) -> None.

    Returns mean_score, solved_rate, mean_queries, and a per-row DataFrame.
    """
    animals, questions = load_pools(animals_pool_path, questions_pool_path)
    animals_set   = set(animals)
    questions_set = set(questions)

    df = pd.read_csv(csv_path)
    if "animal" not in df.columns:
        raise ValueError(f"{csv_path} missing required 'animal' column")
    gold_animals = df.animal.astype(str).str.strip().str.lower().tolist()

    solve_fn = solution.solve if hasattr(solution, "solve") else solution

    per_row = []
    t0 = time.time()
    for i, gold in enumerate(gold_animals):
        interactor = Interactor(
            gold_animal=gold,
            animals_pool=animals_set,
            questions_pool=questions_set,
            budget=budget, cost=cost,
        )
        try:
            solve_fn(interactor)
        except Exception as e:
            if verbose:
                print(f"  [warn] row {i} (gold={gold!r}) raised: {type(e).__name__}: {e}")
        per_row.append({
            "gold": gold,
            "queries_used": interactor.queries_used,
            "solved": int(interactor.solved),
            "score": interactor.score(),
        })
        if verbose and (i + 1) % 25 == 0:
            mean_now = np.mean([r["score"] for r in per_row])
            print(f"  {i+1}/{len(gold_animals)} rows  mean_score={mean_now:.4f}  "
                  f"({time.time() - t0:.1f}s)")

    out_df = pd.DataFrame(per_row)
    results = {
        "n":            len(out_df),
        "mean_score":   float(out_df.score.mean()),
        "solved_rate":  float(out_df.solved.mean()),
        "mean_queries": float(out_df.queries_used.mean()),
        "per_row":      out_df,
    }
    if verbose:
        print()
        print(f"  Dataset:       {csv_path}")
        print(f"  Mean score:    {results['mean_score']:.4f}")
        print(f"  Solved rate:   {results['solved_rate']*100:.1f}%")
        print(f"  Mean queries:  {results['mean_queries']:.2f} / {budget}")
        print(f"  Wall time:     {time.time() - t0:.1f}s")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_DEV))
    p.add_argument("--solution", required=True,
                   help='module:Class or path/to/file.py:Class')
    p.add_argument("--budget", type=int, default=QUERY_BUDGET)
    args = p.parse_args()

    import importlib, importlib.util, sys
    if ":" in args.solution:
        mod_part, cls_name = args.solution.split(":", 1)
    else:
        mod_part, cls_name = args.solution, None
    if mod_part.endswith(".py"):
        path = Path(mod_part).resolve()
        loader = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(loader)
        sys.modules[path.stem] = module
        loader.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)
    SolutionCls = getattr(module, cls_name) if cls_name else module
    animals, questions = load_pools()
    sol = SolutionCls(animals, questions) if cls_name else SolutionCls
    evaluate(sol, args.csv, budget=args.budget)


if __name__ == "__main__":
    main()
