from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from joblib import Parallel, delayed


FitT = TypeVar("FitT")


def _fit_seed_batch(fit_seed: Callable[[int], FitT], seeds: list[int]) -> list[FitT]:
    return [fit_seed(seed) for seed in seeds]


def run_multistart(
    fit_seed: Callable[[int], FitT],
    seeds: Iterable[int],
    n_jobs: int = 1,
) -> tuple[FitT, list[dict[str, Any]]]:
    """Run fixed-seed fits in parallel and select the maximum-likelihood fit.

    Results are sorted by seed before selection so scheduling cannot change the
    winning fit or the restart diagnostics.
    """
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("At least one restart seed is required")
    worker_count = min(max(int(n_jobs), 1), len(seed_list))
    if worker_count == 1:
        fits = _fit_seed_batch(fit_seed, seed_list)
    else:
        seed_batches = [seed_list[offset::worker_count] for offset in range(worker_count)]
        batches = Parallel(
            n_jobs=worker_count,
            backend="loky",
            max_nbytes="1M",
            mmap_mode="r",
            pre_dispatch=worker_count,
        )(
            delayed(_fit_seed_batch)(fit_seed, batch) for batch in seed_batches
        )
        fits = [fit for batch in batches for fit in batch]
    fits = sorted(fits, key=lambda fit: int(getattr(fit, "seed")))
    best = max(fits, key=lambda fit: (float(getattr(fit, "train_loglik")), -int(getattr(fit, "seed"))))
    best_ll = float(getattr(best, "train_loglik"))
    rows: list[dict[str, Any]] = []
    for fit in fits:
        ll = float(getattr(fit, "train_loglik"))
        rows.append({
            "seed": int(getattr(fit, "seed")),
            "train_ll": ll,
            "delta_from_best": ll - best_ll,
            "converged": bool(getattr(fit, "converged")),
            "n_iter": getattr(fit, "n_iter", None),
            "selected": int(getattr(fit, "seed")) == int(getattr(best, "seed")),
            "message": getattr(fit, "result_message", ""),
        })
    return best, rows
