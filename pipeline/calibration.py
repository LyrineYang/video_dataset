from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import ScoreResult


def write_calibration_parquet(results: Iterable[ScoreResult], path: Path) -> None:
    rows = []
    for res in results:
        row = {
            "source_path": str(res.path),
            "scores": res.scores,
            "keep": res.keep,
            "reason": res.reason,
        }
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def compute_quantiles(parquet_path: Path, quantiles: list[float]) -> dict[str, dict[float, float]]:
    df = pd.read_parquet(parquet_path)
    # scores column is dict; expand
    scores_df = df["scores"].apply(pd.Series)
    out: dict[str, dict[float, float]] = {}
    for col in scores_df.columns:
        series = scores_df[col].dropna()
        if series.empty:
            continue
        qs = {q: float(series.quantile(q)) for q in quantiles}
        out[col] = qs
    return out
