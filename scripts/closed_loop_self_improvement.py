#!/usr/bin/env python3
"""Autonomous closed-loop self-improvement pipeline.

Full cycle (no human-in-the-loop):
  1) Train (optional bootstrap)
  2) Deploy agents (run swing pipeline)
  3) Collect outcomes over N-day horizon from trade journal
  4) Generate Critic training report (weak regimes, unreliable tickers, feature drift)
  5) Auto-trigger selective retraining when weak spots are detected
  6) Redeploy with the newly trained model

This script is intentionally orchestration-focused. It reuses existing project
entrypoints (`train_hierarchical.py`, `run_swing_pipeline.py`) and adds outcome
processing + report generation glue logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    model: Optional[str]
    top_n: int
    min_agreement: float
    min_return: float
    max_position: float
    limit: int
    iterations: int
    sleep_hours: float
    outcome_horizon_days: int
    min_samples_for_retrain: int
    min_regime_win_rate: float
    retrain_trigger_loss: float
    n_regimes: int
    output_root: str
    bootstrap_train: bool
    train_use_v10: bool
    train_use_tcn_d: bool
    train_use_gnn: bool
    train_use_fund_mlp: bool
    enable_feature_engineering: bool
    feature_min_ic_improvement: float
    feature_max_candidates: int
    feature_max_tickers_sample: int
    feature_max_rows_per_ticker: int


def run_cmd(cmd: List[str], cwd: Path = PROJECT_ROOT) -> None:
    logger.info("$ %s", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def find_latest_model() -> str:
    candidates = sorted((PROJECT_ROOT / "models").glob("**/forecaster_final.pt"))
    if not candidates:
        raise FileNotFoundError("No forecaster_final.pt found under models/")
    return str(candidates[-1])


def _read_price_history(organized_dir: Path, ticker: str) -> Optional[pd.DataFrame]:
    path = organized_dir / ticker / "price_history.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("datetime")
    close_col = cols.get("close")
    if date_col is None or close_col is None:
        return None
    df = df[[date_col, close_col]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df if len(df) > 0 else None


def _resolve_exit(
    df: pd.DataFrame,
    entry_timestamp: str,
    horizon_days: int,
) -> Optional[Tuple[pd.Timestamp, float]]:
    ts = pd.to_datetime(entry_timestamp, errors="coerce", utc=True)
    if pd.isna(ts):
        return None

    ts_naive = ts.tz_localize(None)
    idx = int(df["date"].searchsorted(ts_naive, side="left"))
    if idx >= len(df):
        return None
    exit_idx = idx + horizon_days
    if exit_idx >= len(df):
        return None
    row = df.iloc[exit_idx]
    return row["date"], float(row["close"])


def collect_outcomes(
    journal_path: Path,
    organized_dir: Path,
    horizon_days: int,
) -> Dict[str, int]:
    """Fill actual outcome fields in trade journal for matured entries.

    For each journal row without `actual_return`, compute time-exit return at
    N trading days after entry using `price_history.csv`.
    """
    if not journal_path.exists():
        return {"updated": 0, "skipped": 0, "total": 0}

    rows: List[Dict[str, Any]] = []
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    updated = 0
    skipped = 0
    cache: Dict[str, Optional[pd.DataFrame]] = {}

    for row in rows:
        if row.get("actual_return") is not None:
            continue

        ticker = str(row.get("ticker", ""))
        if not ticker:
            skipped += 1
            continue

        if ticker not in cache:
            cache[ticker] = _read_price_history(organized_dir, ticker)
        df = cache[ticker]
        if df is None:
            skipped += 1
            continue

        exit_info = _resolve_exit(df, str(row.get("timestamp", "")), horizon_days)
        if exit_info is None:
            # Not enough future candles yet
            continue

        exit_ts, exit_price = exit_info
        entry_price = float(row.get("entry_price") or 0.0)
        if entry_price <= 0:
            skipped += 1
            continue

        direction = str(row.get("direction", "long")).lower()
        if direction == "short":
            actual_return = (entry_price - exit_price) / entry_price
        else:
            actual_return = (exit_price - entry_price) / entry_price

        row["actual_return"] = float(actual_return)
        row["exit_price"] = float(exit_price)
        row["exit_timestamp"] = pd.Timestamp(exit_ts).isoformat()
        row["exit_reason"] = "time_exit"
        updated += 1

    if updated > 0:
        with open(journal_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

    return {"updated": updated, "skipped": skipped, "total": len(rows)}


def _compute_feature_drift(entries: List[Dict[str, Any]], z_threshold: float = 2.0) -> Dict[str, Any]:
    feature_rows: List[Dict[str, float]] = []
    for e in entries:
        rf = e.get("regime_features") or {}
        if isinstance(rf, dict) and rf:
            feature_rows.append({k: float(v) for k, v in rf.items() if isinstance(v, (int, float))})

    if len(feature_rows) < 20:
        return {"available": False, "drifted_features": [], "scores": {}}

    df = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if df.empty:
        return {"available": False, "drifted_features": [], "scores": {}}

    split = max(int(len(df) * 0.8), 1)
    base = df.iloc[:split]
    recent = df.iloc[split:]
    if len(recent) == 0:
        return {"available": False, "drifted_features": [], "scores": {}}

    base_mean = base.mean()
    base_std = base.std().replace(0, np.nan)
    recent_mean = recent.mean()
    z = ((recent_mean - base_mean).abs() / (base_std + 1e-8)).fillna(0.0)

    scores = {k: float(v) for k, v in z.to_dict().items()}
    drifted = [k for k, v in scores.items() if v >= z_threshold]
    return {
        "available": True,
        "drifted_features": sorted(drifted, key=lambda k: scores[k], reverse=True),
        "scores": scores,
    }


def build_training_report(
    journal_path: Path,
    report_dir: Path,
    min_regime_win_rate: float,
    retrain_trigger_loss: float,
    min_samples_for_retrain: int,
    feature_engineering: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    if journal_path.exists():
        with open(journal_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    resolved = [e for e in entries if e.get("actual_return") is not None]
    if not resolved:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_total_entries": len(entries),
            "n_resolved_entries": 0,
            "portfolio": {"avg_return": 0.0, "win_rate": 0.0},
            "weak_regimes": [],
            "unreliable_tickers": [],
            "feature_drift": {"available": False, "drifted_features": [], "scores": {}},
            "feature_engineering": feature_engineering or {},
            "trigger_retrain": False,
            "reason": "No resolved outcomes yet",
        }
    else:
        returns = np.array([float(e["actual_return"]) for e in resolved], dtype=np.float64)
        portfolio_avg = float(np.mean(returns))
        portfolio_win = float(np.mean(returns > 0))

        regime_stats: Dict[str, Dict[str, Any]] = {}
        ticker_stats: Dict[str, Dict[str, Any]] = {}

        for e in resolved:
            regime = str(e.get("regime_label", "unknown"))
            ticker = str(e.get("ticker", "UNKNOWN"))
            r = float(e.get("actual_return", 0.0))

            rs = regime_stats.setdefault(regime, {"n": 0, "wins": 0, "sum_ret": 0.0})
            rs["n"] += 1
            rs["wins"] += 1 if r > 0 else 0
            rs["sum_ret"] += r

            ts = ticker_stats.setdefault(ticker, {"n": 0, "wins": 0, "sum_ret": 0.0})
            ts["n"] += 1
            ts["wins"] += 1 if r > 0 else 0
            ts["sum_ret"] += r

        weak_regimes: List[Dict[str, Any]] = []
        for regime, s in regime_stats.items():
            n = int(s["n"])
            wr = float(s["wins"] / n) if n else 0.0
            avg = float(s["sum_ret"] / n) if n else 0.0
            if n >= min_samples_for_retrain and (wr < min_regime_win_rate or avg < 0):
                weak_regimes.append({"regime": regime, "n": n, "win_rate": wr, "avg_return": avg})

        unreliable_tickers: List[Dict[str, Any]] = []
        for ticker, s in ticker_stats.items():
            n = int(s["n"])
            wr = float(s["wins"] / n) if n else 0.0
            avg = float(s["sum_ret"] / n) if n else 0.0
            if n >= min_samples_for_retrain and (wr < 0.45 or avg < retrain_trigger_loss):
                unreliable_tickers.append({"ticker": ticker, "n": n, "win_rate": wr, "avg_return": avg})
        unreliable_tickers.sort(key=lambda x: (x["avg_return"], x["win_rate"]))

        feature_drift = _compute_feature_drift(resolved)

        trigger_retrain = bool(
            weak_regimes
            or unreliable_tickers
            or (portfolio_avg < retrain_trigger_loss and len(resolved) >= min_samples_for_retrain)
            or (feature_drift.get("available") and len(feature_drift.get("drifted_features", [])) >= 2)
            or ((feature_engineering or {}).get("promoted_count", 0) > 0)
        )

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_total_entries": len(entries),
            "n_resolved_entries": len(resolved),
            "portfolio": {"avg_return": portfolio_avg, "win_rate": portfolio_win},
            "weak_regimes": weak_regimes,
            "unreliable_tickers": unreliable_tickers,
            "feature_drift": feature_drift,
            "feature_engineering": feature_engineering or {},
            "trigger_retrain": trigger_retrain,
            "reason": "auto-analysis",
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = report_dir / f"training_report_{ts}.json"
    latest = report_dir / "latest_training_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    with open(latest, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Training report saved → %s", out)
    logger.info(
        "Report summary | resolved=%d avg_ret=%.5f win_rate=%.2f trigger_retrain=%s",
        report.get("n_resolved_entries", 0),
        report.get("portfolio", {}).get("avg_return", 0.0),
        report.get("portfolio", {}).get("win_rate", 0.0),
        report.get("trigger_retrain", False),
    )
    return report


def maybe_bootstrap_train(cfg: LoopConfig, python_exe: str) -> str:
    model_path = cfg.model
    if model_path and Path(model_path).exists():
        return model_path

    model_path = find_latest_model() if not cfg.bootstrap_train else ""
    if model_path:
        logger.info("Using existing model: %s", model_path)
        return model_path

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / f"bootstrap_{ts}"
    cmd = [
        python_exe,
        "train_hierarchical.py",
        "--phase", "1", "2", "3", "4",
        "--output-dir", str(out_dir),
        "--regime-curriculum",
        "--n-regimes", str(cfg.n_regimes),
    ]
    if cfg.train_use_v10:
        cmd.append("--v10")
    if cfg.train_use_tcn_d:
        cmd.append("--use-tcn-d")
    if cfg.train_use_gnn:
        cmd.append("--use-gnn")
    if cfg.train_use_fund_mlp:
        cmd.append("--use-fund-mlp")

    run_cmd(cmd)
    model_path = str(out_dir / "forecaster_final.pt")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Bootstrap training finished but model missing: {model_path}")
    return model_path


def run_deploy(cfg: LoopConfig, python_exe: str, model_path: str) -> None:
    cmd = [
        python_exe,
        "run_swing_pipeline.py",
        "--model", model_path,
        "--top-n", str(cfg.top_n),
        "--min-agreement", str(cfg.min_agreement),
        "--min-return", str(cfg.min_return),
        "--max-position", str(cfg.max_position),
        "--save",
    ]
    if cfg.limit > 0:
        cmd.extend(["--limit", str(cfg.limit)])
    run_cmd(cmd)


def selective_retrain(
    cfg: LoopConfig,
    python_exe: str,
    current_model: str,
    report: Dict[str, Any],
) -> Optional[str]:
    if not report.get("trigger_retrain", False):
        logger.info("No retrain trigger — skipping selective retraining")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / f"selective_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    promoted_count = int(report.get("feature_engineering", {}).get("promoted_count", 0))
    full_retrain_for_new_features = promoted_count > 0

    if full_retrain_for_new_features:
        cmd = [
            python_exe,
            "train_hierarchical.py",
            "--phase", "0", "1", "2", "3", "4",
            "--output-dir", str(out_dir),
            "--force-preprocess",
            "--regime-curriculum",
            "--n-regimes", str(cfg.n_regimes),
        ]
    else:
        cmd = [
            python_exe,
            "train_hierarchical.py",
            "--resume", current_model,
            "--phase", "3", "4",
            "--output-dir", str(out_dir),
            "--skip-trained",
            "--regime-curriculum",
            "--n-regimes", str(cfg.n_regimes),
        ]
    if cfg.train_use_v10:
        cmd.append("--v10")
    if cfg.train_use_tcn_d:
        cmd.append("--use-tcn-d")
    if cfg.train_use_gnn:
        cmd.append("--use-gnn")
    if cfg.train_use_fund_mlp:
        cmd.append("--use-fund-mlp")

    logger.info(
        "Selective retrain trigger active | weak_regimes=%d unreliable_tickers=%d drift_features=%d promoted_features=%d mode=%s",
        len(report.get("weak_regimes", [])),
        len(report.get("unreliable_tickers", [])),
        len(report.get("feature_drift", {}).get("drifted_features", [])),
        promoted_count,
        "full_retrain" if full_retrain_for_new_features else "phase3_4",
    )
    run_cmd(cmd)

    candidate = out_dir / "forecaster_final.pt"
    if not candidate.exists():
        logger.warning("Selective retrain finished but no forecaster_final.pt in %s", out_dir)
        return None
    return str(candidate)


def run_loop(cfg: LoopConfig) -> None:
    feature_module = __import__(
        "agents.feedback.auto_feature_engineer",
        fromlist=["AnalystAutoFeatureEngineer"],
    )
    AnalystAutoFeatureEngineer = getattr(feature_module, "AnalystAutoFeatureEngineer")

    python_exe = sys.executable
    journal_path = PROJECT_ROOT / "data" / "trade_journal" / "trade_journal.jsonl"
    organized_dir = PROJECT_ROOT / "data" / "organized"
    report_dir = PROJECT_ROOT / "data" / "closed_loop" / "reports"
    feature_engineer = AnalystAutoFeatureEngineer(
        cache_dir="data/feature_cache",
        feedback_dir="data/feature_feedback",
        generated_feature_code_path="src/features/generated_features.py",
    )

    model_path = maybe_bootstrap_train(cfg, python_exe)

    for i in range(cfg.iterations):
        logger.info("=" * 80)
        logger.info("Closed-loop iteration %d/%d", i + 1, cfg.iterations)
        logger.info("Active model: %s", model_path)
        logger.info("=" * 80)

        # 1) Deploy current model
        run_deploy(cfg, python_exe, model_path)

        # 2) Collect outcomes for matured trades
        outcome_stats = collect_outcomes(
            journal_path=journal_path,
            organized_dir=organized_dir,
            horizon_days=cfg.outcome_horizon_days,
        )
        logger.info("Outcome collector | %s", outcome_stats)

        # 3) Analyst as automated feature engineer (propose -> ablate -> promote)
        feature_summary: Dict[str, Any] = {}
        if cfg.enable_feature_engineering:
            feature_summary = feature_engineer.run_cycle(
                max_candidates=cfg.feature_max_candidates,
                max_tickers_sample=cfg.feature_max_tickers_sample,
                max_rows_per_ticker=cfg.feature_max_rows_per_ticker,
                min_ic_improvement=cfg.feature_min_ic_improvement,
            )
            logger.info(
                "Feature engineer | promoted=%d accepted_total=%d baseline_ic=%.4f",
                int(feature_summary.get("promoted_count", 0)),
                int(feature_summary.get("accepted_total", 0)),
                float(feature_summary.get("baseline_ic", 0.0)),
            )

        # 4) Critic training report
        report = build_training_report(
            journal_path=journal_path,
            report_dir=report_dir,
            min_regime_win_rate=cfg.min_regime_win_rate,
            retrain_trigger_loss=cfg.retrain_trigger_loss,
            min_samples_for_retrain=cfg.min_samples_for_retrain,
            feature_engineering=feature_summary,
        )

        # 5) Auto-trigger selective retraining
        new_model = selective_retrain(cfg, python_exe, model_path, report)
        if new_model:
            model_path = new_model
            logger.info("New model deployed: %s", model_path)
            # 6) Redeploy immediately after retrain
            run_deploy(cfg, python_exe, model_path)

        if i < cfg.iterations - 1 and cfg.sleep_hours > 0:
            sleep_seconds = int(cfg.sleep_hours * 3600)
            wake = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
            logger.info("Sleeping %d seconds until next cycle (%s UTC)", sleep_seconds, wake.isoformat())
            time.sleep(sleep_seconds)


def parse_args() -> LoopConfig:
    parser = argparse.ArgumentParser(description="Autonomous closed-loop self-improvement")
    parser.add_argument("--model", type=str, default=None,
                        help="Starting model path. If omitted, uses latest forecaster_final.pt")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-agreement", type=float, default=0.50)
    parser.add_argument("--min-return", type=float, default=0.001)
    parser.add_argument("--max-position", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--iterations", type=int, default=1,
                        help="How many deploy→collect→report→retrain cycles to run")
    parser.add_argument("--sleep-hours", type=float, default=0.0,
                        help="Sleep duration between cycles (for long-running daemon mode)")
    parser.add_argument("--outcome-horizon-days", type=int, default=5,
                        help="Time-exit horizon for auto-populating outcomes in journal")

    parser.add_argument("--min-samples-for-retrain", type=int, default=10)
    parser.add_argument("--min-regime-win-rate", type=float, default=0.48)
    parser.add_argument("--retrain-trigger-loss", type=float, default=-0.001,
                        help="Trigger retrain if avg return drops below this threshold")
    parser.add_argument("--n-regimes", type=int, default=6)

    parser.add_argument("--output-root", type=str, default="models/closed_loop")
    parser.add_argument("--bootstrap-train", action="store_true",
                        help="If no model is found, run full bootstrap training automatically")

    parser.add_argument("--train-use-v10", action="store_true", default=True)
    parser.add_argument("--no-train-use-v10", action="store_false", dest="train_use_v10")
    parser.add_argument("--train-use-tcn-d", action="store_true", default=True)
    parser.add_argument("--no-train-use-tcn-d", action="store_false", dest="train_use_tcn_d")
    parser.add_argument("--train-use-gnn", action="store_true", default=True)
    parser.add_argument("--no-train-use-gnn", action="store_false", dest="train_use_gnn")
    parser.add_argument("--train-use-fund-mlp", action="store_true", default=True)
    parser.add_argument("--no-train-use-fund-mlp", action="store_false", dest="train_use_fund_mlp")

    parser.add_argument("--enable-feature-engineering", action="store_true", default=True,
                        help="Enable Analyst auto feature proposal + ablation + promotion stage")
    parser.add_argument("--no-enable-feature-engineering", action="store_false", dest="enable_feature_engineering")
    parser.add_argument("--feature-min-ic-improvement", type=float, default=0.001,
                        help="Minimum IC gain required to promote a generated feature")
    parser.add_argument("--feature-max-candidates", type=int, default=5)
    parser.add_argument("--feature-max-tickers-sample", type=int, default=200)
    parser.add_argument("--feature-max-rows-per-ticker", type=int, default=250)

    args = parser.parse_args()
    return LoopConfig(
        model=args.model,
        top_n=args.top_n,
        min_agreement=args.min_agreement,
        min_return=args.min_return,
        max_position=args.max_position,
        limit=args.limit,
        iterations=args.iterations,
        sleep_hours=args.sleep_hours,
        outcome_horizon_days=args.outcome_horizon_days,
        min_samples_for_retrain=args.min_samples_for_retrain,
        min_regime_win_rate=args.min_regime_win_rate,
        retrain_trigger_loss=args.retrain_trigger_loss,
        n_regimes=args.n_regimes,
        output_root=args.output_root,
        bootstrap_train=args.bootstrap_train,
        train_use_v10=args.train_use_v10,
        train_use_tcn_d=args.train_use_tcn_d,
        train_use_gnn=args.train_use_gnn,
        train_use_fund_mlp=args.train_use_fund_mlp,
        enable_feature_engineering=args.enable_feature_engineering,
        feature_min_ic_improvement=args.feature_min_ic_improvement,
        feature_max_candidates=args.feature_max_candidates,
        feature_max_tickers_sample=args.feature_max_tickers_sample,
        feature_max_rows_per_ticker=args.feature_max_rows_per_ticker,
    )


def main() -> None:
    cfg = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger.info("Starting autonomous closed-loop with config: %s", cfg)
    run_loop(cfg)


if __name__ == "__main__":
    main()
