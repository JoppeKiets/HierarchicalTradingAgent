#!/usr/bin/env python3
"""Quick data quality check for preprocessed feature cache.

Run this before training to verify:
  1. Feature cache exists and is complete
  2. Daily/minute feature dimensions are consistent
  3. Target distributions are reasonable
  4. Date ranges cover expected period
  5. No excessive NaN/zero rows

Usage:
    python scripts/data_quality_check_v2.py
    python scripts/data_quality_check_v2.py --cache-dir data/feature_cache --verbose
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def check_cache(cache_dir: str, verbose: bool = False):
    """Run all data quality checks."""
    cache = Path(cache_dir)

    issues = []
    warnings = []

    # ─── Check daily cache ────────────────────────────────────────
    daily_dir = cache / "daily"
    if not daily_dir.exists():
        issues.append("❌ No daily feature cache found")
        print("CRITICAL: No daily cache. Run preprocessing first.")
        return

    daily_feat_files = sorted(daily_dir.glob("*_features.npy"))
    daily_tgt_files = sorted(daily_dir.glob("*_targets.npy"))
    daily_date_files = sorted(daily_dir.glob("*_dates.npy"))

    daily_tickers = set(f.stem.replace("_features", "") for f in daily_feat_files)
    daily_tgt_tickers = set(f.stem.replace("_targets", "") for f in daily_tgt_files)
    daily_date_tickers = set(f.stem.replace("_dates", "") for f in daily_date_files)

    print(f"\n{'='*60}")
    print(f"  DATA QUALITY CHECK — {cache_dir}")
    print(f"{'='*60}")

    # --- Daily ---
    print(f"\n  📊 Daily Cache")
    print(f"    Feature files: {len(daily_feat_files)}")
    print(f"    Target files:  {len(daily_tgt_files)}")
    print(f"    Date files:    {len(daily_date_files)}")

    # Check completeness
    missing_tgt = daily_tickers - daily_tgt_tickers
    missing_dates = daily_tickers - daily_date_tickers
    if missing_tgt:
        issues.append(f"❌ {len(missing_tgt)} daily tickers missing target files")
    if missing_dates:
        issues.append(f"❌ {len(missing_dates)} daily tickers missing date files")

    # Sample feature dimensions
    feat_dims = Counter()
    daily_rows = []
    for f in daily_feat_files[:200]:  # Sample 200
        arr = np.load(f, mmap_mode="r")
        feat_dims[arr.shape[1]] += 1
        daily_rows.append(arr.shape[0])

    if len(feat_dims) > 1:
        issues.append(f"❌ Inconsistent daily feature dims: {dict(feat_dims)}")
    else:
        dim = list(feat_dims.keys())[0]
        print(f"    Feature dim:   {dim}")

    if daily_rows:
        print(f"    Rows per ticker: min={min(daily_rows)}, "
              f"median={int(np.median(daily_rows))}, max={max(daily_rows)}")

    # Check target distribution (sample)
    all_tgts = []
    nan_count = 0
    for f in daily_tgt_files[:200]:
        tgt = np.load(f, mmap_mode="r")
        valid = tgt[~np.isnan(tgt)]
        all_tgts.extend(valid.tolist())
        nan_count += np.sum(np.isnan(tgt))

    if all_tgts:
        tgt_arr = np.array(all_tgts)
        print(f"    Target stats:  mean={np.mean(tgt_arr):.6f}, std={np.std(tgt_arr):.6f}")
        print(f"    Target range:  [{np.min(tgt_arr):.4f}, {np.max(tgt_arr):.4f}]")
        print(f"    NaN targets:   {nan_count:,} (expected for last {5} rows per ticker)")

        if np.std(tgt_arr) < 1e-6:
            issues.append("❌ Target std ≈ 0 (all same value)")
        if np.abs(np.mean(tgt_arr)) > 0.05:
            warnings.append(f"⚠️  Target mean is large: {np.mean(tgt_arr):.6f}")

    # Check date ranges
    import datetime
    min_date, max_date = float("inf"), 0
    for f in daily_date_files[:200]:
        dates = np.load(f, mmap_mode="r")
        nonzero = dates[dates > 0]
        if len(nonzero) > 0:
            min_date = min(min_date, nonzero.min())
            max_date = max(max_date, nonzero.max())

    if max_date > 0:
        try:
            d_min = datetime.date.fromordinal(int(min_date))
            d_max = datetime.date.fromordinal(int(max_date))
            span_years = (max_date - min_date) / 365.25
            print(f"    Date range:    {d_min} → {d_max} ({span_years:.1f} years)")

            if span_years < 1:
                warnings.append(f"⚠️  Short daily history: {span_years:.1f} years")
        except Exception:
            pass

    # --- Minute ---
    minute_dir = cache / "minute"
    if minute_dir.exists():
        minute_feat_files = sorted(minute_dir.glob("*_features.npy"))
        print(f"\n  ⏱️  Minute Cache")
        print(f"    Feature files: {len(minute_feat_files)}")

        if minute_feat_files:
            m_dims = Counter()
            m_rows = []
            for f in minute_feat_files[:100]:
                arr = np.load(f, mmap_mode="r")
                m_dims[arr.shape[1]] += 1
                m_rows.append(arr.shape[0])

            if len(m_dims) > 1:
                issues.append(f"❌ Inconsistent minute feature dims: {dict(m_dims)}")
            else:
                print(f"    Feature dim:   {list(m_dims.keys())[0]}")

            if m_rows:
                print(f"    Rows per ticker: min={min(m_rows)}, "
                      f"median={int(np.median(m_rows))}, max={max(m_rows)}")

            # Minute date range
            min_m_date, max_m_date = float("inf"), 0
            for f in sorted(minute_dir.glob("*_dates.npy"))[:100]:
                dates = np.load(f, mmap_mode="r")
                nonzero = dates[dates > 0]
                if len(nonzero) > 0:
                    min_m_date = min(min_m_date, nonzero.min())
                    max_m_date = max(max_m_date, nonzero.max())

            if max_m_date > 0:
                try:
                    d_min = datetime.date.fromordinal(int(min_m_date))
                    d_max = datetime.date.fromordinal(int(max_m_date))
                    print(f"    Date range:    {d_min} → {d_max}")

                    # Check daily/minute overlap
                    if max_date > 0 and min_m_date > max_date:
                        gap_days = int(min_m_date - max_date)
                        warnings.append(
                            f"⚠️  Daily/minute gap: daily ends "
                            f"{datetime.date.fromordinal(int(max_date))}, "
                            f"minute starts {d_min} ({gap_days} day gap). "
                            f"Meta model will use daily-only for most samples."
                        )
                except Exception:
                    pass
    else:
        warnings.append("⚠️  No minute feature cache found")

    # --- Check feature NaN/zero ratio (sample) ---
    print(f"\n  🔍 Feature Quality (sampled)")
    zero_ratios = []
    for f in daily_feat_files[:50]:
        arr = np.load(f, mmap_mode="r")
        zero_ratio = float(np.mean(arr == 0))
        zero_ratios.append(zero_ratio)

    if zero_ratios:
        avg_zero = np.mean(zero_ratios)
        print(f"    Avg zero ratio: {avg_zero:.2%}")
        if avg_zero > 0.5:
            warnings.append(f"⚠️  High zero ratio in features: {avg_zero:.2%}")

    # --- Summary ---
    print(f"\n{'='*60}")
    if issues:
        print("  ❌ ISSUES FOUND:")
        for i in issues:
            print(f"    {i}")
    if warnings:
        print("  ⚠️  WARNINGS:")
        for w in warnings:
            print(f"    {w}")
    if not issues and not warnings:
        print("  ✅ All checks passed!")
    elif not issues:
        print("  ✅ No critical issues (warnings above)")
    print(f"{'='*60}\n")

    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Data quality check for feature cache")
    parser.add_argument("--cache-dir", type=str, default="data/feature_cache")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    ok = check_cache(args.cache_dir, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
