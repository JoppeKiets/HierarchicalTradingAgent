#!/usr/bin/env python3
"""Align per-article FinBERT embeddings to minute-level bars (intraday alignment).

Strategy
--------
For each minute bar (with a real UTC timestamp), we look backward up to
``max_lookback_days`` trading days and collect all articles published *before*
that bar's timestamp.

From those candidates we rank by **impact score**:

    impact = |compound_sentiment| * log1p(article_length)

where ``compound_sentiment = pos - neg`` from FinBERT's softmax output.
We take the top ``top_k`` articles (default 15).  If fewer than ``min_articles``
(default 5) are found in the window, we extend the lookback to include all
available articles before that bar (no hard cutoff — better to use old news
than to have a blank signal).

Result per bar: mean-pool the selected embeddings (768-dim) and build a 6-dim
sentiment summary → (774,) total, aligned to every minute bar.

Output (per ticker):
    data/feature_cache/minute_news/{TICKER}_news_features.npy   (N_bars, 774)
    data/feature_cache/minute_news/{TICKER}_news_timestamps.npy (N_bars,) unix sec

The 774-dim layout:
    [0:768]  mean-pooled FinBERT embedding
    [768]    pos_mean
    [769]    neg_mean
    [770]    neu_mean
    [771]    compound_mean  (pos - neg)
    [772]    compound_std
    [773]    article_count  (raw, not normalised)

Usage:
    python scripts/align_news_to_minute.py                    # All tickers
    python scripts/align_news_to_minute.py --tickers AAPL MSFT
    python scripts/align_news_to_minute.py --force            # Recompute cached
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Conservative proxy: 1 trading day ≈ 1 calendar day for lookback arithmetic
_SECONDS_PER_TRADING_DAY = 24 * 3600


def _impact_score(compound: np.ndarray, art_len: np.ndarray) -> np.ndarray:
    """Per-article impact = |compound_sentiment| * log1p(article_length)."""
    return np.abs(compound) * np.log1p(art_len)


def align_news_to_minute_ticker(
    ticker: str,
    article_cache_dir: str = "data/feature_cache/news_articles",
    minute_history_dir: str = "data/minute_history",
    output_dir: str = "data/feature_cache/minute_news",
    min_articles: int = 5,
    top_k: int = 15,
    max_lookback_days: int = 10,
    force: bool = False,
) -> Optional[Dict]:
    """Align per-article news embeddings to minute bars for one ticker.

    Parameters
    ----------
    min_articles:
        Desired minimum articles per bar.  If the primary window (max_lookback_days)
        yields fewer, we extend back to all available articles before that bar.
    top_k:
        Maximum articles to pool per bar, ranked by impact score descending.
    max_lookback_days:
        Primary lookback window in trading days (1 trad.day ≈ 1 calendar day).

    Returns
    -------
    Metadata dict, or None if the ticker has no minute data.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_dir / f"{ticker}_news_features.npy"
    ts_path   = out_dir / f"{ticker}_news_timestamps.npy"

    if not force and feat_path.exists() and ts_path.exists():
        feat = np.load(feat_path, mmap_mode="r")
        return {"ticker": ticker, "n_bars": feat.shape[0], "status": "cached"}

    # ------------------------------------------------------------------
    # Load per-article cache (produced by preprocess_news_embeddings.py)
    # ------------------------------------------------------------------
    art_dir = Path(article_cache_dir)
    art_emb_path  = art_dir / f"{ticker}_embeddings.npy"
    art_sent_path = art_dir / f"{ticker}_sentiments.npy"
    art_ts_path   = art_dir / f"{ticker}_timestamps.npy"
    art_len_path  = art_dir / f"{ticker}_art_lens.npy"

    has_articles = all(p.exists() for p in [
        art_emb_path, art_sent_path, art_ts_path, art_len_path
    ])

    # Initialise with empty arrays so the names are always bound
    art_embs     = np.empty((0, 768), dtype=np.float32)
    art_sents    = np.empty((0, 3),   dtype=np.float32)
    art_ts_unix  = np.empty((0,),     dtype=np.int64)
    art_impact   = np.empty((0,),     dtype=np.float32)
    art_compound = np.empty((0,),     dtype=np.float32)
    n_articles   = 0

    if has_articles:
        _embs    = np.load(art_emb_path)                       # (A, 768)
        _sents   = np.load(art_sent_path)                      # (A, 3) [pos, neg, neu]
        _ts_unix = np.load(art_ts_path).astype(np.int64)       # (A,) unix sec
        _lens    = np.load(art_len_path).astype(np.float32)    # (A,)

        if len(_ts_unix) == 0:
            has_articles = False
        else:
            # Pre-compute compound and impact for all articles (vectorised)
            _compound = _sents[:, 0] - _sents[:, 1]            # (A,)
            _impact   = _impact_score(_compound, _lens)         # (A,)

            # Sort by timestamp ascending for O(log N) binary-search alignment
            sort_idx     = np.argsort(_ts_unix)
            art_embs     = _embs[sort_idx]
            art_sents    = _sents[sort_idx]
            art_ts_unix  = _ts_unix[sort_idx]
            art_impact   = _impact[sort_idx]
            art_compound = _compound[sort_idx]
            n_articles   = len(art_ts_unix)

    # ------------------------------------------------------------------
    # Load minute bars
    # ------------------------------------------------------------------
    minute_file = Path(minute_history_dir) / f"{ticker}.parquet"
    if not minute_file.exists():
        return None

    try:
        mdf = pd.read_parquet(minute_file)
    except Exception as e:
        logger.warning(f"  {ticker}: failed to load parquet — {e}")
        return None

    if len(mdf) == 0 or "timestamp" not in mdf.columns:
        return None

    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], utc=True)
    bar_ts_unix = (mdf["timestamp"].astype("int64") // 10**9).values  # ns → sec
    n_bars = len(bar_ts_unix)

    # ------------------------------------------------------------------
    # Build output arrays
    # ------------------------------------------------------------------
    EMBED_DIM  = 768
    SENT_DIM   = 6    # pos_mean, neg_mean, neu_mean, compound_mean, compound_std, count
    TOTAL_DIM  = EMBED_DIM + SENT_DIM  # 774

    out_features = np.zeros((n_bars, TOTAL_DIM), dtype=np.float32)
    out_ts       = np.asarray(bar_ts_unix, dtype=np.int64)

    if not has_articles:
        # No per-article cache → write zero-filled output
        np.save(feat_path, out_features)
        np.save(ts_path,   out_ts)
        return {
            "ticker": ticker, "n_bars": n_bars,
            "n_bars_with_news": 0, "status": "computed_no_news",
        }

    max_lookback_sec = max_lookback_days * _SECONDS_PER_TRADING_DAY
    n_bars_with_news = 0

    # Sliding left pointer — advances so that art_ts_unix[left_ptr] is always
    # within max_lookback_sec of the current bar.  This avoids a full O(A)
    # scan per bar, giving O((A + N) log A) overall.
    left_ptr = 0

    for bar_idx in range(n_bars):
        bar_t        = int(bar_ts_unix[bar_idx])
        window_start = bar_t - max_lookback_sec

        # Advance the left pointer past articles older than the primary window
        while left_ptr < n_articles and art_ts_unix[left_ptr] < window_start:
            left_ptr += 1

        # right_ptr = first article index NOT strictly before bar_t (no look-ahead)
        right_ptr = int(np.searchsorted(art_ts_unix, bar_t, side="left"))

        n_candidates = right_ptr - left_ptr

        if n_candidates < min_articles and left_ptr > 0:
            # Extend lookback: use ALL articles published before this bar
            effective_left = 0
            n_candidates   = right_ptr
        else:
            effective_left = left_ptr

        if n_candidates == 0:
            continue  # No articles at all before this bar

        sel_embs     = art_embs[effective_left:right_ptr]
        sel_sents    = art_sents[effective_left:right_ptr]
        sel_impact   = art_impact[effective_left:right_ptr]
        sel_compound = art_compound[effective_left:right_ptr]

        # Rank and take top-k if needed
        if n_candidates > top_k:
            top_idx      = np.argpartition(sel_impact, -top_k)[-top_k:]
            sel_embs     = sel_embs[top_idx]
            sel_sents    = sel_sents[top_idx]
            sel_compound = sel_compound[top_idx]

        # Mean-pool embedding
        mean_emb = sel_embs.mean(axis=0)  # (768,)

        # 6-dim sentiment summary
        pos_mean      = float(sel_sents[:, 0].mean())
        neg_mean      = float(sel_sents[:, 1].mean())
        neu_mean      = float(sel_sents[:, 2].mean())
        compound_mean = float(sel_compound.mean())
        compound_std  = float(sel_compound.std()) if len(sel_compound) > 1 else 0.0
        count         = float(len(sel_embs))

        out_features[bar_idx, :EMBED_DIM] = mean_emb
        out_features[bar_idx, EMBED_DIM:] = [
            pos_mean, neg_mean, neu_mean, compound_mean, compound_std, count,
        ]
        n_bars_with_news += 1

    np.save(feat_path, out_features)
    np.save(ts_path,   out_ts)

    coverage = n_bars_with_news / n_bars if n_bars > 0 else 0.0
    return {
        "ticker":           ticker,
        "n_bars":           n_bars,
        "n_bars_with_news": n_bars_with_news,
        "coverage":         f"{coverage:.1%}",
        "n_articles_total": n_articles,
        "status":           "computed",
    }


def run_alignment(
    article_cache_dir: str = "data/feature_cache/news_articles",
    minute_history_dir: str = "data/minute_history",
    output_dir: str = "data/feature_cache/minute_news",
    tickers: Optional[List[str]] = None,
    max_tickers: int = 0,
    min_articles: int = 5,
    top_k: int = 15,
    max_lookback_days: int = 10,
    force: bool = False,
) -> Dict:
    """Run intraday minute-bar news alignment for all (or specified) tickers."""

    if tickers is None:
        minute_path = Path(minute_history_dir)
        tickers = sorted(f.stem for f in minute_path.glob("*.parquet"))

    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    logger.info(
        f"Aligning {len(tickers)} tickers | "
        f"min_articles={min_articles}, top_k={top_k}, "
        f"max_lookback_days={max_lookback_days}"
    )

    # Coverage sanity check: warn if <20% of tickers have per-article news
    art_dir = Path(article_cache_dir)
    n_with_articles = sum(
        1 for t in tickers
        if (art_dir / f"{t}_embeddings.npy").exists()
        and np.load(art_dir / f"{t}_embeddings.npy", mmap_mode="r").shape[0] > 0
    )
    coverage_pct = 100.0 * n_with_articles / max(len(tickers), 1)
    if coverage_pct < 20.0:
        logger.warning(
            f"⚠️  Only {n_with_articles}/{len(tickers)} ({coverage_pct:.1f}%) tickers "
            f"have per-article news — most bars will be zero-filled.  "
            f"Run preprocess_news_embeddings.py first."
        )
    else:
        logger.info(
            f"  Per-article news coverage: {n_with_articles}/{len(tickers)} "
            f"({coverage_pct:.1f}%) ✓ (above 20% threshold)"
        )

    metadata: Dict = {}
    t0 = time.time()
    n_success = 0

    for i, ticker in enumerate(tickers):
        info = align_news_to_minute_ticker(
            ticker,
            article_cache_dir=article_cache_dir,
            minute_history_dir=minute_history_dir,
            output_dir=output_dir,
            min_articles=min_articles,
            top_k=top_k,
            max_lookback_days=max_lookback_days,
            force=force,
        )
        if info:
            metadata[ticker] = info
            n_success += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tickers) - i - 1) / rate
            logger.info(
                f"  Progress: {i+1}/{len(tickers)} ({n_success} ok) | "
                f"Elapsed {elapsed:.0f}s | ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    logger.info(f"\nComplete: {n_success}/{len(tickers)} tickers, {elapsed:.0f}s")

    # Persist metadata
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "minute_news_alignment_metadata.json"
    serializable = {
        t: {k: v for k, v in info.items() if isinstance(v, (str, int, float))}
        for t, info in metadata.items()
    }
    with open(meta_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Metadata → {meta_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align per-article FinBERT embeddings to minute bars (intraday)"
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to process (default: all with minute data)")
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="Cap number of tickers (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if output already cached")
    parser.add_argument("--min-articles", type=int, default=5,
                        help="Min articles desired before extending lookback (default: 5)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Max articles to pool per bar, ranked by impact (default: 15)")
    parser.add_argument("--max-lookback-days", type=int, default=10,
                        help="Primary lookback window in trading days (default: 10)")
    parser.add_argument("--article-cache-dir", type=str,
                        default="data/feature_cache/news_articles",
                        help="Directory with per-article FinBERT cache")
    parser.add_argument("--minute-history-dir", type=str,
                        default="data/minute_history")
    parser.add_argument("--output-dir", type=str,
                        default="data/feature_cache/minute_news")
    args = parser.parse_args()

    run_alignment(
        article_cache_dir=args.article_cache_dir,
        minute_history_dir=args.minute_history_dir,
        output_dir=args.output_dir,
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        min_articles=args.min_articles,
        top_k=args.top_k,
        max_lookback_days=args.max_lookback_days,
        force=args.force,
    )
