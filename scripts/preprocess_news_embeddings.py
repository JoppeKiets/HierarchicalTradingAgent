#!/usr/bin/env python3
"""Preprocess news articles into FinBERT embeddings and cache to disk.

This is a one-time (or periodic) batch job that:
  1. Loads news_articles.csv for each ticker
  2. Runs the Lexrank_summary (or Article) through FinBERT
  3. Produces TWO caches:

  (a) Daily-aggregate cache (for NewsEncoder sub-model / daily LSTM):
      data/feature_cache/news/{TICKER}_embeddings.npy   (N_days, 768)
      data/feature_cache/news/{TICKER}_sentiment.npy    (N_days, 6)
      data/feature_cache/news/{TICKER}_dates.npy        (N_days,) ordinal dates

  (b) Per-article cache (for intraday minute-bar news alignment):
      data/feature_cache/news_articles/{TICKER}_embeddings.npy  (N_articles, 768)
      data/feature_cache/news_articles/{TICKER}_sentiments.npy  (N_articles, 3)  [pos, neg, neu]
      data/feature_cache/news_articles/{TICKER}_timestamps.npy  (N_articles,)    unix seconds UTC
      data/feature_cache/news_articles/{TICKER}_art_lens.npy    (N_articles,)    article char len

  The per-article cache is only written for articles whose Date timestamp is
  NOT midnight-only (i.e., real intraday timestamps available), which in
  practice means articles collected since ~March 2026.  Midnight articles
  fall through to the daily-aggregate cache only.

Usage:
  python scripts/preprocess_news_embeddings.py                    # All tickers
  python scripts/preprocess_news_embeddings.py --tickers AAPL MSFT
  python scripts/preprocess_news_embeddings.py --max-tickers 100  # First 100
  python scripts/preprocess_news_embeddings.py --force            # Recompute all
  python scripts/preprocess_news_embeddings.py --batch-size 128   # Larger GPU batches
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# FinBERT Encoder
# ============================================================================

class FinBERTEncoder:
    """Batch-encode texts using FinBERT (ProsusAI/finbert).

    Uses the [CLS] embedding (768-dim) as the text representation and
    also extracts sentiment logits (positive, negative, neutral).
    """

    def __init__(self, device: str = "auto", model_name: str = "ProsusAI/finbert"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading FinBERT ({model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        # FinBERT has 3 classes: positive, negative, neutral
        self.sentiment_labels = ["positive", "negative", "neutral"]
        logger.info("FinBERT loaded ✓")

    @torch.no_grad()
    def encode_batch(
        self, texts: List[str], max_length: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a batch of texts.

        Args:
            texts: List of text strings
            max_length: Max token length (128 is fine for summaries ~200-500 chars)

        Returns:
            embeddings: (N, 768) — [CLS] hidden states
            sentiments: (N, 3) — softmax probabilities [positive, negative, neutral]
        """
        if not texts:
            return np.zeros((0, 768), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs, output_hidden_states=True)

        # [CLS] embedding from last hidden layer
        # outputs.hidden_states[-1] is (batch, seq_len, 768)
        cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

        # Sentiment probabilities
        sentiment_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        return cls_embeddings.astype(np.float32), sentiment_probs.astype(np.float32)


# ============================================================================
# Per-ticker preprocessing
# ============================================================================

def preprocess_ticker_news(
    ticker: str,
    organized_dir: str,
    cache_dir: str,
    encoder: FinBERTEncoder,
    batch_size: int = 64,
    force: bool = False,
    news_lag_days: int = 1,
    article_cache_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Process all news articles for one ticker → cached embeddings.

    Writes TWO caches:

    1. Daily-aggregate cache (``cache_dir``):
       {ticker}_embeddings.npy  (N_days, 768) — mean FinBERT per day
       {ticker}_sentiment.npy   (N_days, 6)   — sentiment features per day
       {ticker}_dates.npy       (N_days,)     — ordinal dates

    2. Per-article cache (``article_cache_dir``, optional):
       Written only for articles with a real intraday timestamp (not midnight).
       {ticker}_embeddings.npy  (N_articles, 768)
       {ticker}_sentiments.npy  (N_articles, 3)  — [pos, neg, neu] softmax
       {ticker}_timestamps.npy  (N_articles,)    — unix seconds UTC
       {ticker}_art_lens.npy    (N_articles,)    — raw article char length

    Returns metadata dict or None if failed.
    """
    emb_path = Path(cache_dir) / f"{ticker}_embeddings.npy"
    sent_path = Path(cache_dir) / f"{ticker}_sentiment.npy"
    date_path = Path(cache_dir) / f"{ticker}_dates.npy"

    # Determine if per-article cache already exists too (for skip logic)
    art_emb_path = art_sent_path = art_ts_path = art_len_path = None
    if article_cache_dir is not None:
        art_dir = Path(article_cache_dir)
        art_emb_path  = art_dir / f"{ticker}_embeddings.npy"
        art_sent_path = art_dir / f"{ticker}_sentiments.npy"
        art_ts_path   = art_dir / f"{ticker}_timestamps.npy"
        art_len_path  = art_dir / f"{ticker}_art_lens.npy"

    daily_cached = (not force
                    and emb_path.exists()
                    and sent_path.exists()
                    and date_path.exists())
    art_cached = (article_cache_dir is None
                  or (not force
                      and art_emb_path.exists()
                      and art_sent_path.exists()
                      and art_ts_path.exists()
                      and art_len_path.exists()))

    if daily_cached and art_cached:
        emb = np.load(emb_path, mmap_mode="r")
        return {"ticker": ticker, "n_days": emb.shape[0], "status": "cached"}

    news_file = Path(organized_dir) / ticker / "news_articles.csv"
    if not news_file.exists():
        return None

    try:
        ndf = pd.read_csv(news_file)
    except Exception as e:
        logger.warning(f"  {ticker}: read failed — {e}")
        return None

    if ndf.empty or "Date" not in ndf.columns:
        return None

    # Parse timestamps (UTC)
    ndf["ts"] = pd.to_datetime(ndf["Date"], errors="coerce", utc=True)
    ndf = ndf.dropna(subset=["ts"])
    if ndf.empty:
        return None

    ndf["date_only"] = ndf["ts"].dt.date

    # Identify articles with a real intraday timestamp (not midnight-only)
    is_midnight = (
        (ndf["ts"].dt.hour == 0)
        & (ndf["ts"].dt.minute == 0)
        & (ndf["ts"].dt.second == 0)
    )
    ndf["has_intraday_ts"] = ~is_midnight

    # Choose summary text: prefer Lexrank, fall back to Article, then title
    summary_col = None
    for col in ["Lexrank_summary", "Lsa_summary", "Textrank_summary", "Luhn_summary"]:
        if col in ndf.columns:
            summary_col = col
            break

    if summary_col is None:
        ndf["_text"] = ndf.get("Article_title", "").astype(str)
    else:
        ndf["_text"] = ndf[summary_col].fillna("").astype(str)
        mask_empty = ndf["_text"].str.strip() == ""
        # Fall back: Article body, then title
        if "Article" in ndf.columns:
            ndf.loc[mask_empty, "_text"] = ndf.loc[mask_empty, "Article"].fillna("").astype(str)
        if "Article_title" in ndf.columns:
            still_empty = ndf["_text"].str.strip() == ""
            ndf.loc[still_empty, "_text"] = ndf.loc[still_empty, "Article_title"].astype(str)

    # Article length (raw body, used as impact signal in alignment)
    if "Article" in ndf.columns:
        ndf["_art_len"] = ndf["Article"].fillna("").astype(str).str.len().astype(np.float32)
    else:
        ndf["_art_len"] = ndf["_text"].str.len().astype(np.float32)

    ndf = ndf[ndf["_text"].str.strip().str.len() > 10].copy()
    if ndf.empty:
        return None

    # ----------------------------------------------------------------
    # Encode ALL articles (daily + per-article cache share the same pass)
    # ----------------------------------------------------------------
    max_articles_per_day = 10

    # Build daily groups for daily-aggregate cache
    daily_groups = ndf.groupby("date_only")
    all_dates = sorted(daily_groups.groups.keys())

    texts_buffer: List[str] = []
    row_indices: List[int] = []  # maps each encoded text → original ndf row

    for day in all_dates:
        group = daily_groups.get_group(day).copy()
        if len(group) > max_articles_per_day:
            group = group.sample(max_articles_per_day, random_state=42)
        for idx in group.index:
            texts_buffer.append(str(ndf.loc[idx, "_text"])[:512])
            row_indices.append(idx)

    n_texts = len(texts_buffer)
    all_embeddings = np.zeros((n_texts, 768), dtype=np.float32)
    all_sentiments = np.zeros((n_texts, 3), dtype=np.float32)  # [pos, neg, neu]

    for start in range(0, n_texts, batch_size):
        end = min(start + batch_size, n_texts)
        try:
            embs, sents = encoder.encode_batch(texts_buffer[start:end])
            all_embeddings[start:end] = embs
            all_sentiments[start:end] = sents
        except Exception as e:
            logger.warning(f"  {ticker}: batch encode failed at {start}:{end} — {e}")

    # ----------------------------------------------------------------
    # (a) Build daily-aggregate cache
    # ----------------------------------------------------------------
    if not daily_cached:
        day_embeddings, day_sentiments, day_ordinals = [], [], []
        row_indices_arr = np.array(row_indices)
        ndf_index_to_pos = {idx: pos for pos, idx in enumerate(row_indices)}

        for day in all_dates:
            group = daily_groups.get_group(day)
            positions = [ndf_index_to_pos[i] for i in group.index if i in ndf_index_to_pos]
            if not positions:
                continue

            day_embs = all_embeddings[positions]
            day_sents = all_sentiments[positions]
            n_articles = len(positions)

            mean_emb = day_embs.mean(axis=0)
            pos_mean = day_sents[:, 0].mean()
            neg_mean = day_sents[:, 1].mean()
            neu_mean = day_sents[:, 2].mean()
            compound = day_sents[:, 0] - day_sents[:, 1]
            compound_mean = compound.mean()
            compound_std = compound.std() if len(compound) > 1 else 0.0

            sent_features = np.array([
                pos_mean, neg_mean, neu_mean,
                compound_mean, compound_std,
                float(n_articles),
            ], dtype=np.float32)

            day_embeddings.append(mean_emb)
            day_sentiments.append(sent_features)
            day_ordinals.append(day.toordinal())

        if day_embeddings:
            np.save(emb_path,  np.stack(day_embeddings))
            np.save(sent_path, np.stack(day_sentiments))
            np.save(date_path, np.array(day_ordinals, dtype=np.int32))

    # ----------------------------------------------------------------
    # (b) Build per-article cache (intraday-timestamped articles only)
    # ----------------------------------------------------------------
    if not art_cached and article_cache_dir is not None:
        art_dir.mkdir(parents=True, exist_ok=True)

        # Filter to rows that were encoded AND have a real intraday timestamp
        intraday_mask = ndf.loc[row_indices, "has_intraday_ts"].values
        intraday_positions = [pos for pos, keep in enumerate(intraday_mask) if keep]
        intraday_row_ids   = [row_indices[pos] for pos in intraday_positions]

        if intraday_positions:
            art_embs  = all_embeddings[intraday_positions]       # (N, 768)
            art_sents = all_sentiments[intraday_positions]        # (N, 3)
            art_ts    = ndf.loc[intraday_row_ids, "ts"].values   # datetime64[ns, UTC]
            art_lens  = ndf.loc[intraday_row_ids, "_art_len"].values.astype(np.float32)

            # Convert timestamps to unix seconds (int64)
            art_ts_unix = np.array(
                [int(pd.Timestamp(t).timestamp()) for t in art_ts],
                dtype=np.int64,
            )

            np.save(art_emb_path,  art_embs)
            np.save(art_sent_path, art_sents)
            np.save(art_ts_path,   art_ts_unix)
            np.save(art_len_path,  art_lens)
        else:
            # No intraday articles — write empty arrays so the file exists
            np.save(art_emb_path,  np.zeros((0, 768), dtype=np.float32))
            np.save(art_sent_path, np.zeros((0, 3),   dtype=np.float32))
            np.save(art_ts_path,   np.zeros((0,),     dtype=np.int64))
            np.save(art_len_path,  np.zeros((0,),     dtype=np.float32))

    # Return metadata
    n_days = len(all_dates)
    n_intraday = int(ndf["has_intraday_ts"].sum())
    return {
        "ticker": ticker,
        "n_days": n_days,
        "n_articles": n_texts,
        "n_intraday_articles": n_intraday,
        "date_range": f"{all_dates[0]} to {all_dates[-1]}" if all_dates else "empty",
        "status": "computed",
    }


# ============================================================================
# Main batch processing
# ============================================================================

def run_preprocessing(
    organized_dir: str = "data/organized",
    cache_dir: str = "data/feature_cache/news",
    article_cache_dir: str = "data/feature_cache/news_articles",
    tickers: Optional[List[str]] = None,
    max_tickers: int = 0,
    batch_size: int = 64,
    force: bool = False,
    device: str = "auto",
):
    """Run FinBERT preprocessing for all tickers.

    Writes both the daily-aggregate cache (``cache_dir``) and the per-article
    intraday cache (``article_cache_dir``).
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(article_cache_dir, exist_ok=True)

    # Discover tickers with news
    if tickers is None:
        org_path = Path(organized_dir)
        tickers = sorted([
            d.name for d in org_path.iterdir()
            if d.is_dir() and (d / "news_articles.csv").exists()
        ])
    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    logger.info(f"Processing {len(tickers)} tickers with news data")

    encoder = FinBERTEncoder(device=device)

    metadata = {}
    t0 = time.time()
    n_success = 0
    n_articles_total = 0
    n_intraday_total = 0

    for i, ticker in enumerate(tickers):
        info = preprocess_ticker_news(
            ticker, organized_dir, cache_dir, encoder,
            batch_size=batch_size, force=force,
            article_cache_dir=article_cache_dir,
        )
        if info:
            metadata[ticker] = info
            n_success += 1
            n_articles_total += info.get("n_articles", 0)
            n_intraday_total += info.get("n_intraday_articles", 0)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tickers) - i - 1) / rate
            logger.info(
                f"  Progress: {i+1}/{len(tickers)} ({n_success} ok) | "
                f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s | "
                f"{n_articles_total:,} articles ({n_intraday_total:,} intraday)"
            )

    elapsed = time.time() - t0
    logger.info(
        f"\nDone: {n_success}/{len(tickers)} tickers, "
        f"{n_articles_total:,} articles ({n_intraday_total:,} with intraday timestamps), "
        f"{elapsed:.0f}s"
    )

    # Save metadata
    meta_path = Path(cache_dir) / "news_embedding_metadata.json"
    serializable = {}
    for t, info in metadata.items():
        serializable[t] = {k: v for k, v in info.items() if isinstance(v, (str, int, float))}
    with open(meta_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Metadata → {meta_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess news → FinBERT embeddings")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to process")
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="Limit number of tickers (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="FinBERT batch size (increase for more GPU RAM)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cached")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu'")
    parser.add_argument("--organized-dir", type=str, default="data/organized")
    parser.add_argument("--cache-dir", type=str, default="data/feature_cache/news")
    parser.add_argument("--article-cache-dir", type=str,
                        default="data/feature_cache/news_articles",
                        help="Output dir for per-article intraday embeddings.")
    args = parser.parse_args()

    run_preprocessing(
        organized_dir=args.organized_dir,
        cache_dir=args.cache_dir,
        article_cache_dir=args.article_cache_dir,
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        batch_size=args.batch_size,
        force=args.force,
        device=args.device,
    )
