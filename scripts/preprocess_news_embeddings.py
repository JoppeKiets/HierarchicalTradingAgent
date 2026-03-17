#!/usr/bin/env python3
"""Preprocess news articles into FinBERT embeddings and cache to disk.

This is a one-time (or periodic) batch job that:
  1. Loads news_articles.csv for each ticker
  2. Runs the Lexrank_summary (shortest summary) through FinBERT
  3. Aggregates per-day: mean embedding + sentiment scores
  4. Saves per-ticker cached embeddings as .npy files

Output (per ticker):
  data/feature_cache/news/{TICKER}_embeddings.npy   (N_days, 768) — FinBERT [CLS] mean per day
  data/feature_cache/news/{TICKER}_sentiment.npy    (N_days, 6)   — sentiment features per day
  data/feature_cache/news/{TICKER}_dates.npy        (N_days,)     — ordinal dates

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
) -> Optional[Dict]:
    """Process all news articles for one ticker → cached embeddings.

    For each calendar day with news:
      - Run FinBERT on each article's summary
      - Aggregate: mean embedding, mean/std/max sentiment, article count
      - Save with ordinal dates for alignment

    Returns metadata dict or None if failed.
    """
    emb_path = Path(cache_dir) / f"{ticker}_embeddings.npy"
    sent_path = Path(cache_dir) / f"{ticker}_sentiment.npy"
    date_path = Path(cache_dir) / f"{ticker}_dates.npy"

    if not force and emb_path.exists() and sent_path.exists() and date_path.exists():
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

    # Parse dates
    ndf["date"] = pd.to_datetime(ndf["Date"], errors="coerce", utc=True)
    ndf = ndf.dropna(subset=["date"])
    if ndf.empty:
        return None

    ndf["date_only"] = ndf["date"].dt.date

    # Choose summary text: prefer Lexrank (shortest), fall back to title
    summary_col = None
    for col in ["Lexrank_summary", "Lsa_summary", "Textrank_summary", "Luhn_summary"]:
        if col in ndf.columns:
            summary_col = col
            break

    if summary_col is None:
        # Fall back to title
        ndf["_text"] = ndf.get("Article_title", "").astype(str)
    else:
        # Use summary, fall back to title if summary is NaN/empty
        ndf["_text"] = ndf[summary_col].fillna("")
        mask_empty = ndf["_text"].str.strip() == ""
        if "Article_title" in ndf.columns:
            ndf.loc[mask_empty, "_text"] = ndf.loc[mask_empty, "Article_title"].astype(str)

    # Filter out empty texts
    ndf = ndf[ndf["_text"].str.strip().str.len() > 10].copy()
    if ndf.empty:
        return None

    # Option to cap articles per day for speed
    max_articles_per_day = 10 
    
    # Process per day, collect embeddings
    daily_groups = ndf.groupby("date_only")
    all_dates = sorted(daily_groups.groups.keys())

    day_embeddings = []   # (n_days, 768)
    day_sentiments = []   # (n_days, 6) = [sent_pos_mean, sent_neg_mean, sent_neu_mean,
    #                                       sent_compound_mean, sent_compound_std, news_count]
    day_ordinals = []

    texts_buffer = []
    date_indices = []     # maps each text → day index

    # Collect all texts with their day mapping
    for day_idx, day in enumerate(all_dates):
        group = daily_groups.get_group(day)
        if len(group) > max_articles_per_day:
            group = group.sample(max_articles_per_day, random_state=42)
        for _, row in group.iterrows():
            texts_buffer.append(str(row["_text"])[:512])  # truncate very long texts
            date_indices.append(day_idx)

    # Batch encode all texts at once
    n_texts = len(texts_buffer)
    all_embeddings = np.zeros((n_texts, 768), dtype=np.float32)
    all_sentiments = np.zeros((n_texts, 3), dtype=np.float32)

    for start in range(0, n_texts, batch_size):
        end = min(start + batch_size, n_texts)
        batch_texts = texts_buffer[start:end]
        try:
            embs, sents = encoder.encode_batch(batch_texts)
            all_embeddings[start:end] = embs
            all_sentiments[start:end] = sents
        except Exception as e:
            logger.warning(f"  {ticker}: batch encode failed at {start}:{end} — {e}")
            # Leave as zeros

    # Aggregate per day
    date_indices = np.array(date_indices)
    for day_idx, day in enumerate(all_dates):
        mask = date_indices == day_idx
        if not mask.any():
            continue

        day_embs = all_embeddings[mask]
        day_sents = all_sentiments[mask]
        n_articles = mask.sum()

        # Mean embedding for the day
        mean_emb = day_embs.mean(axis=0)

        # Sentiment features: [pos_mean, neg_mean, neu_mean, compound_mean, compound_std, count]
        pos_mean = day_sents[:, 0].mean()
        neg_mean = day_sents[:, 1].mean()
        neu_mean = day_sents[:, 2].mean()
        # Compound = positive - negative (like VADER compound)
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

    if not day_embeddings:
        return None

    day_embeddings = np.stack(day_embeddings)    # (n_days, 768)
    day_sentiments = np.stack(day_sentiments)    # (n_days, 6)
    day_ordinals = np.array(day_ordinals, dtype=np.int32)

    # Save
    np.save(emb_path, day_embeddings)
    np.save(sent_path, day_sentiments)
    np.save(date_path, day_ordinals)

    return {
        "ticker": ticker,
        "n_days": len(day_embeddings),
        "n_articles": n_texts,
        "date_range": f"{all_dates[0]} to {all_dates[-1]}",
        "status": "computed",
    }


# ============================================================================
# Main batch processing
# ============================================================================

def run_preprocessing(
    organized_dir: str = "data/organized",
    cache_dir: str = "data/feature_cache/news",
    tickers: Optional[List[str]] = None,
    max_tickers: int = 0,
    batch_size: int = 64,
    force: bool = False,
    device: str = "auto",
):
    """Run FinBERT preprocessing for all tickers."""
    os.makedirs(cache_dir, exist_ok=True)

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

    for i, ticker in enumerate(tickers):
        info = preprocess_ticker_news(
            ticker, organized_dir, cache_dir, encoder,
            batch_size=batch_size, force=force,
        )
        if info:
            metadata[ticker] = info
            n_success += 1
            n_articles_total += info.get("n_articles", 0)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tickers) - i - 1) / rate
            logger.info(
                f"  Progress: {i+1}/{len(tickers)} ({n_success} ok) | "
                f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s | "
                f"{n_articles_total:,} articles encoded"
            )

    elapsed = time.time() - t0
    logger.info(
        f"\nDone: {n_success}/{len(tickers)} tickers, "
        f"{n_articles_total:,} articles, {elapsed:.0f}s"
    )

    # Save metadata
    meta_path = Path(cache_dir) / "news_embedding_metadata.json"
    # Make metadata JSON-serializable
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
    args = parser.parse_args()

    run_preprocessing(
        organized_dir=args.organized_dir,
        cache_dir=args.cache_dir,
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        batch_size=args.batch_size,
        force=args.force,
        device=args.device,
    )
