#!/usr/bin/env python3
"""Enhanced FinBERT Processing: Summarization + Encoding.

Usage:
    python scripts/process_collected_news.py --all

This script:
1.  Looks for new news in organized/{TICKER}/news_articles.csv
2.  Summarizes long text (Lexrank) if missing
3.  Runs FinBERT (scripts/preprocess_news_embeddings.py) on new summaries
4.  Re-aligns sequences (src/news_data.py)
"""

import os
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List

import pandas as pd
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    HAS_SUMMARIZER = True
except ImportError:
    HAS_SUMMARIZER = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def summarize_ticker_news(ticker: str, data_dir: str = "data/organized"):
    """Fills in Lexrank_summary for articles that have text but no summary."""
    if not HAS_SUMMARIZER:
        logger.warning(f"Sumy not installed. Skipping summarization for {ticker}")
        return False
        
    news_file = Path(data_dir) / ticker / "news_articles.csv"
    if not news_file.exists():
        return False

    try:
        df = pd.read_csv(news_file)
        if "Article" not in df.columns or "Lexrank_summary" not in df.columns:
            return False
            
        # Filter for rows that need a summary
        mask = df["Article"].notna() & df["Lexrank_summary"].isna()
        if not mask.any():
            return False
            
        summarizer = LexRankSummarizer()
        
        for idx in df[mask].index:
            text = str(df.loc[idx, "Article"])
            if len(text) < 200:
                df.at[idx, "Lexrank_summary"] = text
                continue
                
            try:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                summary_sentences = summarizer(parser.document, 2)  # 2 sentences
                summary = " ".join([str(s) for s in summary_sentences])
                df.at[idx, "Lexrank_summary"] = summary
            except Exception as e:
                logger.error(f"Error in summarization at index {idx}: {e}")
                
        df.to_csv(news_file, index=False)
        logger.info(f"[{ticker}] Summarized {mask.sum()} news articles")
        return True
        
    except Exception as e:
        logger.error(f"Error summarizing {ticker}: {e}")
        return False

def run_pipeline(tickers: List[str], max_tickers: int = None, force_reencode: bool = False, continuous: bool = False, interval: int = 3600):
    """Full news processing pipeline."""
    if max_tickers:
        tickers = tickers[:max_tickers]
        
    while True:
        processed_count = 0
        tickers_to_reencode = []
        start_time = time.time()
        
        for ticker in tickers:
            updated = summarize_ticker_news(ticker)
            if updated or force_reencode:
                tickers_to_reencode.append(ticker)
                processed_count += 1
                
        if tickers_to_reencode:
            # Step 1: Run FinBERT encoding
            logger.info(f"\nPhase 1: Running FinBERT on {len(tickers_to_reencode)} tickers...")
            subprocess.run([
                "python", "scripts/preprocess_news_embeddings.py",
                "--tickers"
            ] + tickers_to_reencode, check=True)

            # Step 2: Re-align news sequences (using news_data.py)
            logger.info("\nPhase 2: Re-aligning news sequences with price data...")
            preprocess_cmd = (
                f"from src.news_data import preprocess_all_news, NewsDataConfig; "
                f"from src.hierarchical_data import HierarchicalDataConfig; "
                f"cfg = NewsDataConfig(seq_len=720); "
                f"preprocess_all_news({tickers_to_reencode}, cfg, force=True)"
            )
            subprocess.run(["python", "-c", preprocess_cmd], check=True)

            logger.info(f"News processing finished logic for round. Processed {processed_count} tickers.")
        else:
            logger.info("No news processing updates needed in this round.")

        if not continuous:
            break
            
        elapsed = time.time() - start_time
        wait_time = max(0, interval - elapsed)
        if wait_time > 0:
            logger.info(f"Waiting {wait_time:.0f}s until next news processing round...")
            time.sleep(wait_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to process")
    parser.add_argument("--max-tickers", type=int, help="Limit number of tickers")
    parser.add_argument("--all", action="store_true", help="Process all tickers")
    parser.add_argument("--force", action="store_true", help="Force re-encoding even if no new summaries")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous processing mode")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between rounds")
    args = parser.parse_args()

    target_tickers = []
    if args.tickers:
        target_tickers = args.tickers
    elif args.all:
        target_tickers = [d.name for d in Path("data/organized").iterdir() if d.is_dir()]
    
    if not target_tickers:
        print("No tickers specified. Use --tickers, --all, or --max-tickers.")
    else:
        run_pipeline(target_tickers, args.max_tickers, args.force, args.continuous, args.interval)
