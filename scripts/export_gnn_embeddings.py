#!/usr/bin/env python3
"""Export GNN node embeddings for all (ticker, date) pairs in each split.

This script loads a trained GNN model and generates embeddings for all tickers/dates,
saving them in .npz format for use as auxiliary features in the main pipeline.

Usage:
    python scripts/export_gnn_embeddings.py \
        --model-path models/hierarchical_v11/checkpoint_phase1_7.pt \
        --output-dir data/feature_store \
        --data-cfg-kwargs split_mode=temporal daily_seq_len=720 minute_seq_len=780
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hierarchical_data import (
    HierarchicalDataConfig,
    LazyDailyDataset,
    create_graph_dataloaders,
)
from src.hierarchical_models import SectorGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gnn_model(model_path: str, device: torch.device) -> SectorGNN:
    """Load a trained GNN model from checkpoint."""
    logger.info(f"Loading GNN model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract GNN config from checkpoint
    if "gnn" in checkpoint:
        gnn_state = checkpoint["gnn"]
    else:
        gnn_state = checkpoint
    
    # Infer GNN dimensions from state dict
    # Look for input_proj.0.weight which has shape (hidden_dim, input_dim)
    if "input_proj.1.weight" in gnn_state:
        input_dim = gnn_state["input_proj.1.weight"].shape[1]
        hidden_dim = gnn_state["input_proj.1.weight"].shape[0]
    else:
        raise ValueError("Could not infer GNN dimensions from checkpoint")
    
    # Create GNN model with inferred dimensions
    gnn = SectorGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=4,
        n_heads=4,
        dropout=0.15,
        embedding_dim=64,
    )
    
    gnn.load_state_dict(gnn_state)
    gnn = gnn.to(device).eval()
    logger.info(f"✓ Loaded GNN model (input_dim={input_dim}, hidden_dim={hidden_dim})")
    return gnn


@torch.no_grad()
def extract_gnn_embeddings(
    gnn: SectorGNN,
    graph_loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract GNN embeddings for each split.
    
    Returns:
        dict with keys "train", "val", "test", each containing (embeddings, tickers, dates)
    """
    results = {}
    
    for split_name in ["train", "val", "test"]:
        if split_name not in graph_loaders:
            logger.warning(f"  No loader for split '{split_name}' — skipping")
            continue
        
        loader = graph_loaders[split_name]
        all_embeddings = []
        all_tickers = []
        all_dates = []
        
        logger.info(f"\nExtracting embeddings for {split_name} split...")
        
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, dict):
                nf = batch["node_features"].squeeze(0).to(device)
                mask = batch["mask"].squeeze(0).to(device)
                ei = batch["edge_index"].squeeze(0).to(device)
                od = int(batch["ordinal_date"])
                tickers_list = batch["tickers"]
            elif isinstance(batch, (list, tuple)):
                b = batch[0]
                nf = b["node_features"].to(device)
                mask = b["mask"].to(device)
                ei = b["edge_index"].to(device)
                od = int(b["ordinal_date"])
                tickers_list = b["tickers"]
            else:
                continue
            
            # Forward pass
            out = gnn(nf, ei, mask)
            embeddings = out["embedding"].cpu().numpy()  # (n_valid, embed_dim)
            
            # Collect valid tickers and dates
            valid_tickers = [tickers_list[j] for j in range(len(tickers_list)) if mask[j].item()]
            
            for j, ticker in enumerate(valid_tickers):
                all_embeddings.append(embeddings[j])
                all_tickers.append(ticker)
                all_dates.append(od)
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Processed {batch_idx + 1} batches, {len(all_embeddings)} samples so far...")
        
        if all_embeddings:
            embeddings_arr = np.array(all_embeddings, dtype=np.float32)
            tickers_arr = np.array(all_tickers, dtype=object)
            dates_arr = np.array(all_dates, dtype=np.int32)
            
            results[split_name] = (embeddings_arr, tickers_arr, dates_arr)
            logger.info(f"✓ {split_name}: {embeddings_arr.shape[0]} samples, "
                       f"embedding_dim={embeddings_arr.shape[1]}")
        else:
            logger.warning(f"  No embeddings extracted for {split_name}")
    
    return results


def save_embeddings(
    embeddings_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: str,
) -> None:
    """Save embeddings to .npz files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, (embeddings, tickers, dates) in embeddings_dict.items():
        output_path = Path(output_dir) / f"gnn_embeddings_{split_name}.npz"
        np.savez(
            output_path,
            embeddings=embeddings,
            tickers=tickers,
            dates=dates,
        )
        logger.info(f"✓ Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export GNN embeddings for use as auxiliary features in the main pipeline."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained GNN model checkpoint (e.g., models/hierarchical_v11/checkpoint_phase1_7.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/feature_store",
        help="Output directory for .npz files (default: data/feature_store)",
    )
    parser.add_argument(
        "--organized-dir",
        type=str,
        default="data/organized",
        help="Path to organized data directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/feature_cache",
        help="Path to feature cache directory",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="temporal",
        choices=["temporal", "ticker"],
        help="Split strategy",
    )
    parser.add_argument(
        "--daily-seq-len",
        type=int,
        default=720,
        help="Daily sequence length",
    )
    parser.add_argument(
        "--minute-seq-len",
        type=int,
        default=780,
        help="Minute sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu)",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Check that model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    
    # Load GNN model
    gnn = load_gnn_model(args.model_path, device)
    
    # Create data config
    data_cfg = HierarchicalDataConfig(
        organized_dir=args.organized_dir,
        cache_dir=args.cache_dir,
        split_mode=args.split_mode,
        daily_seq_len=args.daily_seq_len,
        minute_seq_len=args.minute_seq_len,
    )
    
    # Discover tickers
    organized_path = Path(args.organized_dir)
    if not organized_path.exists():
        logger.error(f"Organized data directory not found: {args.organized_dir}")
        sys.exit(1)
    
    tickers = sorted([d.name for d in organized_path.iterdir() if d.is_dir()])
    logger.info(f"Found {len(tickers)} tickers")
    
    # Create graph dataloaders
    logger.info("Creating graph dataloaders...")
    try:
        graph_loaders = create_graph_dataloaders(
            splits={"train": tickers, "val": tickers, "test": tickers},
            cfg=data_cfg,
            batch_size=1,
            num_workers=0,
            min_tickers_per_date=10,
        )
    except Exception as e:
        logger.error(f"Failed to create graph dataloaders: {e}")
        sys.exit(1)
    
    # Extract embeddings
    logger.info("Extracting GNN embeddings...")
    embeddings_dict = extract_gnn_embeddings(gnn, graph_loaders, device)
    
    # Save embeddings
    logger.info(f"Saving embeddings to {args.output_dir}...")
    save_embeddings(embeddings_dict, args.output_dir)
    
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
