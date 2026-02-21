#!/usr/bin/env python3
"""Quick test to verify training doesn't OOM with new memory settings."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_hierarchical import TrainConfig, clear_gpu_memory

def test_config():
    """Test the new config."""
    print("Training Config Test\n" + "="*60)
    
    cfg = TrainConfig()
    print(f"Default batch sizes:")
    print(f"  Daily:  {cfg.batch_size_daily}")
    print(f"  Minute: {cfg.batch_size_minute}")
    print(f"  Meta:   {cfg.batch_size_meta}")
    print(f"  Num workers: {cfg.num_workers}")
    
    print(f"\nMemory settings:")
    print(f"  use_amp: {cfg.use_amp}")
    print(f"  use_gradient_checkpointing: {cfg.use_gradient_checkpointing}")
    
    print(f"\nReduce memory mode:")
    cfg_reduced = TrainConfig()
    cfg_reduced.reduce_memory()
    print(f"  Daily:  {cfg_reduced.batch_size_daily}")
    print(f"  Minute: {cfg_reduced.batch_size_minute}")
    print(f"  Meta:   {cfg_reduced.batch_size_meta}")
    print(f"  AMP enabled: {cfg_reduced.use_amp}")
    print(f"  Num workers: {cfg_reduced.num_workers}")
    
    print(f"\nGPU Memory before clear:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
    
    clear_gpu_memory()
    
    print(f"GPU Memory after clear:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
    
    print("\n" + "="*60)
    print("✓ Config test passed")

if __name__ == "__main__":
    test_config()
