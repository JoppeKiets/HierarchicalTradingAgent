#!/usr/bin/env python3
"""Quick memory profiling for the hierarchical system."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.hierarchical_models import HierarchicalForecaster, HierarchicalModelConfig

def test_memory():
    """Test memory usage of models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def print_mem():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - allocated
            print(f"  Allocated: {allocated:.2f}GB / Reserved: {reserved:.2f}GB / Free: {free:.2f}GB / Total: {total:.2f}GB")
    
    print("Memory Test\n" + "="*60)
    
    # Test model creation
    print("\n1. Creating model...")
    cfg = HierarchicalModelConfig(
        daily_input_dim=56,
        minute_input_dim=23,
    )
    model = HierarchicalForecaster(cfg).to(device)
    print_mem()
    
    # Test forward pass - daily batch
    print("\n2. Forward pass - daily batch (batch=64, seq=720, feat=56)...")
    torch.cuda.reset_peak_memory_stats()
    x_daily = torch.randn(64, 720, 56, device=device)
    x_minute = torch.randn(64, 780, 23, device=device)
    regime = torch.randn(64, 8, device=device)
    
    with torch.no_grad():
        out = model(x_daily, x_minute, regime)
    print_mem()
    
    # Test training batch (larger)
    print("\n3. Forward pass - larger batch (batch=128)...")
    torch.cuda.reset_peak_memory_stats()
    x_daily = torch.randn(128, 720, 56, device=device)
    x_minute = torch.randn(128, 780, 23, device=device)
    regime = torch.randn(128, 8, device=device)
    
    with torch.no_grad():
        out = model(x_daily, x_minute, regime)
    print_mem()
    
    # Test with gradient (backward)
    print("\n4. Forward + backward - batch (batch=32)...")
    torch.cuda.reset_peak_memory_stats()
    x_daily = torch.randn(32, 720, 56, device=device, requires_grad=True)
    x_minute = torch.randn(32, 780, 23, device=device, requires_grad=True)
    regime = torch.randn(32, 8, device=device)
    
    out = model(x_daily, x_minute, regime)
    loss = out["prediction"].sum()
    loss.backward()
    print_mem()
    
    print("\n" + "="*60)
    print("Recommendations:")
    print("  - Use --low-memory flag to reduce batch sizes")
    print("  - Or use: --batch-size 32 (for phase 1-2)")
    print("  - Phase 3 (meta) can use larger batches (up to 512)")

if __name__ == "__main__":
    test_memory()
