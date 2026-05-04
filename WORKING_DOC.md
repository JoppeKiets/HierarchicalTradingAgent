# HierarchicalTradingAgent — Master Working Document

**Project:** HierarchicalTradingAgent  
**Repo:** JoppeKiets/HierarchicalTradingAgent (branch: main)  
**Environment:** `/home/joppe-kietselaer/Desktop/coding/tradingAgent/.venv`  
**Version:** v0.6 (post daily-only ablation, Alpaca backfill running)  
**Status:** wf_bugfix_v4 (full ensemble): mean Sharpe **1.45** (5/5 positive). wf_daily_only_v2 (ablation): mean Sharpe **1.03** (4/5 positive) — full ensemble wins by +0.42. Alpaca backfill running (PID 1988290, ~1,344/7,694 tickers done, ~21h remaining). Next action: wait for backfill, then retrain full ensemble with historical minute data.

---

## 🔁 RESUME SNAPSHOT

```
Top Priorities:
  1. [TASK-011] Alpaca backfill running (PID 1988290) — 1,344/7,694 tickers, ~21h remaining
  2. After backfill: --force-preprocess to rebuild minute feature cache
  3. After cache rebuild: retrain full ensemble (wf_v5) with real historical minute data
  4. Expect minute model contribution to improve significantly (was near-zero coverage before)

Current Blockers:
  None — backfill running cleanly, no errors

Check backfill: tail -10 logs/alpaca_backfill.log
Check training: tail -20 results/wf_daily_only_v2.log

Resume Command: continue
```

---

## 📋 MASTER CHECKLIST

| ID | Title | Priority | Status | Estimate | Acceptance Criteria |
|----|-------|----------|--------|----------|---------------------|
| TASK-001 | Fix backtest: save + use raw returns instead of z-scores | BLOCKER | **Done ✅** | 2h | `backtest_long_short` receives clipped raw returns (±20%); total_return reflects real P&L; no more NaN Sharpe from negative compounding |
| TASK-002 | Fix AMP: skip NaN/inf batches in SubModelTrainer | HIGH | **Done ✅** | 1h | GradScaler init_scale=256; NaN-gradient batch skip applied; scaler.step skipped on inf grad_norm |
| TASK-003 | Fix nanmean/nanstd for Sharpe aggregation | HIGH | **Done ✅** | 20m | np.nanmean/nanstd used; aggregate Sharpe is numeric even when individual folds crash |
| TASK-004 | Cross-sectional z-score normalization at inference time | MEDIUM | **Done ✅** | 1h | Per-date z-score applied to predictions before backtest; RankIC improves ≥ 0.005 |
| TASK-005 | Add `--purge-horizon N` to default val/test split | MEDIUM | **Done ✅** | 30m | `--purge-horizon` CLI arg exists; defaults to `cfg.forecast_horizon` (1 day) |
| TASK-006 | Ablation: daily-only run (skip lstm_m, tft_m, news) | MEDIUM | **Done ✅** | 3h compute | Mean Sharpe=1.03 (4/5 positive) vs full ensemble 1.45 → keep full ensemble |
| TASK-011 | Backfill historical minute data via Alpaca API (2016→present) | HIGH | **In Progress 🔄** | ~1 day compute | 1,344/7,694 tickers done; ~21h remaining; 0 errors |
| TASK-007 | Add ranking loss (pairwise) to SubModelTrainer | MEDIUM | **Done ✅** | 2h | `rank_loss_weight=0.5` already default in TrainConfig; 50/50 MSE+rank blend active |
| TASK-008 | Fix `sharpe_mean=NaN` in aggregate summary | LOW | **Done ✅** | 20m | `np.nanmean` used; aggregate Sharpe is numeric |
| TASK-009 | Reduce model size: LSTM hidden 256→128, TFT d_model 512→256 | LOW | **Done ✅** | 1h + rerun | Already at targets: lstm hidden=128, tft hidden=64 |
| TASK-010 | Diagnose and fix news GNN NaN (news=0 every fold) | LOW | **Done ✅** | 2h | news contributes > 0 predictions in at least W4 |
| TASK-012 | Retrain full ensemble (wf_v5) with historical minute data | HIGH | **Pending ⏳** | ~6h compute | Mean Sharpe ≥ 1.6 (5/5 positive); minute attention weights positive |
| TASK-013 | Add `--resume-from-fold N` to walk-forward pipeline | HIGH | **Candidate** | 2h | Fold N–K can be re-run from checkpoint without redoing completed folds |
| TASK-014 | Integrate transaction costs into `backtest_long_short` | HIGH | **Candidate** | 3h | Sharpe after 5 bps commission + 5 bps slippage remains positive across all 5 windows |

---

## 🐛 BUG & IMPROVEMENT REGISTRY

---

### BUG-001 — Backtest uses EWMA z-scores as raw portfolio returns

- **Type:** Bug  
- **Priority:** BLOCKER  
- **Status:** Confirmed → Fix In Progress  
- **Impact:** 4/5 walk-forward windows show total_return = −1.0 (total ruin); Sharpe = NaN. All backtest P&L numbers are meaningless.

**Reproduction Steps:**
```python
import numpy as np
from src.hierarchical_data import LazyDailyDataset, HierarchicalDataConfig
import os
cfg = HierarchicalDataConfig()
ds = LazyDailyDataset("AAPL", cfg)
targets = [float(ds[i][1]) for i in range(min(500, len(ds)))]
print(f"std={np.std(targets):.3f}")  # → 0.987 (z-scores, NOT raw returns)
```

**Observed:** `targets` passed to `backtest_long_short` have std≈1.0, range ±5. `np.cumprod(1 + dr)` where `dr` is a z-score → immediate negative wealth for any day where top-k mean z-score < −1.

**Expected:** `targets` should be the raw clipped forward returns (±0.20), std≈0.012.

**Evidence:**
- `data/feature_cache/daily/AAPL_targets.npy`: valid values min=−5.0, max=5.0, std=0.987
- `HierarchicalDataConfig.target_transform = "ewma_zscore"` (default)
- `backtest_long_short` calls `np.cumprod(1 + dr)` with no de-normalization
- W0 Test IC = +0.012 (positive signal!) but total_return = −1.000

**Root Cause:** `preprocess_daily_ticker` saves z-scored targets to `*_targets.npy`. `LazyDailyDataset.__getitem__` returns these z-scores as `y`. `collect_test_predictions` collects these as `targets`. `backtest_long_short` uses them as dollar returns.

**Proposed Fix:**
1. In `preprocess_daily_ticker`: also save `{ticker}_raw_targets.npy` (clipped ±20% raw returns, never z-scored).
2. In `LazyDailyDataset.__getitem__`: return tuple `(seq, z_target, raw_target, ordinal_date, ticker)`.
3. In `collect_test_predictions`: collect both `aligned_tgts` (z-score, for IC) and `aligned_raw_tgts` (raw return, for backtest).
4. In `backtest_long_short` call sites: pass `raw_targets` instead of `targets`.

**Test Plan:**
```bash
# After fix: re-run E2 validation
python -c "
import numpy as np
from src.hierarchical_data import LazyDailyDataset, HierarchicalDataConfig
cfg = HierarchicalDataConfig()
ds = LazyDailyDataset('AAPL', cfg)
_, z, raw, _, _ = ds[100]
print('z-score:', float(z), '  raw:', float(raw))
# Expected: z in ±5, raw in ±0.20
"
```

**Rollback Plan:** Revert changes to `src/hierarchical_data.py` and `scripts/walk_forward_hierarchical.py`. No model retraining required (cached .npy files are unaffected until `--force-preprocess` is run).

---

### BUG-002 — Persistent inf/nan gradient norms in LSTM_M and TFT_D

- **Type:** Bug  
- **Priority:** HIGH  
- **Status:** Confirmed (log evidence)

**Evidence:**
```
[LSTM_M] batch 800/8865 grad_norm=inf  ← starts here, never recovers
[TFT_D]  batch 1700/3631 grad_norm=nan ← mid-epoch oscillation
```
LSTM_M: inf from epoch 1 batch 800 through all 20 epochs. Training MSE barely improves (0.366 → 0.297). Val IC drops epoch 1 (0.245) to epoch 6 early stop → model learned nothing new after first epoch.

**Root Cause:** AMP (`amp=True`) with float16 causes attention/LSTM activations to overflow. `GradScaler` doesn't recover because `scaler.step()` is called even when gradients contain inf/nan — `clip_grad_norm_` returns `inf` but does not zero out the gradients.

**Proposed Fix:**
1. In `SubModelTrainer.train_epoch`, after `scaler.unscale_()`, check `torch.isfinite(grad_norm)` and skip `scaler.step()` if False.
2. Reduce initial `GradScaler` scale: `GradScaler(init_scale=256)`.
3. Add `--no-amp` flag, default to False but use True for diagnostics.

---

### BUG-003 — SubModelTrainer has no NaN-batch skip (unlike MetaTrainer)

- **Type:** Bug  
- **Priority:** HIGH  
- **Status:** Confirmed (code review)

**Evidence:** `MetaTrainer.train_epoch` (line 580) has `if not torch.isfinite(pred).all(): skip`. `SubModelTrainer.train_epoch` (line 198) has no such check. NaN predictions from corrupt weights propagate to loss → nan loss → nan gradients → cascade.

**Proposed Fix:** Add the same isfinite guard to `SubModelTrainer.train_epoch`.

---

### BUG-004 — `sharpe_mean = NaN` in aggregate summary

- **Type:** Bug  
- **Priority:** LOW  
- **Status:** Confirmed

**Evidence:** W0, W2, W3 Sharpe = NaN (backtest crashed internally). `np.mean([NaN, 0.142, NaN, NaN, -0.283])` = NaN.

**Fix:** Use `np.nanmean` in aggregate computation.

---

### BUG-005 — Minute model trained on 55 days of data, deployed across 60-year test windows

- **Type:** Structural / Design  
- **Priority:** MEDIUM  
- **Status:** Fix in progress (TASK-011)

**Evidence:** Minute cache covers 2026-01-26 → 2026-04-06 (~55 days). All 5 test windows span 1962–2026. W4 test (2014–2026) has only 4.7% minute key coverage. Attention weights: `lstm_m=−0.183, tft_m=−0.170` (net negative).

**Fix:** Backfill 5+ years of historical minute data via Alpaca API (free tier, goes back to 2016). Script: `scripts/backfill_minute_alpaca.py`. Run daily-only ablation in the meantime with `--skip-models lstm_m tft_m`.

---

### TASK-011 — Alpaca Historical Minute Backfill

- **Type:** Data Infrastructure  
- **Priority:** HIGH  
- **Status:** In Progress 🔄

**Context:**
- yfinance only provides 8 days of 1-min history; all existing parquets start ~2026-01-26
- Alpaca free paper-trading API provides 1-min bars back to 2016 for all US equities
- 5,184 tickers currently have yfinance parquets; 7,694 tickers exist in `data/organized/`

**Setup done:**
- `alpaca-py` installed in `.venv` (`pip install alpaca-py`)
- API keys stored in `.env` (gitignored): `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- `python-dotenv` required in conda env: `conda run -n trading311 pip install python-dotenv`
- Backfill script: `scripts/backfill_minute_alpaca.py`

**Run command (full backfill 2020→present):**
```bash
conda run -n trading311 python scripts/backfill_minute_alpaca.py \
    --start 2020-01-01 \
    --tickers-from data/organized \
    --resume \
    --workers 2 \
    > logs/alpaca_backfill.log 2>&1 &
```

**Notes:**
- `--resume` skips tickers whose parquet already covers the requested range
- `--workers 2` keeps well under Alpaca's ~200 req/min free-tier rate limit
- Existing yfinance bars (Jan 2026 onward) are preserved and merged (no duplicates)
- Alpaca bars schema: `timestamp, open, high, low, close, volume` (no pre-computed indicators — those are computed in `_preprocess_minute_ticker`)
- After backfill: re-run `--force-preprocess` to rebuild minute feature cache, then retrain with full ensemble

**Acceptance Criteria:** ≥ 80% of tickers in each walk-forward test window have ≥ 1 year of minute bars.

---

### IMPROVEMENT-001 — Cross-sectional z-score normalization of predictions

- **Type:** Improvement  
- **Priority:** MEDIUM  
- **Status:** Candidate

**Hypothesis:** Per-date z-score of predictions before ranking improves cross-sectional discrimination. Validate with E4 experiment.

---

### IMPROVEMENT-002 — Replace MSE loss with pairwise ranking loss

- **Type:** Improvement  
- **Priority:** MEDIUM  
- **Status:** Candidate (code for `pairwise_rank_loss` already in `hierarchical_trainers.py`)

---

### TASK-012 — Retrain full ensemble (wf_v5) with historical minute data

- **Type:** Training Run  
- **Priority:** HIGH  
- **Status:** Pending ⏳ (blocked on TASK-011 backfill completing)

**Context:** Alpaca backfill (TASK-011) fills minute bars back to 2016 for ~7,694 tickers. Current minute coverage in walk-forward test windows is < 5% (bars start 2026-01-26). Once backfill is done, the feature cache must be rebuilt (`--force-preprocess`) and the full ensemble retrained. Daily-only ablation (wf_daily_only_v2) showed minute+news add +0.42 Sharpe when data exists; expect further gain with real historical bars.

**Run command:**
```bash
# 1. Rebuild minute feature cache
python scripts/walk_forward_hierarchical.py --force-preprocess --dry-run

# 2. Retrain full ensemble
python scripts/walk_forward_hierarchical.py \
    --rolling \
    --output-dir results/wf_v5_full_minute \
    --epochs 20
```

**Acceptance Criteria:**
- ≥ 80% of tickers in each window have ≥ 1 year of minute bars after cache rebuild
- `lstm_m` and `tft_m` attention weights > 0 in at least 4/5 windows
- Mean Sharpe ≥ 1.6 (5/5 positive)

**Rollback Plan:** Keep `results/wf_bugfix_v4/` checkpoint as the reference; if wf_v5 underperforms, revert to wf_bugfix_v4 model weights.

---

### TASK-013 — Add `--resume-from-fold N` to walk-forward pipeline

- **Type:** Infrastructure  
- **Priority:** HIGH  
- **Status:** Candidate

**Context:** Walk-forward runs take 6–24 h. If a run crashes at W4 (as happened with `wf_daily_only`), the only option is to rerun all folds from scratch. Fold checkpoints are already saved to `{output_dir}/fold_{w_idx}_forecaster.pt` (line 1228–1229 of `scripts/walk_forward_hierarchical.py`), so the infrastructure for resume already exists — the CLI flag and the loop skip logic just need to be wired in.

**Proposed Fix:**
1. Add `--resume-from-fold N` (int, default −1) to the argparse block in `walk_forward_hierarchical.py`.
2. In the main walk-forward loop, `if w_idx < args.resume_from_fold: continue` before training.
3. For the skipped folds, load predictions from `{output_dir}/fold_{w_idx}_preds.npz` (add npz save alongside the `.pt` checkpoint).
4. Aggregate results over all folds (completed + loaded).

**Acceptance Criteria:**
- `--resume-from-fold 4` skips folds 0–3, loads their saved predictions, trains only W4, and produces an identical aggregate summary.
- Unit test: run 2-fold experiment, kill after fold 0, resume, assert same fold-0 predictions.

**Estimated Effort:** small (2 h)

**Rollback Plan:** Flag defaults to −1 (no-op); removing the added code block restores original behavior.

---

### TASK-014 — Integrate transaction costs into `backtest_long_short`

- **Type:** Bug / Accuracy  
- **Priority:** HIGH  
- **Status:** Candidate

**Context:** `backtest_long_short` (line 746, `scripts/walk_forward_hierarchical.py`) computes portfolio returns as the simple mean of top-k / bottom-k raw returns with no deduction for commissions or slippage. `src/backtester.py` already implements a full `Backtester` class with configurable `commission_bps` and `slippage_bps` but it is not called from the walk-forward evaluation. All reported Sharpe ratios are therefore optimistic by an unknown but meaningful amount (at 10 bps round-trip, a 20-stock daily-rebalanced portfolio incurs ~2.5% annual drag).

**Proposed Fix:**
1. Add `commission_bps: float = 5.0` and `slippage_bps: float = 5.0` parameters to `backtest_long_short`.
2. Compute daily turnover (fraction of portfolio that changes between adjacent dates) and apply `cost = turnover * (commission_bps + slippage_bps) / 10_000` as a daily deduction before `np.cumprod`.
3. Expose `--commission-bps` and `--slippage-bps` CLI args (default 5 each).
4. Log net-of-cost Sharpe alongside gross Sharpe in the walk-forward summary JSON.

**Acceptance Criteria:**
- Net-of-cost Sharpe is printed and saved in `walk_forward_summary.json` for every window.
- With `--commission-bps 5 --slippage-bps 5` the mean net Sharpe for wf_bugfix_v4 is reported (may be lower than 1.45).
- With `--commission-bps 0 --slippage-bps 0` output is identical to current behavior.

**Estimated Effort:** small (3 h)

**Rollback Plan:** Default to 0 bps cost until validated; revert `backtest_long_short` signature change if downstream callers break.

---

### IMPROVEMENT-003 — Add turnover and portfolio concentration metrics to backtest output

- **Type:** Improvement  
- **Priority:** MEDIUM  
- **Status:** Candidate

**Hypothesis:** Daily turnover and portfolio HHI (Herfindahl-Hirschman Index) are leading indicators of strategy capacity and cost sensitivity. Without them, Sharpe numbers alone are insufficient to assess deployability. Both metrics can be computed inside `backtest_long_short` with < 20 lines of code and surfaced in the fold summary JSON.

---

### IMPROVEMENT-004 — Pre-training minute-data staleness check

- **Type:** Improvement / Safety  
- **Priority:** MEDIUM  
- **Status:** Candidate

**Hypothesis:** After future incremental backfills or data migrations, it is easy to accidentally retrain with a stale minute feature cache. Add a pre-flight check at the start of Phase 2 training (minute models) that verifies the median `mtime` of `data/feature_cache/minute/*.npy` files is newer than the median `mtime` of `data/minute_history/*.parquet` files, and warns (or aborts with `--strict-freshness`) if the cache is stale. This costs ~10 lines of code in `walk_forward_hierarchical.py` and prevents silent data bugs.

---

### IMPROVEMENT-005 — Hyperparameter sweep for `top_k` and `rank_loss_weight`

- **Type:** Improvement  
- **Priority:** MEDIUM  
- **Status:** Candidate (blocked on wf_v5 completing as stable baseline)

**Hypothesis:** `top_k=20` and `rank_loss_weight=0.5` are reasonable defaults but have not been tuned against held-out data. A small grid sweep (`top_k ∈ {10, 20, 30}`, `rank_loss_weight ∈ {0.25, 0.5, 0.75}`) over the 5-fold walk-forward using the already-cached features could identify a configuration that lifts net Sharpe by 0.1–0.2 at near-zero extra training cost (only MetaMLP needs retraining for `rank_loss_weight`; only backtest re-evaluation for `top_k`). Use W0–W3 as the search set; validate on W4.

---

## 📅 SESSION LOG

### Session 1 — 2026-04-28

**Actions Taken:**
- Performed full diagnostic across all artifacts: walk_forward_summary.json, walk_forward.log, src/hierarchical_data.py, src/hierarchical_trainers.py, scripts/walk_forward_hierarchical.py
- Confirmed BUG-001: AAPL_targets.npy has std=0.987, range ±5 (z-scores, not raw returns)
- Confirmed BUG-002: LSTM_M grad_norm=inf from batch 800/epoch 1 onward
- Confirmed BUG-003: No NaN-batch skip in SubModelTrainer
- Created WORKING_DOC.md
- **Starting TASK-001 execution**

**Results:** 5 bugs identified. Root causes established. Fix plan ready.

**Next Immediate Action:** TASK-001 — Patch `preprocess_daily_ticker` to save raw targets, update `LazyDailyDataset` and `collect_test_predictions`, fix `backtest_long_short` call.

---

*End of document — append new sessions below*

---

### Session 2 — 2026-05-01 / 2026-05-02

**Actions Taken:**
- Launched wf_bugfix_v2 (2 windows, 10 epochs): 2/2 positive Sharpe (mean 1.18), no crashes ✅
- Launched wf_bugfix_v3_full (5 windows, 20 epochs): **5/5 positive Sharpe** (mean 0.94), no -100% windows ✅
- Diagnosed news model contributing 0 predictions: root cause was `meta_trainer.train()` never passed news loaders
- **TASK-010 fix applied across 3 files:**
  - `scripts/walk_forward_hierarchical.py`: Added `WindowNewsDataset` class; updated `make_window_loaders` to create news loaders; passed `n_train_news_dl`/`n_val_news_dl` to `meta_trainer.train()`
  - `scripts/walk_forward_hierarchical.py`: Added `from src.news_data import LazyNewsDataset, NewsDataConfig` import
  - `src/hierarchical_trainers.py`: Fixed `_collect_sub_outputs` else-branch to handle news modality's distinct 5-tuple `(seq, target, ordinal_date, ticker, news_density)` and call `sub_model(x, news_coverage=news_cov_dev)`
- Also fixed: minute/fundamental dataset `__getitem__` to return 5-tuples (raw_target placeholder added); fixed 2x minute batch unpacks in walk_forward; deeper AMP loss-finite check before backward
- Launched wf_bugfix_v4 (5 windows, 20 epochs) with news model active

**Results (wf_bugfix_v3_full):**
- W0: Sharpe=0.99, TestIC=+0.011
- W1: Sharpe=0.82, TestIC=−0.007
- W2: Sharpe=0.74, TestIC=−0.004
- W3: Sharpe=1.81, TestIC=−0.011 ← high Sharpe despite negative IC (ranking working)
- W4: Sharpe=0.35, TestIC=+0.013
- Mean Sharpe: 0.94 ± 0.48 | 5/5 positive | news=0 (bug, now fixed)

**Next Immediate Action:** Monitor wf_bugfix_v4 — expect news to contribute predictions and IC to improve above 0.02+

**wf_bugfix_v4 final results (5/5 windows, 20 epochs, news enabled):**
- W0: Sharpe=1.71, TestIC=+0.0102 | W1: Sharpe=1.43, TestIC=+0.0105
- W2: Sharpe=2.02, TestIC=+0.0024 | W3: Sharpe=0.99, TestIC=+0.0081
- W4: Sharpe=1.11, TestIC=+0.0119
- **Mean Sharpe: 1.45 ± 0.38 | 5/5 positive Sharpe | 5/5 positive TestIC ✅**
- news contributed 1.77M keys (95% of union keys) — fix verified working

**TASK-006 / --skip-models implementation:**
- Added `use_minute_models: bool = True` to `HierarchicalModelConfig`; `count_sub_models()` base now 2 (daily only), +2 if minute, +1 if news
- `HierarchicalForecaster.__init__`: minute construction gated by `cfg.use_minute_models`
- `walk_forward_hierarchical.py`: `--skip-models` CLI arg; wired into `model_cfg`; Phase 2 skipped when minute excluded
- Launched: `wf_daily_only` ablation (`--skip-models lstm_m tft_m news`, 5 windows, 20 epochs)

**Next Immediate Action:** Wait for `wf_daily_only` results. If daily-only Mean Sharpe > 1.45, drop minute/news permanently.

**wf_bugfix_v4 final results (5/5 windows, 20 epochs, news enabled):**
- W0: Sharpe=1.71, TestIC=+0.0102
- W1: Sharpe=1.43, TestIC=+0.0105
- W2: Sharpe=2.02, TestIC=+0.0024
- W3: Sharpe=0.99, TestIC=+0.0081
- W4: Sharpe=1.11, TestIC=+0.0119
- **Mean Sharpe: 1.45 ± 0.38 | 5/5 positive | all TestIC positive ✅**
- news contributed 1.77M keys (95% of union) — fix verified working
- Meta train grew to 1.85M samples (vs 527K before news fix)

**TASK-006 implementation (--skip-models flag):**
- Added `use_minute_models: bool = True` to `HierarchicalModelConfig`
- Updated `count_sub_models()`: base is now 2 (lstm_d, tft_d), +2 if minute, +1 if news
- `HierarchicalForecaster.__init__`: minute model construction gated by `cfg.use_minute_models`
- `walk_forward_hierarchical.py`: `--skip-models` CLI arg; wired into `model_cfg`; Phase 2 skipped when minute models excluded
- Launched: `wf_daily_only` ablation (`--skip-models lstm_m tft_m news`)

---

### Session 3 — 2026-05-02 (continued)

**Actions Taken:**
- Confirmed wf_bugfix_v4 results (news model now active)
- Audited remaining Todo tasks: TASK-005/007/008/009 already implemented in codebase
- Launched TASK-006 daily-only ablation: `wf_ablation_daily_only` (skip lstm_m, tft_m, news)

**Results (wf_bugfix_v4 — all bugs fixed + news wired):**
- W0: Sharpe=1.71, TestIC=+0.010
- W1: Sharpe=1.43, TestIC=+0.011
- W2: Sharpe=2.02, TestIC=+0.002
- W3: Sharpe=0.99, TestIC=+0.008
- W4: Sharpe=1.11, TestIC=+0.012
- **Mean Sharpe: 1.45 ± 0.38 | 5/5 positive Sharpe | 5/5 positive IC**
- News contributed 1.77M keys in W4 (95% of union); meta training set = 1.85M samples
- IC-weighted Sharpe: 1.50

**Next:** Compare daily-only ablation results vs v4 to determine if minute/news models add alpha.

---

### Session 4 — 2026-05-03

**Actions Taken:**
- Investigated `wf_daily_only` crash: `collect_test_predictions` still called `forecaster.lstm_m` even when minute models were skipped → `KeyError: 'lstm_m'`
- Fixed: added `if minute_loader is not None and "lstm_m" in forecaster.sub_model_names:` guard around the minute collection block in `collect_test_predictions` (`scripts/walk_forward_hierarchical.py`)
- Relaunched as `wf_daily_only_v2` (PID 1979561, currently running)
- **Partial results from wf_daily_only (windows 0–3 completed, window 4 crashed before reporting):**
  - W0: Sharpe=1.086, TestIC=+0.024 | W1: Sharpe=1.500, TestIC=+0.013
  - W2: Sharpe=1.733, TestIC=+0.003 | W3: Sharpe=0.498, TestIC=+0.032
  - Mean Sharpe (0–3): **1.20** vs v4 full ensemble **1.45** → minute+news add ~+0.25 Sharpe
- Audited minute data pipeline end-to-end; identified root limitation: yfinance only provides 8 days of history; all 5,184 parquets start 2026-01-26; this is why lstm_m/tft_m attention weights were negative in historical walk-forward windows (near-zero minute key coverage)
- **TASK-011 (Alpaca backfill) set up:**
  - Alpaca free paper-trading account: API key `PK3QYEW67BF5...` stored in `.env` (gitignored)
  - `alpaca-py==0.43.4` installed in `.venv`
  - Connection verified: AAPL 2016-01-04 returned successfully (data back to 2016 confirmed)
  - `scripts/backfill_minute_alpaca.py` written:
    - Paginates Alpaca `StockBarsRequest` (1-min, up to 10k bars/request)
    - Merges with existing yfinance parquets (no duplicate timestamps)
    - `--resume` flag: skips tickers already covering the requested range
    - `--workers N` for parallelism (default 2, safe under free-tier 200 req/min)
    - Reads keys from `.env` via `python-dotenv`

**Current Status:**
- `wf_daily_only_v2`: running (~2h remaining)
- `backfill_minute_alpaca.py`: ready; needs `python-dotenv` in conda env

**Next Immediate Actions:**
1. Install dotenv + run test: `conda run -n trading311 pip install python-dotenv && conda run -n trading311 python scripts/backfill_minute_alpaca.py --start 2023-01-01 --end 2023-01-10 --tickers AAPL MSFT`
2. If test passes, launch full backfill: `--start 2020-01-01 --tickers-from data/organized --resume --workers 2`
3. Wait for `wf_daily_only_v2` final summary to confirm ablation result
4. After backfill complete: `--force-preprocess` to rebuild minute feature cache, then retrain full ensemble

**UPDATE — Session 4 completed:**

**wf_daily_only_v2 FINAL RESULTS (daily-only ablation):**
| Window | Val IC | Test IC | RankIC | DirAcc | Sharpe |
|--------|--------|---------|--------|--------|--------|
| W0 | 0.040 | +0.023 | 0.022 | 0.456 | 0.873 |
| W1 | 0.024 | +0.033 | 0.030 | 0.509 | 1.760 |
| W2 | 0.032 | −0.012 | 0.028 | 0.508 | 1.632 |
| W3 | 0.035 | +0.031 | 0.024 | 0.508 | 0.979 |
| W4 | 0.038 | +0.016 | 0.017 | 0.506 | −0.089 |
| **Mean** | | **+0.018** | 0.024 | 0.498 | **1.031** |

**Decision: Keep full ensemble.** Daily-only mean Sharpe 1.03 vs full ensemble (v4) 1.45 → minute+news add **+0.42 Sharpe**. Once backfill provides real historical minute data, gap expected to widen further.

**Alpaca backfill launched and running cleanly:**
- Fixed bugs: `.pyc` cache causing old code to run; tz-aware vs tz-naive datetime comparison in `fetch_bars` pagination cursor and `get_existing_range`
- Running: `setsid conda run -n trading311 python -B -u scripts/backfill_minute_alpaca.py --start 2016-01-01 --tickers-from data/organized --resume --workers 3 --delay 0.4`
- Status: 1,344/7,694 tickers (~17.5%), ~21h remaining, 0 errors

**Next Immediate Actions:**
1. Monitor backfill: `tail -10 logs/alpaca_backfill.log`
2. When backfill completes: `python scripts/walk_forward_hierarchical.py --force-preprocess`
3. Retrain: `python scripts/walk_forward_hierarchical.py --rolling --output-dir results/wf_v5_full_minute`

---

### Session 5 — 2026-05-03 (analyst pass)

**Actions Taken:**
- Full project artifact review: WORKING_DOC.md, README.md, `src/` directory, `scripts/` directory, `src/backtester.py`, `src/hierarchical_trainers.py`, walk-forward checkpoint and backtest sections
- Identified 3 new HIGH-priority tasks (TASK-012, TASK-013, TASK-014) and 3 MEDIUM-priority improvements (IMPROVEMENT-003, IMPROVEMENT-004, IMPROVEMENT-005)
- Added all entries to MASTER CHECKLIST and BUG & IMPROVEMENT REGISTRY

**Key Findings:**
- Reported Sharpe ratios are gross-of-costs; `src/backtester.py` has a complete tx-cost engine that is never wired into the walk-forward evaluation path → **TASK-014**
- Walk-forward fold checkpoints are already saved to `fold_{w_idx}_forecaster.pt` but no CLI flag exposes resume logic → **TASK-013**
- Next obvious training run (wf_v5 post-backfill) formalised with run command and acceptance criteria → **TASK-012**
- No pre-flight guard against stale minute feature cache → **IMPROVEMENT-004**
- No turnover or portfolio concentration metrics in backtest output → **IMPROVEMENT-003**
- `top_k` and `rank_loss_weight` have never been tuned; 5-fold pipeline now stable enough for a grid search → **IMPROVEMENT-005**

**Next Immediate Actions:**
1. Implement TASK-013 (`--resume-from-fold`) before launching wf_v5 to protect the long run
2. Implement TASK-014 (tx-costs) to get honest Sharpe numbers before wf_v5 results are reported
3. Monitor backfill: `tail -10 logs/alpaca_backfill.log`
4. When backfill completes: `--force-preprocess` then launch TASK-012 (wf_v5)

*End of document — append new sessions below*
