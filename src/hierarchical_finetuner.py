"""Joint fine-tuning trainer for hierarchical forecasting.

Contains:
  - JointFineTuner: Phase 4 sequential round-robin fine-tuning of all models
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.hierarchical_config import TrainConfig
from src.hierarchical_data import HierarchicalDataConfig, _build_regime_dataframe
from src.hierarchical_metrics import compute_metrics
from src.hierarchical_models import HierarchicalForecaster
from src.hierarchical_trainers import _save_training_curve, clear_gpu_memory

logger = logging.getLogger(__name__)


# ============================================================================
# Phase 4: Sequential Round-Robin Fine-Tuning (memory-efficient)
# ============================================================================

class JointFineTuner:
    """Phase 4: Fine-tune sub-models one-at-a-time through the meta loss.

    Memory-efficient alternative to loading all 6 models on GPU at once.

    Strategy (per round):
      For each sub-model (lstm_d, tft_d, lstm_m, tft_m, [news]):
        1. Move that sub-model + meta to GPU; keep others on CPU.
        2. Unfreeze only that sub-model + meta; freeze everything else.
        3. Forward: run the active sub-model live (with gradients), but
           use cached frozen outputs for the other sub-models.
        4. Backprop through active sub-model → meta only.
        5. Move the sub-model back to CPU, advance to the next.

    This gives the same end-to-end gradient signal as full joint training
    but uses ~1/5 the GPU memory (only 1 sub-model + meta on GPU at a time).

    After all sub-models get one turn, that is one "round". We repeat for
    N rounds (= epochs_phase4).
    """

    def __init__(
        self,
        forecaster: HierarchicalForecaster,
        device: torch.device,
        tcfg: TrainConfig,
        data_cfg: HierarchicalDataConfig,
    ):
        self.forecaster = forecaster
        self.device = device
        self.tcfg = tcfg
        self.data_cfg = data_cfg
        self.criterion = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()
        self.scaler = GradScaler() if (tcfg.use_amp and device.type == "cuda") else None
        # More workers for Phase 4 — cache collection is CPU-bound with 24 cores available
        self.num_workers = min(tcfg.num_workers * 2, 12)

    # ------------------------------------------------------------------
    # Helpers: collect frozen sub-model outputs (like MetaTrainer does)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_frozen_outputs(
        self,
        forecaster: HierarchicalForecaster,
        daily_loader: DataLoader,
        minute_loader: DataLoader,
        data_cfg: HierarchicalDataConfig,
        news_loader: Optional[DataLoader] = None,
        fund_loader: Optional[DataLoader] = None,
        graph_loader: Optional[DataLoader] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run all sub-models and cache predictions + embeddings.

        Uses the registry pattern: iterates over forecaster.sub_model_names,
        runs each on its modality's loader, then aligns by (ticker, date).

        Returns a flat dict with:
            keys:          list of (ticker, ordinal_date) tuples
            targets:       (N,) tensor
            regimes:       (N, R) tensor
            {name}_pred:   (N,) tensor for each sub-model
            {name}_emb:    (N, E) tensor for each sub-model
        """
        E = forecaster.cfg.embedding_dim
        R = forecaster.cfg.regime_dim
        sub_names = forecaster.sub_model_names

        # Map modality names → data loaders
        modality_loaders = {
            "daily": daily_loader,
            "minute": minute_loader,
            "news": news_loader,
            "fundamental": fund_loader,
            "graph": graph_loader,
        }

        # Build regime lookup
        regime_df = _build_regime_dataframe(data_cfg)
        regime_lookup: Dict[int, np.ndarray] = {}
        if not regime_df.empty:
            for d, row in regime_df.iterrows():
                if hasattr(d, 'toordinal'):
                    regime_lookup[d.toordinal()] = row.values.astype(np.float32)

        # --- Run each sub-model one at a time to minimize VRAM ---
        # per_model[name][(ticker, date)] = {"pred": ..., "emb": ...}
        per_model: Dict[str, Dict[Tuple[str, int], Dict]] = {}
        daily_targets: Dict[Tuple[str, int], torch.Tensor] = {}
        daily_regimes: Dict[Tuple[str, int], torch.Tensor] = {}

        for name in sub_names:
            modality = HierarchicalForecaster.MODALITY[name]
            loader = modality_loaders.get(modality)
            if loader is None:
                per_model[name] = {}
                continue

            sub_model = forecaster.sub_models[name]
            sub_model.to(self.device).eval()
            model_data: Dict[Tuple[str, int], Dict] = {}

            if modality == "graph":
                # GNN: cross-sectional loader
                for batch in loader:
                    if isinstance(batch, dict):
                        nf = batch["node_features"].squeeze(0).to(self.device)
                        tgt = batch["targets"].squeeze(0)
                        mask = batch["mask"].squeeze(0)
                        ei = batch["edge_index"].squeeze(0).to(self.device)
                        od = int(batch["ordinal_date"])
                        tickers_list = batch["tickers"]
                    elif isinstance(batch, (list, tuple)):
                        b = batch[0]
                        nf = b["node_features"].to(self.device)
                        tgt = b["targets"]
                        mask = b["mask"]
                        ei = b["edge_index"].to(self.device)
                        od = int(b["ordinal_date"])
                        tickers_list = b["tickers"]
                    else:
                        continue
                    out = sub_model(nf, ei, mask)
                    valid_tickers = [tickers_list[j] for j in range(len(tickers_list)) if mask[j]]
                    for vi, ticker in enumerate(valid_tickers):
                        key = (ticker, od)
                        model_data[key] = {
                            "pred": out["prediction"][vi].cpu(),
                            "emb": out["embedding"][vi].cpu(),
                        }
            else:
                for batch in loader:
                    x, y, ordinal_dates, tickers_batch = batch
                    x_dev = x.to(self.device, non_blocking=True)
                    out = sub_model(x_dev)
                    for i in range(len(y)):
                        key = (tickers_batch[i], int(ordinal_dates[i]))
                        model_data[key] = {
                            "pred": out["prediction"][i].cpu(),
                            "emb": out["embedding"][i].cpu(),
                        }
                        # Collect targets + regime from daily modality
                        if modality == "daily" and key not in daily_targets:
                            daily_targets[key] = y[i]
                            ord_date = int(ordinal_dates[i])
                            daily_regimes[key] = (
                                torch.from_numpy(regime_lookup[ord_date])
                                if ord_date in regime_lookup
                                else torch.zeros(R)
                            )

            sub_model.cpu()
            per_model[name] = model_data

        clear_gpu_memory()

        # Warn if any sub-model produced NaN/Inf outputs (e.g. diverged in a prior phase)
        for name in sub_names:
            md = per_model[name]
            if md:
                sample_pred = next(iter(md.values()))["pred"]
                sample_emb  = next(iter(md.values()))["emb"]
                if not torch.isfinite(sample_pred).all() or not torch.isfinite(sample_emb).all():
                    nan_count = sum(
                        1 for v in md.values()
                        if not torch.isfinite(v["pred"]).all() or not torch.isfinite(v["emb"]).all()
                    )
                    logger.warning(
                        f"  [{name}] {nan_count}/{len(md)} cached outputs contain NaN/Inf "
                        f"(model likely diverged in a prior phase) — replacing with zeros"
                    )

        # --- Merge into aligned flat tensors ---
        # IMPORTANT: Use UNION of keys from all modalities (not just daily).
        # This ensures that minute samples with dates not in daily still get
        # training pairs (with zero-filled embeddings from missing modalities).
        # Without this, minute batches would fail to match and be skipped entirely.
        zero_pred = torch.zeros(1).squeeze()
        zero_emb = torch.zeros(E)

        all_keys_set = set(daily_targets.keys())
        for name in sub_names:
            all_keys_set.update(per_model[name].keys())

        keys_ordered = sorted(all_keys_set)  # Sort for reproducibility
        all_preds = {name: [] for name in sub_names}
        all_embs = {name: [] for name in sub_names}
        all_regimes, all_targets = [], []

        for key in keys_ordered:
            # Get target (zero-fill if missing from daily)
            if key in daily_targets:
                all_targets.append(daily_targets[key])
                all_regimes.append(daily_regimes[key])
            else:
                # Key from non-daily modality; use zero target and regime
                # (This is rare but can happen if minute has data beyond daily coverage)
                all_targets.append(torch.tensor(0.0))
                all_regimes.append(torch.zeros(R))

            for name in sub_names:
                md = per_model[name].get(key)
                if md is not None:
                    # Replace any NaN/Inf in cached outputs with zero so they don't
                    # propagate into meta input and cause unbounded losses in Phase 4
                    pred = torch.nan_to_num(md["pred"], nan=0.0, posinf=0.0, neginf=0.0)
                    emb  = torch.nan_to_num(md["emb"],  nan=0.0, posinf=0.0, neginf=0.0)
                    all_preds[name].append(pred)
                    all_embs[name].append(emb)
                else:
                    all_preds[name].append(zero_pred)
                    all_embs[name].append(zero_emb)

        result = {
            "keys": keys_ordered,
            "targets": torch.stack(all_targets),
            "regimes": torch.stack(all_regimes),
        }
        for name in sub_names:
            result[f"{name}_pred"] = torch.stack(all_preds[name])
            result[f"{name}_emb"] = torch.stack(all_embs[name])

        # Log coverage breakdown
        daily_coverage = sum(1 for k in keys_ordered if k in daily_targets)
        logger.info(
            f"  Frozen outputs collected: {len(keys_ordered):,} samples "
            f"({daily_coverage:,} from daily, "
            f"{len(keys_ordered) - daily_coverage:,} from other modalities)"
        )
        return result

    def _build_meta_input(
        self,
        cached: Dict[str, torch.Tensor],
        active_name: str,
        active_pred: torch.Tensor,
        active_emb: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Build meta model inputs, swapping in live outputs for the active sub-model.

        Returns (preds_stacked, emb_list, regime) ready for MetaMLP.
        """
        sub_names = self.forecaster.sub_model_names

        preds_list = []
        emb_list = []
        for name in sub_names:
            if name == active_name:
                preds_list.append(active_pred)
                emb_list.append(active_emb)
            else:
                preds_list.append(cached[f"{name}_pred"][indices].to(self.device))
                emb_list.append(cached[f"{name}_emb"][indices].to(self.device))

        preds = torch.stack(preds_list, dim=-1)  # (batch, N_sub)
        regime = cached["regimes"][indices].to(self.device)
        return preds, emb_list, regime

    # ------------------------------------------------------------------
    # Per-sub-model fine-tune step
    # ------------------------------------------------------------------
    def _finetune_one_submodel(
        self,
        sub_model: nn.Module,
        sub_name: str,
        data_loader: DataLoader,
        val_data_loader: DataLoader,
        cached_train: Dict[str, torch.Tensor],
        cached_val: Dict[str, torch.Tensor],
        train_key_to_idx: Dict[Tuple[str, int], int],
        val_key_to_idx: Dict[Tuple[str, int], int],
        optimizer: torch.optim.Optimizer,
        epoch_label: str,
    ) -> Tuple[float, float, Dict]:
        """Fine-tune one sub-model + meta for one pass over its data.

        1. Move sub_model + meta to GPU
        2. Unfreeze sub_model + meta, freeze everything else
        3. For each batch from sub_model's data loader:
           - Forward through sub_model (live, with gradients)
           - Build meta input using cached outputs for other sub-models
           - Forward through meta
           - Backprop loss through sub_model + meta
        4. Move sub_model back to CPU

        Returns (train_loss, val_loss, val_metrics).
        """
        # Ensure all sub-models and meta are on CPU first, then move active ones to GPU.
        # This prevents stale device state from a previous sub-model's turn.
        for name in self.forecaster.sub_model_names:
            self.forecaster.sub_models[name].cpu()
        self.forecaster.meta.cpu()
        clear_gpu_memory()

        sub_model.to(self.device)
        self.forecaster.meta.to(self.device)

        # Freeze everything, then unfreeze active sub-model + meta
        for p in self.forecaster.parameters():
            p.requires_grad = False
        for p in sub_model.parameters():
            p.requires_grad = True
        for p in self.forecaster.meta.parameters():
            p.requires_grad = True

        sub_model.train()
        self.forecaster.meta.train()

        # --- Train pass ---
        total_loss, n_b = 0.0, 0
        n_cache_miss, n_nan_loss = 0, 0
        for batch in data_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x_dev = x.to(self.device, non_blocking=True)
            target = y.to(self.device, non_blocking=True)

            # Find matching indices in the cached output arrays
            indices = []
            valid_mask = []
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                idx = train_key_to_idx.get(key, -1)
                indices.append(idx if idx >= 0 else 0)  # placeholder 0 for missing
                valid_mask.append(idx >= 0)

            indices_t = torch.tensor(indices, dtype=torch.long)
            valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool)

            if not valid_mask_t.any():
                n_cache_miss += 1
                continue

            # Forward through active sub-model
            out = sub_model(x_dev)
            active_pred = out["prediction"]   # (batch,)
            active_emb = out["embedding"]     # (batch, E)

            # Only use samples that matched in the cached output arrays
            active_pred_v = active_pred[valid_mask_t]
            active_emb_v = active_emb[valid_mask_t]
            target_v = target[valid_mask_t]
            indices_v = indices_t[valid_mask_t]

            # Replace any NaN/Inf in active outputs with zero before building meta input
            active_pred_v = torch.nan_to_num(active_pred_v, nan=0.0, posinf=0.0, neginf=0.0)
            active_emb_v  = torch.nan_to_num(active_emb_v,  nan=0.0, posinf=0.0, neginf=0.0)

            # Build meta inputs (swap in live outputs for active model)
            preds, emb_list, regime = self._build_meta_input(
                cached_train, sub_name, active_pred_v, active_emb_v, indices_v,
            )

            # Forward through meta
            meta_out = self.forecaster.meta(preds, emb_list, regime)
            loss = self.criterion(meta_out["prediction"], target_v)

            # Auxiliary loss on the sub-model's own prediction
            loss = loss + 0.1 * self.criterion(active_pred_v, target_v)

            # Skip batch if loss exploded to NaN/Inf (e.g. first round instability)
            if not torch.isfinite(loss):
                logger.warning(f"  [{sub_name}] Non-finite loss ({loss.item():.4g}) — skipping batch")
                n_nan_loss += 1
                continue

            optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(sub_model.parameters()) + list(self.forecaster.meta.parameters()),
                    self.tcfg.grad_clip * 0.5,
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(sub_model.parameters()) + list(self.forecaster.meta.parameters()),
                    self.tcfg.grad_clip * 0.5,
                )
                optimizer.step()
            total_loss += loss.item()
            n_b += 1

        train_loss = total_loss / max(n_b, 1)
        if n_b == 0:
            if n_nan_loss > 0:
                logger.warning(
                    f"  [{sub_name}] All {n_nan_loss} training batches had non-finite loss — "
                    "sub-model or cached embeddings may contain NaN. "
                    "Check if GNN/meta diverged in a prior phase."
                )
            elif n_cache_miss > 0:
                logger.warning(
                    f"  [{sub_name}] All {n_cache_miss} training batches had no valid cache matches — "
                    "check that Phase 3 cache keys align with Phase 4 data loaders."
                )

        # --- Validation pass (use val data loader for this sub-model's modality) ---
        sub_model.eval()
        self.forecaster.meta.eval()
        val_preds_l, val_tgts_l = [], []
        val_total, val_n = 0.0, 0

        # For validation, iterate the dedicated validation data loader
        with torch.no_grad():
            for batch in val_data_loader:
                x, y, ordinal_dates, tickers_batch = batch
                x_dev = x.to(self.device, non_blocking=True)
                target = y.to(self.device, non_blocking=True)

                indices = []
                valid_mask = []
                for i in range(len(y)):
                    key = (tickers_batch[i], int(ordinal_dates[i]))
                    idx = val_key_to_idx.get(key, -1)
                    indices.append(idx if idx >= 0 else 0)
                    valid_mask.append(idx >= 0)

                indices_t = torch.tensor(indices, dtype=torch.long)
                valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool)
                if not valid_mask_t.any():
                    continue

                out = sub_model(x_dev)
                pred_v = out["prediction"][valid_mask_t]
                emb_v = out["embedding"][valid_mask_t]
                indices_v = indices_t[valid_mask_t]
                target_v = target[valid_mask_t]

                preds, emb_list, regime = self._build_meta_input(
                    cached_val, sub_name, pred_v, emb_v, indices_v,
                )
                meta_out = self.forecaster.meta(preds, emb_list, regime)

                batch_loss = self.criterion(meta_out["prediction"], target_v)
                if torch.isfinite(batch_loss):
                    val_total += batch_loss.item()
                    val_n += 1
                val_preds_l.append(meta_out["prediction"].cpu().numpy())
                val_tgts_l.append(target_v.cpu().numpy())

        val_loss = val_total / max(val_n, 1)
        if val_preds_l:
            metrics = compute_metrics(np.concatenate(val_preds_l), np.concatenate(val_tgts_l))
        else:
            metrics = {"ic": 0.0, "rank_ic": 0.0, "directional_accuracy": 0.5}

        # Move both active models back to CPU to free VRAM
        sub_model.cpu()
        self.forecaster.meta.cpu()
        clear_gpu_memory()

        logger.info(
            f"  [{epoch_label}][{sub_name:7s}] train={train_loss:.6f} | "
            f"val={val_loss:.6f} | IC={metrics['ic']:.4f} | "
            f"RankIC={metrics['rank_ic']:.4f}"
        )

        return train_loss, val_loss, metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        d_train_dl: DataLoader, m_train_dl: DataLoader,
        d_val_dl: DataLoader, m_val_dl: DataLoader,
        n_rounds: int, save_dir: str,
        n_train_news_dl: Optional[DataLoader] = None,
        n_val_news_dl: Optional[DataLoader] = None,
        fund_train_dl: Optional[DataLoader] = None,
        fund_val_dl: Optional[DataLoader] = None,
        graph_train_dl: Optional[DataLoader] = None,
        graph_val_dl: Optional[DataLoader] = None,
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "finetune_best.pt")

        # --- Data-source → loader mapping ---
        DATA_SOURCE_LOADERS: Dict[str, Tuple[Optional[DataLoader], Optional[DataLoader]]] = {
            "daily":       (d_train_dl, d_val_dl),
            "minute":      (m_train_dl, m_val_dl),
            "news":        (n_train_news_dl, n_val_news_dl),
            "fundamental": (fund_train_dl, fund_val_dl),
            "graph":       (graph_train_dl, graph_val_dl),
        }

        # --- Step 1: Collect frozen outputs for all sub-models ---
        logger.info("Collecting frozen sub-model outputs for round-robin fine-tuning...")
        self.forecaster.eval()

        cached_train = self._collect_frozen_outputs(
            self.forecaster, d_train_dl, m_train_dl, self.data_cfg,
            news_loader=n_train_news_dl,
            fund_loader=fund_train_dl,
            graph_loader=graph_train_dl,
        )
        cached_val = self._collect_frozen_outputs(
            self.forecaster, d_val_dl, m_val_dl, self.data_cfg,
            news_loader=n_val_news_dl,
            fund_loader=fund_val_dl,
            graph_loader=graph_val_dl,
        )

        # Build key → index maps for fast lookup
        train_key_to_idx = {k: i for i, k in enumerate(cached_train["keys"])}
        val_key_to_idx = {k: i for i, k in enumerate(cached_val["keys"])}

        logger.info(f"  Train samples: {len(train_key_to_idx):,} | Val: {len(val_key_to_idx):,}")

        # --- Step 2: Set up sub-model → data loader mapping (registry-based) ---
        sub_models: List[Tuple[str, nn.Module, DataLoader, Optional[DataLoader]]] = []
        for name in self.forecaster.sub_model_names:
            ds = HierarchicalForecaster.MODALITY[name]
            # GNN uses cross-sectional batches; skip in round-robin fine-tuning
            # (GNN is trained separately in Phase 1.7 and its frozen outputs
            # are still included in the meta input via cached_train/cached_val)
            if ds == "graph":
                logger.info(f"  Skipping {name}: GNN uses cross-sectional batches (Phase 4 unsupported)")
                continue
            train_dl_pair, val_dl_pair = DATA_SOURCE_LOADERS[ds]
            if train_dl_pair is None:
                logger.warning(f"  Skipping {name}: no data loader for source '{ds}'")
                continue
            # Skip sub-models whose training loader has no samples — this
            # happens when the minute cache is stale and the test window is
            # empty.  Fine-tuning with zero batches produces no gradients and
            # can corrupt the meta model's cached embeddings for that slot.
            if len(train_dl_pair.dataset) == 0:
                logger.warning(
                    f"  Skipping {name} in Phase 4: training loader is empty "
                    f"(modality='{ds}', n_samples=0). "
                    f"Its phase-1/2 best weights are preserved unchanged."
                )
                continue
            sub_models.append((
                name,
                self.forecaster.sub_models[name],
                train_dl_pair,
                val_dl_pair,
            ))

        # Persistent optimizers — one per sub-model, created once so momentum
        # accumulates properly across rounds instead of resetting every round.
        persistent_optimizers: Dict[str, torch.optim.Optimizer] = {
            name: torch.optim.AdamW(
                list(self.forecaster.sub_models[name].parameters()) + list(self.forecaster.meta.parameters()),
                lr=self.tcfg.lr_finetune,
                weight_decay=self.tcfg.weight_decay,
            )
            for name, _, _, _ in sub_models
        }

        history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_rank_ic": []}
        best_val = float("inf")
        best_ic = -float("inf")
        patience_ctr = 0
        use_ic_stopping = (self.tcfg.early_stop_metric == "ic")

        total_params = sum(p.numel() for p in self.forecaster.parameters())
        logger.info(f"\n{'='*60}")
        logger.info(f"Sequential round-robin fine-tuning for {n_rounds} rounds")
        logger.info(f"  {len(sub_models)} sub-models, {total_params:,} total params")
        logger.info(f"  LR={self.tcfg.lr_finetune:.1e}, grad_clip={self.tcfg.grad_clip}")
        logger.info(f"  Only 1 sub-model + meta on GPU at a time (memory-safe)")
        logger.info(f"{'='*60}")

        for rnd in range(1, n_rounds + 1):
            t0 = time.time()
            round_train_losses = []
            round_val_losses = []
            round_val_ics = []

            for sub_name, sub_model, train_dl, val_dl in sub_models:
                # Use the persistent optimizer for this sub-model
                optimizer = persistent_optimizers[sub_name]

                tr_loss, v_loss, v_metrics = self._finetune_one_submodel(
                    sub_model, sub_name,
                    train_dl, val_dl, cached_train, cached_val,
                    train_key_to_idx, val_key_to_idx,
                    optimizer, f"Rnd {rnd:2d}",
                )

                round_train_losses.append(tr_loss)
                round_val_losses.append(v_loss)
                round_val_ics.append(v_metrics["ic"])

                # Update cached outputs for this sub-model after fine-tuning
                # so the next sub-model in this round sees the updated outputs
                self._update_cached_outputs(
                    sub_model, sub_name, train_dl, cached_train, train_key_to_idx,
                )
                self._update_cached_outputs(
                    sub_model, sub_name, val_dl, cached_val, val_key_to_idx,
                )

            elapsed = time.time() - t0
            avg_train = np.mean(round_train_losses)
            avg_val = np.mean(round_val_losses)
            avg_ic = np.mean(round_val_ics)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            history["val_ic"].append(avg_ic)
            history["val_rank_ic"].append(avg_ic)  # approximate

            # Gradient norm diagnostics — helps identify dead sub-models
            # (zero grad norm = no learning, likely frozen or disconnected)
            grad_norms = {}
            for sub_name, sub_mod in self.forecaster.sub_models.items():
                norms = [
                    p.grad.data.norm(2).item() ** 2
                    for p in sub_mod.parameters() if p.grad is not None
                ]
                grad_norms[sub_name] = float(sum(norms) ** 0.5) if norms else 0.0
            meta_norms = [
                p.grad.data.norm(2).item() ** 2
                for p in self.forecaster.meta.parameters() if p.grad is not None
            ]
            grad_norms["meta"] = float(sum(meta_norms) ** 0.5) if meta_norms else 0.0
            gnorm_str = " ".join(f"{k}={v:.2e}" for k, v in grad_norms.items())

            logger.info(
                f"[Joint] Round {rnd:3d}/{n_rounds} | "
                f"avg_train={avg_train:.6f} | avg_val={avg_val:.6f} | "
                f"avg_IC={avg_ic:.4f} | {elapsed:.1f}s | grad_norms: {gnorm_str}"
            )

            # Early stopping check
            if use_ic_stopping:
                improved = avg_ic > best_ic + self.tcfg.ic_min_delta
                if improved or (best_ic <= 0 and avg_val < best_val - self.tcfg.min_delta):
                    if improved:
                        best_ic = avg_ic
                    best_val = avg_val
                    patience_ctr = 0
                    self.forecaster.save(best_path)
                    logger.info(f"  ★ New best joint IC={avg_ic:.4f} | val={avg_val:.6f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.tcfg.patience:
                        logger.info(f"  Early stopping at round {rnd} (best IC={best_ic:.4f})")
                        break
            else:
                if avg_val < best_val - self.tcfg.min_delta:
                    best_val = avg_val
                    patience_ctr = 0
                    self.forecaster.save(best_path)
                    logger.info(f"  ★ New best joint val={avg_val:.6f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.tcfg.patience:
                        logger.info(f"  Early stopping at round {rnd}")
                        break

        # Reload best weights
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            for name in self.forecaster.sub_model_names:
                if name in best_ckpt:
                    self.forecaster.sub_models[name].load_state_dict(best_ckpt[name])
            if "meta" in best_ckpt:
                self.forecaster.meta.load_state_dict(best_ckpt["meta"])
        _save_training_curve(history, "Joint", save_dir)
        return history

    @torch.no_grad()
    def _update_cached_outputs(
        self,
        sub_model: nn.Module,
        sub_name: str,
        data_loader: DataLoader,
        cached: Dict[str, torch.Tensor],
        key_to_idx: Dict[Tuple[str, int], int],
    ):
        """Re-run a sub-model after fine-tuning and update cached preds/embs.

        This way the next sub-model in the round sees the updated outputs.
        """
        sub_model.to(self.device).eval()
        for batch in data_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x_dev = x.to(self.device, non_blocking=True)
            out = sub_model(x_dev)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                idx = key_to_idx.get(key, -1)
                if idx >= 0:
                    cached[f"{sub_name}_pred"][idx] = out["prediction"][i].cpu()
                    cached[f"{sub_name}_emb"][idx] = out["embedding"][i].cpu()
        sub_model.cpu()
        clear_gpu_memory()
