"""Debug script: construct HierarchicalForecaster and print TFT minute param breakdown.

This file is intended to be runnable from the repo root.
"""
import sys
import pprint

sys.path.append('.')
from src.hierarchical_models import HierarchicalForecaster, HierarchicalModelConfig

cfg = HierarchicalModelConfig(
    use_news_model=True,
    use_tcn_d=True,
    use_fund_mlp=True,
    use_gnn_features=True,
)
# V10 overrides used in the run (minute models small in v10)
cfg.minute_hidden_dim = 64
cfg.minute_n_layers = 1
cfg.minute_dropout = 0.3

forecaster = HierarchicalForecaster(cfg)

print('Model config:')
pprint.pprint(vars(cfg))

print('\nPer-submodel params and input dims:')
for name in forecaster.sub_model_names:
    m = forecaster.sub_models[name]
    inp = getattr(m, 'input_dim', None)
    if inp is None and hasattr(m, 'config'):
        inp = getattr(m.config, 'input_dim', None)
    print(f"{name:8s}: params={m.count_parameters():,}, input_dim={inp}")

print('\nDetailed tft_m breakdown:')
total = 0
by_module = {}
for name, p in forecaster.tft_m.named_parameters():
    n = p.numel()
    total += n
    top = name.split('.')[0]
    by_module[top] = by_module.get(top, 0) + n

print("tft_m total params:", total)
for mod, cnt in sorted(by_module.items(), key=lambda x: -x[1])[:40]:
    print(f"{mod:30s} {cnt:,}")

print('\nGrand total (sub-models + meta):', sum(m.count_parameters() for m in forecaster.sub_models.values()) + forecaster.meta.count_parameters())