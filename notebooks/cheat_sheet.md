# PTF v2 — Cheat Sheet


## The 4 layers

| | Class | Does |
|---|---|---|
| **D1** | `TimeSeries` | raw data → tensors, column roles |
| **D2** | `<X>DataModule` | windowing, scaling, splits, dataloaders |
| **M** | `BaseModel` subclass | the network + train/predict steps |
| **P** | `<Model>_pkg_v2` | `fit` / `predict` / checkpointing |

D2 publishes `metadata` → M sizes itself from it. Models never see pandas.



## Column roles (D1)

Three **independent** axes — a column appears on each that applies.

| Axis | Values | Question |
|---|---|---|
| type | `num` / `cat` | number or category? |
| time | `static` or omit | constant per series? |
| future | `known` / `unknown` | do I know its future values? |

Plus: `time`, `target`, `group`.

```python
TimeSeries(
    data=df, 
    time="time_idx", 
    target="y",
    group=["series_id"],
    num=["price", "temperature"], 
    cat=["promo_type"],
    known=["price", "promo_type"],
    unknown=["temperature"],
    static=["store_size"],
)
```

- `group` = what identifies one series in a long DataFrame


## DataModule (D2)

```python
EncoderDecoderTimeSeriesDataModule(
    time_series_dataset=dataset,
    max_encoder_length=30,       # past steps in
    max_prediction_length=6,     # future steps out
    batch_size=32,
    scalers={...},               # per feature
    target_normalizer=...,      
    train_val_test_split=(0.7, 0.15, 0.15),
)
```

- `EncoderDecoderTimeSeriesDataModule` → TFT etc.
- `TslibDataModule` → TimeXer, Informer, AutoFormer (`context_length` / `prediction_length`)
- Split is **by groups**, not by time
- `train_dataloader` shuffles; `val` / `test` do not


## Scalers

Accepted per feature: `StandardScaler`, `RobustScaler`, `MinMaxScaler`,
`MaxAbsScaler`, `TorchNormalizer`, `GroupNormalizer`, `EncoderNormalizer`.

```python
scalers={"price": StandardScaler(), "temp": MinMaxScaler()}
```

Features not listed are untouched.

### `target_normalizer="auto"` picks:

| Condition | Result |
|---|---|
| categorical target | `NaNLabelEncoder` |
| `max_encoder_length > 20` and `min_encoder_length > 1` | `EncoderNormalizer` |
| else + `group` set | `GroupNormalizer` |
| else | `TorchNormalizer` |
| positive, skew > 2.5 | `transformation="log"` |
| positive, skew ≤ 2.5 | `"relu"` |
| has negatives | no transformation |
| multiple targets | wrapped in `MultiNormalizer` |

Multi-target: pass a list → auto-wrapped in `MultiNormalizer`.


## Model (M) + Package (P)

```python
# manual
model = TFT(loss=MAE(), hidden_size=64, metadata=dm.metadata)
Trainer(max_epochs=5).fit(model, dm)

# high-level
pkg = TFT_pkg_v2(datamodule_cfg=..., model_cfg=..., trainer_cfg=...)
pkg.fit(dataset)
preds = pkg.predict(dataset, return_info=[ "x", "y"])
```

Use M directly for your own Trainer/callbacks. Use P to just run it.

## Some things to keep in mind while trying out things in v2
- We still have no categorical support
  - The pipeline assume everything is numeric!
- Lacks multi-target support
- No support for `DistributionLoss`
- Train-test splitting is not implemented extensively
  - Only splitting based on groups
  - Eg, first 2 groups in train, 1 in test
- Not ALL v1 models are implemented in v2
- No hyperparam optimization
- No plotting funcitonalities
- The prediction of model is not configurable to get dfs etc, it is always `dict`