# B4 bridge corrected analysis

Split `eval`; seed 0; 74 clips; 64 non-oracle noise replays per clip.

The scalar is computed from the s=0.90 grid feature and is therefore better-informed than a true intermediate scalar.

Material readout uses mean embedding cosine as a Bernoulli keep-accuracy and is tagged `UNCALIBRATED_COSINE`; the material-excluded mean is the citable value.

| recovery mean | axes | value | 95% CI | citable |
|---|---|---:|---:|---|
| incl_material | presence, timing, class, material | 0.526367 | [0.311186, 0.636955] | no |
| excl_material | presence, timing, class | 0.515191 | [0.360404, 0.537848] | **yes** |

Tier token: `BRIDGE_PARTIAL`. Seed sweep tokens: 0:BRIDGE_PARTIAL, 1:BRIDGE_PARTIAL, 2:BRIDGE_PARTIAL, 3:BRIDGE_PARTIAL. Seed-stable: **True**.

## Policy labels

- `diffrs_scalar`: `final_window_scalar_reject`
- `smc_scalar`: `final_window_scalar_resample`

## Exact scoring-call ledger

| JSON policy key | report label | exact scoring calls | replays | mean per noise replay |
|---|---|---:|---:|---:|
| oracle_axis_gated | oracle_axis_gated | 3070 | 1 | - |
| full_bon | full_bon | 1184 | 1 | - |
| same_compute_bon | same_compute_bon | 753 | 1 | - |
| diffrs_scalar | final_window_scalar_reject | 1795 | 1 | - |
| smc_scalar | final_window_scalar_resample | 1958 | 1 | - |
| final_rerank | final_rerank | 1184 | 1 | - |
| random_prune | random_prune | 1776 | 1 | - |
| non_oracle_axis_gated | non_oracle_axis_gated | 148212 | 64 | 2315.812500 |
