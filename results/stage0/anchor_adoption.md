# Anchor Adoption Decision — §3.2 (human marks vs audio-track onset)

30-clip human check: 27 marked, 3 marked 'no event', coverage 0.9000.

| stat | value (s) |
|---|---|
| MAE | 1.7415 |
| median \|Δ\| (= σ_anchor) | 0.9692 |
| RMSE | 2.9022 |
| std | 2.6151 |
| bias (human − audio) | -1.2587 |

**Decision: `AUDIO_ANCHOR_NOT_ADOPTED`** — σ_anchor = 0.9692s > 0.35s → keep approved chain; anchor source = `approved_chain(metadata>visual>light_human)`; **timing_bin_s = 1.5147** (bins ≥ 2·σ_anchor, floor 0.5s).

