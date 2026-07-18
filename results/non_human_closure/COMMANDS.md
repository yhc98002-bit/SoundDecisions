# Exact launch, resume, validation, merge, and reproduction commands

All scientific writers are create-only. Set `SD_NEW_ROOT` to a new, explicit
directory; never point a replay or reducer at the canonical root. Re-query GPU
occupancy immediately before each launch and replace only the explicit
`CUDA_VISIBLE_DEVICES` binding shown below.

```bash
export SD_REPO=/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-non-human-closure
export SD_ASSETS=/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions
export SD_ART=/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-non-human-artifacts/nhc_20260717T151345p0800_a052920
export SD_NEW_ROOT=/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-non-human-artifacts/<new-create-only-run-id>
export SD_PY=$SD_ASSETS/.venv/bin/python
export SD_PROTOCOL=$SD_REPO/experiment/non_human_closure/PROTOCOL.json
export SD_PROTOCOL_SHA=5c4fc4025995c16e355feb8cc02fbb3627891d47f6df052becde4845eaa7bd09
export SD_INVENTORY=$SD_ART/class/inventory_merged/B2_WAV_INVENTORY_MANIFEST.json
export SD_CLASS_CKPT=$SD_ASSETS/weights/measurers/Cnn14_16k_mAP=0.438.pth
export SD_MMAUDIO=$SD_ASSETS/third_party/MMAudio
export SD_WEIGHTS=$SD_ASSETS/weights/measurers
export SD_CLIPS=$SD_ASSETS/data/FoleyBench/clips
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=$SD_ASSETS/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
cd "$SD_REPO"
```

Inspect occupancy on each node before launch:

```bash
ssh an12 nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits
ssh an29 nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits
```

## Class posterior shards

The original seven placements were `an12:4,5,6` and `an29:0,1,3,4`, TP1,
one shard per GPU. Shard 0 used batch 32; shards 1–6 used batch 8. The exact
argv for every original worker is retained in each canonical shard completion
under `provenance.command`. A new worker is launched as:

```bash
CUDA_VISIBLE_DEVICES=<observed-free-gpu> "$SD_PY" scripts/b2_class_closure.py measure \
  --inventory-manifest "$SD_INVENTORY" \
  --protocol "$SD_PROTOCOL" \
  --out-dir "$SD_NEW_ROOT/class/full_shards/shard<I>_of7_<node>_gpu<G>" \
  --shard <I>/7 --device cuda:0 --batch-size <8-or-32> \
  --checkpoint "$SD_CLASS_CKPT" --coarse-map configs/coarse_class_map.json
```

Resume means validate each completed immutable shard, then launch only missing
shard IDs into new directories:

```bash
"$SD_PY" scripts/b2_class_closure.py validate-shard \
  --completion <shard-dir>/CLASS_POSTERIOR_SHARD_<I>_OF_00007.completion.json \
  --inventory-manifest "$SD_INVENTORY"
```

Merge and reproduce the frozen exploratory analysis:

```bash
"$SD_PY" scripts/b2_class_closure.py merge \
  --inventory-manifest "$SD_INVENTORY" \
  --completion <shard0-completion> --completion <shard1-completion> \
  --completion <shard2-completion> --completion <shard3-completion> \
  --completion <shard4-completion> --completion <shard5-completion> \
  --completion <shard6-completion> \
  --out-dir "$SD_NEW_ROOT/class/merged"

"$SD_PY" scripts/b2_class_closure.py analyze \
  --merged-completion "$SD_NEW_ROOT/class/merged/CLASS_POSTERIORS_MERGED.completion.json" \
  --protocol "$SD_PROTOCOL" --out-dir "$SD_NEW_ROOT/class/analysis" \
  --historical-json results/arc4_wpA2/class_reconstruction.json
```

The separately labeled post-hoc video-determined sensitivity is:

```bash
"$SD_PY" scripts/class_video_determined_sensitivity.py \
  --merged-completion "$SD_ART/class/merged_v2/CLASS_POSTERIORS_MERGED.completion.json" \
  --registered-analysis results/non_human_closure/CLASS_MULTISEED_COMMITMENT.json \
  --commitment-csv results/non_human_closure/CLASS_MULTISEED_COMMITMENT.csv \
  --out "$SD_NEW_ROOT/CLASS_VIDEO_DETERMINED_SENSITIVITY.json"
```

## B-1 same-forward gate

Selection and canonical packet creation:

```bash
"$SD_PY" scripts/b1_lineage_pilot.py select \
  --mmaudio-root "$SD_MMAUDIO" --weights-dir "$SD_WEIGHTS" --clips-root "$SD_CLIPS" \
  --output-root "$SD_NEW_ROOT/b1/selection" --attempt-id selection_v1 \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA"

CUDA_VISIBLE_DEVICES=<observed-free-gpu> "$SD_PY" scripts/b1_lineage_pilot.py make-packets \
  --mmaudio-root "$SD_MMAUDIO" --weights-dir "$SD_WEIGHTS" --clips-root "$SD_CLIPS" \
  --output-root "$SD_NEW_ROOT/b1/packets" --attempt-id packets_v1 \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA" \
  --selection-attempt "$SD_NEW_ROOT/b1/selection/selection_v1" --device cuda:0
```

Run two independent calibration replays, reduce the tolerance, then run and
reduce two fresh held-out replays. Do not pass clip `1002` to calibration:

```bash
CUDA_VISIBLE_DEVICES=<G0> "$SD_PY" scripts/b1_lineage_pilot.py replay \
  --mmaudio-root "$SD_MMAUDIO" --weights-dir "$SD_WEIGHTS" --clips-root "$SD_CLIPS" \
  --output-root "$SD_NEW_ROOT/b1/replay" --attempt-id cal_a \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA" \
  --packet-attempt "$SD_NEW_ROOT/b1/packets/packets_v1" \
  --role calibration --device cuda:0 --repeats 1 --repeat-offset 0

CUDA_VISIBLE_DEVICES=<G1> "$SD_PY" scripts/b1_lineage_pilot.py replay \
  --mmaudio-root "$SD_MMAUDIO" --weights-dir "$SD_WEIGHTS" --clips-root "$SD_CLIPS" \
  --output-root "$SD_NEW_ROOT/b1/replay" --attempt-id cal_b \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA" \
  --packet-attempt "$SD_NEW_ROOT/b1/packets/packets_v1" \
  --role calibration --device cuda:0 --repeats 1 --repeat-offset 1

"$SD_PY" scripts/b1_lineage_pilot.py calibrate \
  --output-root "$SD_NEW_ROOT/b1/calibration" --attempt-id calibration_v1 \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA" \
  --replay-attempt "$SD_NEW_ROOT/b1/replay/cal_a" \
  --replay-attempt "$SD_NEW_ROOT/b1/replay/cal_b"

export SD_CAL_ATTEMPT="$SD_NEW_ROOT/b1/calibration/calibration_v1"
export SD_TOL_SHA=$(sha256sum "$SD_CAL_ATTEMPT/TOLERANCE.json" | awk '{print $1}')
export SD_TOL_SCIENCE_SHA=$(jq -S 'del(.created_utc, .source_replay_attempts)' \
  "$SD_CAL_ATTEMPT/TOLERANCE.json" | sha256sum | awk '{print $1}')
test "$SD_TOL_SCIENCE_SHA" = cc653623b1474e75719382a2863e19c89bdd98a3b1c568b3524096556a103fc7

CUDA_VISIBLE_DEVICES=<G0> "$SD_PY" scripts/b1_lineage_pilot.py replay \
  --mmaudio-root "$SD_MMAUDIO" --weights-dir "$SD_WEIGHTS" --clips-root "$SD_CLIPS" \
  --output-root "$SD_NEW_ROOT/b1/replay" --attempt-id heldout_a \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA" \
  --packet-attempt "$SD_NEW_ROOT/b1/packets/packets_v1" \
  --role heldout --device cuda:0 --repeats 1 --repeat-offset 0 \
  --calibration-attempt "$SD_CAL_ATTEMPT" --tolerance-sha256 "$SD_TOL_SHA"

CUDA_VISIBLE_DEVICES=<G1> "$SD_PY" scripts/b1_lineage_pilot.py replay \
  --mmaudio-root "$SD_MMAUDIO" --weights-dir "$SD_WEIGHTS" --clips-root "$SD_CLIPS" \
  --output-root "$SD_NEW_ROOT/b1/replay" --attempt-id heldout_b \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA" \
  --packet-attempt "$SD_NEW_ROOT/b1/packets/packets_v1" \
  --role heldout --device cuda:0 --repeats 1 --repeat-offset 0 \
  --calibration-attempt "$SD_CAL_ATTEMPT" --tolerance-sha256 "$SD_TOL_SHA"

"$SD_PY" scripts/b1_lineage_pilot.py heldout \
  --output-root "$SD_NEW_ROOT/b1/heldout" --attempt-id heldout_v1 \
  --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA" \
  --replay-attempt "$SD_NEW_ROOT/b1/replay/heldout_a" \
  --replay-attempt "$SD_NEW_ROOT/b1/replay/heldout_b" \
  --calibration-attempt "$SD_CAL_ATTEMPT" --tolerance-sha256 "$SD_TOL_SHA"
```

The projection equality check binds the threshold values, registered clips and
progress grid, comparison classes, method, protocol, strata, and counts. It
excludes only `created_utc` and `source_replay_attempts`, whose absolute paths
and completion hashes are necessarily fresh-run-specific. The canonical full
file SHA is `2abbbae24cb249c22894d08c03a506374ce5b8a5fbf742be7e553f2220a80863`;
the held-out commands correctly consume the new file's full SHA in
`$SD_TOL_SHA`. Clip `1002` never contributes to either authority. Every stage
can be resumed only by recursive validation and launching a new missing
attempt:

```bash
"$SD_PY" scripts/b1_lineage_pilot.py validate \
  --attempt-root <attempt-root> --expected-stage <selection|packets|replay|calibration|heldout> \
  --protocol-sha256 "$SD_PROTOCOL_SHA"
```

## Full feature recollection

The original eight shards used `an12` GPUs 4–7, two independent TP1 replicas
per GPU. For each `<I>` in `0..7`:

```bash
CUDA_VISIBLE_DEVICES=<observed-free-gpu> "$SD_PY" scripts/b2_feature_recollection.py collect \
  --inventory-manifest "$SD_INVENTORY" \
  --heldout-attempt "$SD_ART/b1/heldout/heldout_v1" \
  --output-root "$SD_NEW_ROOT/b1_full/feature_shards" \
  --attempt-id shard<I>_of8 --shard <I>/8 \
  --mmaudio-root "$SD_MMAUDIO" --weights-dir "$SD_WEIGHTS" --clips-root "$SD_CLIPS" \
  --device cuda:0 --protocol "$SD_PROTOCOL" --protocol-sha256 "$SD_PROTOCOL_SHA"

"$SD_PY" scripts/b2_feature_recollection.py validate-shard \
  --completion <feature-shard>/FEATURE_SHARD_COMPLETION.json --deep
```

Merge only the exact validated union:

```bash
"$SD_PY" scripts/b2_feature_recollection.py merge \
  --completion <feature0> --completion <feature1> --completion <feature2> --completion <feature3> \
  --completion <feature4> --completion <feature5> --completion <feature6> --completion <feature7> \
  --out-dir "$SD_NEW_ROOT/b1_full/merged"
```

## Nested grouped Class readout

```bash
"$SD_PY" scripts/class_internal_readout.py prepare-targets \
  --class-completion "$SD_ART/class/merged_v2/CLASS_POSTERIORS_MERGED.completion.json" \
  --feature-completion "$SD_ART/b1_full/merged_v2/FEATURE_RECOLLECTION_COMPLETION.json" \
  --protocol "$SD_PROTOCOL" \
  --implementation experiment/non_human_closure/CLASS_READOUT_IMPLEMENTATION.json \
  --out-dir "$SD_NEW_ROOT/class_readout/targets"
```

Launch one immutable worker for each registered progress point
`0.05,0.15,0.25,0.35,0.45,0.60,0.75,0.90`, using only currently free GPUs:

```bash
CUDA_VISIBLE_DEVICES=<observed-free-gpu> "$SD_PY" scripts/class_internal_readout.py fit-progress \
  --feature-completion "$SD_ART/b1_full/merged_v2/FEATURE_RECOLLECTION_COMPLETION.json" \
  --target-completion "$SD_NEW_ROOT/class_readout/targets/TARGETS_COMPLETION.json" \
  --protocol "$SD_PROTOCOL" \
  --implementation experiment/non_human_closure/CLASS_READOUT_IMPLEMENTATION.json \
  --out-dir "$SD_NEW_ROOT/class_readout/shards/s<S>" --progress <S> --device cuda:0

"$SD_PY" scripts/class_internal_readout.py validate \
  --kind shard --completion "$SD_NEW_ROOT/class_readout/shards/s<S>/READOUT_SHARD_COMPLETION.json"
```

```bash
"$SD_PY" scripts/class_internal_readout.py merge \
  --completion <s005> --completion <s015> --completion <s025> --completion <s035> \
  --completion <s045> --completion <s060> --completion <s075> --completion <s090> \
  --out-dir "$SD_NEW_ROOT/class_readout/merged" \
  --bootstrap-draws 5000 --bootstrap-seed 20260717
```

## Material feasibility and canonical materialization

The canonical Material stop was produced at commit `ea007728...` under protocol
`a9f2a846...`. The detached-worktree command below binds the exact source,
protocol, input, and ffmpeg bytes. The output embeds the new absolute worktree
path, so validate its scientific projection rather than expecting the whole
report file to have the canonical path-dependent SHA. Exit code `3` is the
expected fail-closed `INCOMPLETE_ARTIFACTS` result:

```bash
export SD_MAT_COMMIT=ea0077285f22117910e2722d69bddc642f554e1c
export SD_MAT_PROTOCOL_SHA=a9f2a84653495045be039b17d1113de21fe1f6e951fffc0e5b65deb925473f39
export SD_MAT_TOOL_SHA=b985f2f7ddc16c12423376ccc999b72da5aa3c4592b49dce7975776606439e50
export SD_MAT_MODULE_SHA=d271d3e48a7aba4b98a1fef3c88c1b015334478e2ec53972e93dcca2009d55b1
export SD_MAT_FFMPEG=/HOME/paratera_xy/pxy1289/.local/bin/ffmpeg
export SD_MAT_FFMPEG_SHA=e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99
export SD_MAT_PARENT=$(mktemp -d /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-non-human-artifacts/material-ea007728-repro.XXXXXX)
export SD_MAT_REPO=$SD_MAT_PARENT/source
export SD_MAT_OUT=$SD_MAT_PARENT/output

git -C "$SD_REPO" worktree add --detach "$SD_MAT_REPO" "$SD_MAT_COMMIT"
test "$(git -C "$SD_MAT_REPO" rev-parse HEAD)" = "$SD_MAT_COMMIT"
printf '%s  %s\n' \
  "$SD_MAT_PROTOCOL_SHA" "$SD_MAT_REPO/experiment/non_human_closure/PROTOCOL.json" \
  "$SD_MAT_TOOL_SHA" "$SD_MAT_REPO/scripts/material_reference_feasibility.py" \
  "$SD_MAT_MODULE_SHA" "$SD_MAT_REPO/foley_cw/material_reference_feasibility.py" \
  "$SD_MAT_FFMPEG_SHA" "$SD_MAT_FFMPEG" | sha256sum -c -
printf '%s  %s\n' \
  85fd0fc905dbcf031e2ce43ab9d4832cd1b14d7c42fb3e01e34fb4549e013ad7 "$SD_ASSETS/results/stage0/measurements/measurements.jsonl" \
  695b5a93e3d52ea8fcb406a95aa7b6891c60f09fff93f6299f5359e739e5216f "$SD_ASSETS/data/FoleyBench/clips_index.csv" \
  18eac6ab1181d15b89fedbd77128c2e40150cf3fcc1ffc1719cb56d205b1d79b "$SD_ASSETS/results/stage0/anchors.json" | sha256sum -c -

set +e
"$SD_PY" "$SD_MAT_REPO/scripts/material_reference_feasibility.py" \
  --measurements "$SD_ASSETS/results/stage0/measurements/measurements.jsonl" \
  --phase2-journal-dir "$SD_ASSETS/results/stage0/journal" \
  --metadata-csv "$SD_ASSETS/data/FoleyBench/clips_index.csv" \
  --anchors-json "$SD_ASSETS/results/stage0/anchors.json" \
  --clips-root "$SD_CLIPS" \
  --protocol "$SD_MAT_REPO/experiment/non_human_closure/PROTOCOL.json" \
  --ffmpeg "$SD_MAT_FFMPEG" --out-dir "$SD_MAT_OUT" --loudness-workers 4
SD_MAT_RC=$?
set -e
test "$SD_MAT_RC" -eq 3
jq -e '.status == "INCOMPLETE_ARTIFACTS" and .coverage.legacy_journal_videos == 200 and .coverage.legacy_cells_inventoried == 6400 and .coverage.complete_candidate_videos == 3 and .coverage.valid_cells == 96' \
  "$SD_MAT_OUT/MATERIAL_REFERENCE_INSUFFICIENCY.json"

test "$(rg -c '"gen_id": ".+__p1cfg1_ind[0-3]".*"axis_id": "material".*"kind": "embedding".*"role": "p1cfg1_independent"' \
  "$SD_ASSETS/results/stage0/measurements/measurements.jsonl")" -eq 800
```

Running the current branch with `$SD_PROTOCOL` is a rule-equivalent smoke
reproduction only: commit `86f18c5` changed the B-1 asset contract and therefore
the full protocol bytes, while leaving the Material matching rules unchanged.
It is not the byte-exact canonical Material protocol invocation.

```bash
set +e
"$SD_PY" scripts/material_reference_feasibility.py \
  --measurements "$SD_ASSETS/results/stage0/measurements/measurements.jsonl" \
  --phase2-journal-dir "$SD_ASSETS/results/stage0/journal" \
  --metadata-csv "$SD_ASSETS/data/FoleyBench/clips_index.csv" \
  --anchors-json "$SD_ASSETS/results/stage0/anchors.json" \
  --clips-root "$SD_CLIPS" --protocol "$SD_PROTOCOL" \
  --ffmpeg "$SD_MAT_FFMPEG" \
  --out-dir "$SD_NEW_ROOT/material/current_protocol_smoke" --loudness-workers 4
SD_MAT_SMOKE_RC=$?
set -e
test "$SD_MAT_SMOKE_RC" -eq 3
```

Materialize and validate the integrated bundle into another create-only path:

```bash
"$SD_PY" scripts/materialize_non_human_closure.py materialize \
  --artifact-root "$SD_ART" --result-dir "$SD_NEW_ROOT/deliverable" \
  --support-dir results/non_human_closure

"$SD_PY" scripts/materialize_non_human_closure.py validate \
  --artifact-root "$SD_ART" --result-dir "$SD_NEW_ROOT/deliverable"
```

Final tests and committed-bundle checks:

```bash
"$SD_PY" -m pytest -q \
  tests/test_b1_lineage.py tests/test_b2_class_closure.py \
  tests/test_b2_feature_recollection.py tests/test_b2_inventory_merge.py \
  tests/test_class_internal_readout.py tests/test_class_video_determined_sensitivity.py \
  tests/test_class_probes.py tests/test_material_reference_feasibility.py \
  tests/test_materialize_non_human_closure.py tests/test_validate_non_human_closure_bundle.py

sha256sum -c results/non_human_closure/CHECKSUMS.sha256
```
