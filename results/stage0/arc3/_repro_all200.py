import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from foley_cw import policy_offline as P
import scripts.phase4_policy as PH

records = PH.load_pool_records()
s_commit = PH.load_scommit()
gates = P.gates_from_scommit(s_commit, PH.PHASE1_S_GRID, PH.AXES)
pools = []
for clip in sorted(records):
    pool = PH.build_pool(clip, records[clip])
    if pool is not None:
        pools.append(pool)
print("n pools (all):", len(pools), flush=True)
all_scores = np.concatenate([p.final_score for p in pools])
diffrs_tau = float(np.median(all_scores))

metrics = P.run_all_policies(pools, gates=gates, axes=PH.AXES, num_steps=PH.NUM_STEPS,
                             seed=0, diffrs_tau=diffrs_tau, smc_temp=0.1, random_prune_frac=0.5)
print("REPRODUCED (all 200) final_correctness + per-axis:")
for pol in P.POLICIES:
    m = metrics[pol]
    print(f"  {pol:18s} final={m['final_correctness']:.3f}  P={m['correct_presence']:.3f} T={m['correct_timing']:.3f} C={m['correct_class']:.3f} M={m['correct_material']:.3f}")

argmax_corr = []; rand_corr = []; best = []
rng = np.random.default_rng(0)
for p in pools:
    fc = p.final_correct(P.DEFAULT_AXES)
    argmax_corr.append(fc[int(np.argmax(p.final_score))]); rand_corr.append(fc[rng.integers(len(fc))]); best.append(fc.max())
print(f"argmax-score winner all-axis-correct: {np.mean(argmax_corr):.3f}")
print(f"random candidate all-axis-correct   : {np.mean(rand_corr):.3f}")
print(f"oracle best-possible (>=1 correct)  : {np.mean(best):.3f}")
corrs = []
for p in pools:
    fc = p.final_correct(P.DEFAULT_AXES).astype(float); s = p.final_score
    if fc.std() > 0 and s.std() > 0:
        corrs.append(np.corrcoef(s, fc)[0, 1])
print(f"mean within-pool corr(scalar, all-axis proxy): {np.nanmean(corrs):.3f}")

import zlib
def _rng(*parts):
    ent = [0] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(ent))
gated_nfe = {}
for p in pools:
    r = P.simulate_policy(p, "oracle_axis_gated", gates=gates, axes=PH.AXES, num_steps=PH.NUM_STEPS, rng=_rng("oracle_axis_gated", p.clip), diffrs_tau=diffrs_tau, smc_temp=0.1)
    gated_nfe[p.clip] = r.total_nfe
W = {pol: [] for pol in ["full_bon", "final_rerank", "diffrs_scalar", "smc_scalar"]}
for p in pools:
    for pol in W:
        r = P.simulate_policy(p, pol, gates=gates, axes=PH.AXES, num_steps=PH.NUM_STEPS, rng=_rng(pol, p.clip), budget_nfe=gated_nfe.get(p.clip), diffrs_tau=diffrs_tau, smc_temp=0.1)
        W[pol].append(r.winner)
def agree(a, b): return np.mean([x == y for x, y in zip(W[a], W[b])])
print(f"winner agreement full_bon vs final_rerank : {agree('full_bon','final_rerank'):.3f}")
print(f"winner agreement full_bon vs diffrs_scalar: {agree('full_bon','diffrs_scalar'):.3f}")
print(f"winner agreement full_bon vs smc_scalar   : {agree('full_bon','smc_scalar'):.3f}")
