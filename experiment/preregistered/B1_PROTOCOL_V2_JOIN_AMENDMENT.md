# Arc-4 B-1 protocol V2 join and resume amendment

Frozen before the B-1 completeness gate completed and before any B-1
evaluation metric or prediction was produced or read.

The V2 implementation adds these conservative integrity requirements:

1. Every retained class label must belong to the frozen coarse-class universe
   in `configs/coarse_class_map.json` (SHA256
   `55b5a1d4116caa4503a6b4b17192425da487a9c4385a287e343d850795be4fe7`)
   or equal `abstain`.
2. For all 25,600 trajectory/progress bundles, the recollected per-token
   `token_mean` must match the independently retained Phase-1 `pooled` feature
   at `rtol=2e-3, atol=2e-3`. This proves the retap/cache trajectory join before
   any label is fitted. The tolerance was set after a non-outcome diagnostic on
   the first 10 lexicographically sorted bundle pairs: relative L2 error ranged
   from 1.8e-5 to 9.5e-5 and cosine was approximately one. No class label,
   prediction, or evaluation metric was inspected for that diagnostic.
3. Inner candidates write immutable, implementation-hashed journals after each
   family/probe/layer cell; selected outer specifications write one immutable
   journal per progress value. Resume validates identities, confident targets,
   predictions, and recomputed metrics. No token is emitted until all 448 inner
   candidate journals and all eight selected outer journals validate.
4. The collection-completion gate validates every declared launch, aggregate,
   session, exit-code, and worker-log hash; requires exit code zero and the
   complete aggregate counts; and pins node placement, GPUs 4-7, TP1 x four,
   local HF weights, and offline execution.

These checks can only turn an otherwise runnable evaluation into
`B1_INCOMPLETE`; they do not relax or tune a scientific threshold.
