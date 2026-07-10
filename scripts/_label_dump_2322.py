import sys, os, zlib, json
sys.path.insert(0, '/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions')
import numpy as np
from foley_cw import score_sde as K
from foley_cw.config import load_config
from foley_cw.types import ScheduleSpec
from foley_cw.mmaudio_backend import MMAudioBackend
from foley_cw.real_measurer import RealFoleyMeasurer

def rng_for(seed, *parts):
    return np.random.default_rng(np.random.SeedSequence([seed] + [zlib.crc32(str(p).encode()) for p in parts]))

cfg_all = load_config()
ax = {a.id: a for a in cfg_all.axes}
sch = ScheduleSpec(n_steps=20, scan_points=(0.05, 0.90), K_forks=8, N_independent=8)
m = RealFoleyMeasurer(device='cuda')
for cfgv in (1.0, 4.5):
    be = MMAudioBackend(variant='small_16k', device='cuda', full_precision=True,
                        cfg_strength=cfgv, num_steps=20, duration_sec=8.0, enable_conditions=True)
    cond = be.make_video_cond('data/FoleyBench/clips/2322.mp4', video_id='2322')
    labels, top2 = [], []
    for j in range(8):
        tr = K.generate_trajectory(be, cond, sch, rng_for(0, '2322', 'ind', j), alpha=0.0,
                                   record_points=(0.05,))
        probs, _ = m._panns_forward(tr['audio'])
        idx = np.argsort(probs)[::-1][:3]
        import csv
        labels.append(m.measure(tr['audio'], ax['class']).label)
        top2.append([(int(i), round(float(probs[i]), 3)) for i in idx])
    print(f'cfg={cfgv} independent class labels:', labels)
    print(f'cfg={cfgv} top-3 AudioSet (idx, prob):')
    for j, t in enumerate(top2):
        print('  ', j, t)
    del be
