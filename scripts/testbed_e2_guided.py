"""E2: level-shift prototype steering (hybrid_needles shift slice) in the testbed.

Mirrors parent50: nu=0.2, 50 DDIM, lam=0.3 band + lam_anom=1 proto, unit grads,
6 prototypes shared by all samples. Varies backbone and guidance space.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import (GPChannel, Schedule, LinearDenoiser, BlockEncoder, Shell,
                                  GuideTerm, guided_ddim, proto_energy, level_shift)
from anogen.testbed.metrics import gallery_report
from anogen.shell.realism import diag_cusum, diag_persistence

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed"); OUT.mkdir(parents=True, exist_ok=True)
W, B, N = 512, 128, 384
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng); lo, hi = float(train.min()), float(train.max())
enc = BlockEncoder(W, 8)
shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
long = ch.sample(N, W + B, rng); donor = long[:, B:]
# six real-like level shifts: steps of +0.25 span at spread positions
ls_pos = np.array([90, 150, 210, 270, 330, 400]); ls_size = np.full(6, 0.25)
protos_x = level_shift(ch.sample(6, W, rng), ls_pos, ls_size)
real_ls = level_shift(ch.sample(N, W, rng), rng.integers(40, 470, N), np.full(N, 0.25))
protos = enc.encode(protos_x)
pidx = np.random.default_rng(17).integers(0, 6, N)   # 64 samples per proto, as in chunked_guided_ddim

def shifted_protos(shift_blocks):
    """Proto embedding translated in time by whole blocks (edge-replicate)."""
    nb = enc.n_blocks; out = np.empty((N, protos.shape[1]))
    for i in range(N):
        a = protos[pidx[i]].reshape(2, nb); s = int(shift_blocks[i])
        idx = np.clip(np.arange(nb) - s, 0, nb - 1)
        out[i] = a[:, idx].reshape(-1)
    return out

def run(kind, space, final=1.0, lam=0.3, lam_anom=1.0, burn=False, proto_shift=False, t_window=(0.0, 1.0)):
    P = shifted_protos(np.random.default_rng(5).integers(-25, 26, N)) if proto_shift else protos[pidx]
    terms = [GuideTerm(lam, lambda x: shell.band(x)), GuideTerm(lam_anom, lambda x: proto_energy(enc, x, P))]
    if burn:
        den = LinearDenoiser(ch, W + B, sch, kind); x = guided_ddim(den, long, terms, rng=np.random.default_rng(1), space=space, final_scale=final, crop=B, t_window=t_window)
    else:
        den = LinearDenoiser(ch, W, sch, kind); x = guided_ddim(den, donor, terms, rng=np.random.default_rng(1), space=space, final_scale=final, t_window=t_window)
    return x, P

def step_report(x, P):
    u = (x - lo) / (hi - lo)
    cus = diag_cusum(u, min_segment=8)
    pers = diag_persistence(u)
    # targeting: where does the proto ask for the step? first block where the proto mean rises by half the step
    nb = enc.n_blocks
    want = np.array([np.argmax(p[:nb] > (p[:nb].min() + 0.5 * (p[:nb].max() - p[:nb].min()))) * enc.block / W for p in P])
    got = cus["position"]
    return {"shelf_rate": float(np.mean((np.abs(pers) > 0.5 * 0.25 / (hi - lo)) )),
            "cusum_pos_mean": float(np.mean(got)), "cusum_pos_std": float(np.std(got)),
            "target_pos_err": float(np.median(np.abs(got - want))),
            "proto_dist": float(np.mean(np.linalg.norm(enc.encode(x) - P, axis=1)))}

rows = {}
rep = {**gallery_report(real_ls, lo, hi), **step_report(real_ls, enc.encode(real_ls))}
rows["real_level_shift"] = rep
grid = [
    ("repo: causal, x-space, final kick", dict(kind="causal", space="x")),
    ("causal, x-space, no final kick", dict(kind="causal", space="x", final=0.0)),
    ("causal, x-space, final 0.25", dict(kind="causal", space="x", final=0.25)),
    ("causal, x0-space", dict(kind="causal", space="x0")),
    ("causal, x0-space, final 0.25", dict(kind="causal", space="x0", final=0.25)),
    ("causal, eps-space", dict(kind="causal", space="eps")),
    ("causal, eps-space lam x3", dict(kind="causal", space="eps", lam=0.9, lam_anom=3.0)),
    ("flip_ramp, x0-space", dict(kind="flip_ramp", space="x0")),
    ("flip_ramp, x0-space, final 0.25", dict(kind="flip_ramp", space="x0", final=0.25)),
    ("causal+burnin, x0-space", dict(kind="causal", space="x0", burn=True)),
    ("bidir, x-space, final kick", dict(kind="bidir", space="x")),
    ("bidir, x0-space", dict(kind="bidir", space="x0")),
    ("bidir, eps-space lam x3", dict(kind="bidir", space="eps", lam=0.9, lam_anom=3.0)),
    ("bidir, x0-space, proto shift", dict(kind="bidir", space="x0", proto_shift=True)),
    ("flip_ramp, x0-space, proto shift", dict(kind="flip_ramp", space="x0", proto_shift=True)),
]
for name, kw in grid:
    t0 = time.time()
    x, P = run(**kw)
    rows[name] = {**gallery_report(x, lo, hi), **step_report(x, P)}
for name, r in rows.items():
    print(f"{name:34s} env={r['env_window_frac']:.2f} max={r['env_max_excess']:.2f} dp999={r['diff_p999']:.3f} "
          f"cus={r['cusum_mean']:+.2f} pk@s={r['peak_at_start']:.2f} pk@e={r['peak_at_end']:.2f} "
          f"shelf={r['shelf_rate']:.2f} cpos={r['cusum_pos_mean']:.2f}±{r['cusum_pos_std']:.2f} "
          f"tgt_err={r['target_pos_err']:.2f} pdist={r['proto_dist']:.2f}", flush=True)
(OUT / "e2_guided_level_shift.json").write_text(json.dumps(rows, indent=1))
