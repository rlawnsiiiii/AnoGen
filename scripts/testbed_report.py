"""Print the markdown tables for docs/TESTBED.md from results/testbed/*.json."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
e1 = {b: json.loads((d / f"e1_unguided_B{b}.json").read_text()) for b in (128, 256) if (d / f"e1_unguided_B{b}.json").is_file()}
print("### E1 unguided\n")
print("| start | denoiser | peak@start | peak@end | std ratio start / mid / end | Δ-std ratio start / mid / end |")
print("|---|---|---:|---:|---|---|")
g = e1[256]["galleries"] if 256 in e1 else e1[128]["galleries"]
for key, r in g.items():
    if key == "real_nominal":
        continue
    nu, kind = key.split("_", 1)
    if kind == "causal+burnin":
        kind = "causal + 256-bin burn-in"
    print(f"| ν={nu[2:]} | {kind} | {r['peak_at_start']:.3f} | {r['peak_at_end']:.3f} | "
          f"{r['std_ratio_start']:.2f} / {r['std_ratio_mid']:.2f} / {r['std_ratio_end']:.2f} | "
          f"{r['dstd_ratio_start']:.2f} / {r['dstd_ratio_mid']:.2f} / {r['dstd_ratio_end']:.2f} |")
if 128 in e1:
    for key in ("nu1.0_causal+burnin", "nu0.2_causal+burnin"):
        r = e1[128]["galleries"][key]
        print(f"| ν={key[2:5]} | causal + 128-bin burn-in | {r['peak_at_start']:.3f} | {r['peak_at_end']:.3f} | | |")
r = g["real_nominal"]
print(f"| — | real nominal (target) | {r['peak_at_start']:.3f} | {r['peak_at_end']:.3f} | 1 / 1 / 1 | 1 / 1 / 1 |")
print("\nx̂₀ error at t=40 (ν=0.2 edit time), MSE first 16 / middle 16 / last 16 bins:\n")
for k, v in e1[256 if 256 in e1 else 128]["x0hat_error_t40"].items():
    print(f"- {k}: {v['mse_first16']:.4f} / {v['mse_mid']:.4f} / {v['mse_last16']:.4f}")
e3 = json.loads((d / "e3_slices.json").read_text())
print("\n### E3a guidance VJP\n")
print("| denoiser | t | flat ∇: mass in first 10 % | centre ∇: centroid of landed edit |")
print("|---|---:|---:|---:|")
for k, v in e3["jacobian_mass_first10pct"].items():
    kind, t = k.rsplit("_t", 1)
    print(f"| {kind} | {t} | {v['flat_first10']:.3f} | {v['mid_centroid']:.3f} |")
print("\n### E3b band-only slice\n")
print("| sampler | env exit | diff p99.9 | peak@start | peak@end | position entropy | occupancy |")
print("|---|---:|---:|---:|---:|---:|---:|")
for k, r in e3.items():
    if k.startswith("band_only") or k == "real_nominal":
        print(f"| {k.replace('band_only: ', '')} | {r['env_window_frac']:.2f} | {r['diff_p999']:.3f} | {r['peak_at_start']:.2f} | "
              f"{r['peak_at_end']:.2f} | {r['peak_pos_entropy']:.2f} | {r.get('occupancy', float('nan')):.2f} |")
e2 = json.loads((d / "e2_guided_level_shift.json").read_text())
print("\n### E2 level-shift proto slice\n")
print("| sampler | env exit | diff p99.9 | CUSUM | peak@start | peak@end | shelf rate | step pos. mean ± sd | proto dist. |")
print("|---|---:|---:|---:|---:|---:|---:|---|---:|")
for k, r in e2.items():
    print(f"| {k} | {r['env_window_frac']:.2f} | {r['diff_p999']:.3f} | {r['cusum_mean']:+.2f} | {r['peak_at_start']:.2f} | "
          f"{r['peak_at_end']:.2f} | {r['shelf_rate']:.2f} | {r['cusum_pos_mean']:.2f} ± {r['cusum_pos_std']:.2f} | {r['proto_dist']:.2f} |")
e5 = json.loads((d / "e5_from_noise.json").read_text())
print("\n### E5 from-noise start\n")
print("| data mean | denoiser | start | generated mean | level bias (sd) |")
print("|---:|---|---|---:|---:|")
for k, r in e5.items():
    m, kind, init = k.split("_", 2)
    print(f"| {m[4:]} | {kind} | {init} | {r['level_mean']:.3f} | {r['level_bias_in_sd']:+.2f} |")
p6 = d / "e6_contrast.json"
if p6.is_file():
    e6 = json.loads(p6.read_text())
    real = e6["real level shifts"]
    print("\n### E6 level shifts: 6 shared prototypes vs a parametric step contrast\n")
    print(f"| sampler | target | shelf rate | step amp mean | step pos sd | pos W1 to real | diff p99.9 | start | end |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, r in e6.items():
        s, t = (k.split(" | ") + ["(target)"])[:2]
        print(f"| {s} | {t} | {r['shelf_rate']:.2f} | {r['step_amp_mean']:.3f} | {r['step_pos_std']:.3f} | "
              f"{r['step_pos_w1_to_real']:.3f} | {r['diff_p999']:.3f} | {r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} |")
p7 = d / "e7_phi.json"
if p7.is_file():
    e7 = json.loads(p7.read_text())
    print("\n### E7 what φ sees\n")
    print("Share of the pairwise squared φ distance between real nominal windows, by component:\n")
    print("| " + " | ".join(e7["share_of_pairwise_sq_distance"]) + " |")
    print("|" + "---:|" * len(e7["share_of_pairwise_sq_distance"]))
    print("| " + " | ".join(f"{v:.3f}" for v in e7["share_of_pairwise_sq_distance"].values()) + " |")
    print("\nUnguided regeneration of real nominal windows (no steering): mean shift of the three")
    print("high-frequency band energies, in real-nominal standard deviations:\n")
    print("| gallery | fft_band2 | fft_band3 | fft_band4 | ARP vs held-out real nominals |")
    print("|---|---:|---:|---:|---:|")
    for k, r in e7["galleries"].items():
        sh = r.get("mean_shift_in_real_sd") or {}
        print(f"| {k} | {sh.get('fft_band2', 0):+.2f} | {sh.get('fft_band3', 0):+.2f} | {sh.get('fft_band4', 0):+.2f} | {r['arp_real_queries']:.3f} |")
