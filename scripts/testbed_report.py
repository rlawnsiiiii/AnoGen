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
p10 = d / "e10_schedule.json"
if p10.is_file():
    e10 = json.loads(p10.read_text())
    print("\n### E10 texture the schedule can reproduce (closed form, exact bidirectional denoiser)\n")
    print("Generated / true variance, total and of first differences (texture), deterministic DDIM from t = T−1:\n")
    print("| texture sd | schedule | steps | variance | first-difference variance |")
    print("|---:|---|---:|---:|---:|")
    for k, r in e10.items():
        tex, name, steps = [x.strip() for x in k.split("|")]
        print(f"| {tex.split()[-1]} | {name} | {steps.split()[0]} | {r['var_ratio']:.2f} | {r['diff_var_ratio']:.2f} |")
p8 = d / "e8_trained_nets.json"
if p8.is_file():
    e8 = json.loads(p8.read_text())
    print("\n### E8 trained nonlinear denoisers (numpy dilated CNN, causal vs 'same' padding, same size)\n")
    print("| denoiser | x̂₀ MSE first 16 / middle / last 16 (t=40) | guidance VJP centroid | unguided ν=1 start / end | ν=0.2 start / end |")
    print("|---|---|---:|---|---|")
    for name, err in e8["x0_err_t40"].items():
        u1 = e8["unguided"][f"{name} nu=1.0"]
        u2 = e8["unguided"][f"{name} nu=0.2"]
        vj = e8["vjp_centroid_t40"].get(name)
        print(f"| {name} | {err['first16']:.4f} / {err['mid']:.4f} / {err['last16']:.4f} | {'' if vj is None else f'{vj:.2f}'} | "
              f"{u1['peak_at_start']:.3f} / {u1['peak_at_end']:.3f} | {u2['peak_at_start']:.3f} / {u2['peak_at_end']:.3f} |")
    if e8.get("level_shift"):
        print("\nLevel-shift prototype slice with the trained nets:\n")
        print("| sampler | start | end | diff p99.9 |")
        print("|---|---:|---:|---:|")
        for k, r in e8["level_shift"].items():
            print(f"| {k} | {r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} | {r['diff_p999']:.3f} |")
    if e8.get("detection"):
        print("\nLabel-free detection by denoising error (spikes of 0.15 at uniform positions vs nominal):\n")
        print("| denoiser | AUROC | spike in first 10 % | spike elsewhere |")
        print("|---|---:|---:|---:|")
        for k, r in e8["detection"].items():
            print(f"| {k} | {r['auroc']:.3f} | {r['auroc_spike_in_first_10pct']:.3f} | {r['auroc_spike_elsewhere']:.3f} |")
p8c = d / "e8c_context.json"
if p8c.is_file():
    e8c = json.loads(p8c.read_text())
    print("\n### E8c trained nets with real context around the window (generate W + context, keep W)\n")
    print("| configuration | start | end |")
    print("|---|---:|---:|")
    for k, r in e8c.items():
        print(f"| {k} | {r['peak_at_start']:.3f} | {r['peak_at_end']:.3f} |")
p2b = d / "e2b_twindow.json"
if p2b.is_file():
    e2b = json.loads(p2b.read_text())
    print("\n### E2b leaving the end of the trajectory unguided (level-shift slice, exact denoisers, x̂₀-space)\n")
    print("| denoiser | final scale | guided t / t* window | diff p99.9 | start | end | proto dist. |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for k, r in e2b.items():
        parts = k.split(" ")
        print(f"| {parts[0]} | {parts[3]} | {' '.join(parts[5:])} | {r['diff_p999']:.3f} | {r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} | {r['proto_dist']:.2f} |")
p6l = d / "e6_contrast_local64.json"
if p6l.is_file():
    e6l = json.loads(p6l.read_text())
    print("\n### E6b ±64-bin step contrast instead of whole-window segments\n")
    print("| sampler | target | shelf rate | step amp mean | pos W1 to real |")
    print("|---|---|---:|---:|---:|")
    for k, r in e6l.items():
        s, t = (k.split(" | ") + ["(target)"])[:2]
        print(f"| {s} | {t} | {r['shelf_rate']:.2f} | {r['step_amp_mean']:.3f} | {r['step_pos_w1_to_real']:.3f} |")
p10b = d / "e10b_trained_geo.json"
if p10b.is_file():
    e10b = json.loads(p10b.read_text())
    print("\n### E10b trained bidirectional nets: linear vs geometric schedule (32 bins of context each side)\n")
    print("| net | run | variance ratio | first-difference variance ratio | high-band log10 power shift | start | end |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for k, r in e10b.items():
        net, run = k.split(" | ")
        print(f"| {net} | {run} | {r['var_ratio']:.2f} | {r['diff_var_ratio']:.2f} | {r['hf_logpower_shift']:+.2f} | {r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} |")
