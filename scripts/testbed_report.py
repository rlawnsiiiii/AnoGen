"""Print the markdown tables for docs/TESTBED.md from results/testbed/*.json."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
BAR = "\\|"  # a literal | inside a markdown table cell
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
    print("| sampler | target | shelf rate | step amp mean | step pos sd | pos W1 to real | diff p99.9 | start | end |")
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
p11 = d / "e11_variance_profile.json"
if p11.is_file():
    e11 = json.loads(p11.read_text())
    print("\n### E11 where the error goes: generated / true variance by position (closed form, exact denoisers)\n")
    print("| denoiser, edit | variance first 16 / middle / last 16 | first-difference variance first 16 / middle / last 16 | deviation from donor first 16 / middle |")
    print("|---|---|---|---|")
    for k, r in e11.items():
        v, dv, dev = r["var_ratio"], r["diff_var_ratio"], r["deviation_from_donor"]
        print(f"| {k} | {v['first16']:.2f} / {v['middle']:.2f} / {v['last16']:.2f} | {dv['first16']:.2f} / {dv['middle']:.2f} / {dv['last16']:.2f} | {dev['first16']:.2f} / {dev['middle']:.2f} |")
p12 = d / "e12_phi_smoothing.json"
if p12.is_file():
    e12 = json.loads(p12.read_text())
    print("\n### E12 what ARP in φ rewards: galleries without anomaly content vs an oracle\n")
    print("| gallery | ARP, frozen φ | ARP, standardized φ | ARP, φ without HF bands | diff sd vs donors |")
    print("|---|---:|---:|---:|---:|")
    for k, r in e12.items():
        print(f"| {k} | {r['arp_frozen_phi']:.3f} | {r['arp_standardized_phi']:.3f} | {r['arp_phi_without_hf_bands']:.3f} | {r['diff_sd_ratio']:.2f}× |")
p13 = d / "e13_recurrence.json"
if p13.is_file():
    e13 = json.loads(p13.read_text())
    print("\n### E13 x̂₀-space guidance with trained nets: self-recurrence vs smoothing the edit (level-shift slice)\n")
    print("| denoiser | variant | diff p99.9 | start | end | proto dist. |")
    print("|---|---|---:|---:|---:|---:|")
    for k, r in e13.items():
        a, b = (k.split(" | ") + ["(target)"])[:2]
        pd_ = r.get("proto_dist")
        print(f"| {a} | {b} | {r['diff_p999']:.3f} | {r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} | {'' if pd_ is None else f'{pd_:.2f}'} |")
p14 = d / "e14_masked.json"
if p14.is_file():
    e14 = json.loads(p14.read_text())
    print("\n### E14 masked (RePaint) level shifts vs a whole-window contrast (exact bidirectional denoiser)\n")
    print("| sampler | onset error (bins, median) | amplitude / target | seam: max diff near onset | texture inside | change outside the mask |")
    print("|---|---:|---:|---:|---:|---:|")
    for k, r in e14.items():
        print(f"| {k} | {r['onset_abs_err_median']:.1f} | {r['amp_at_requested_over_target']:.2f} | {r['seam_max_diff_median']:.3f} | {r['texture_inside_ratio']:.2f} | {r['outside_change_mean']:.3f} |")
for name, title in (("e15_detect_pooling.json", "E15 pooling the per-bin denoising error (trained bidirectional net, t = 10, 20, 40; AUROC)"),
                    ("e15_detect_pooling_t60-100-150.json", "E15 the same at t = 60, 100, 150")):
    p15 = d / name
    if p15.is_file():
        e15 = json.loads(p15.read_text())
        print(f"\n### {title}\n")
        print("| anomaly | mean | max16 | top16 | MSMA (mean) | MSMA (max16) |")
        print("|---|---:|---:|---:|---:|---:|")
        for k, r in e15.items():
            print(f"| {k} | {r['mean']:.3f} | {r['max16']:.3f} | {r['top16']:.3f} | {r['msma']:.3f} | {r['msma_max16']:.3f} |")
p15b = d / "e15b_multiscale.json"
if p15b.is_file():
    e15b = json.loads(p15b.read_text())
    cols = list(next(iter(e15b.values())).keys())
    print("\n### E15b one score for every kind: top-16 error per t, z-scored on held-out nominals (AUROC)\n")
    print("| anomaly | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for k, r in e15b.items():
        print(f"| {k} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
p12b = d / "e12b_level_shift_queries.json"
if p12b.is_file():
    e12b = json.loads(p12b.read_text())
    print("\n### E12b level-shift queries only: ARP of the real donors, unchanged or low-passed\n")
    print("| queries | ARP frozen φ: donors / low-pass 1 / low-pass 2 | ARP φ without HF: donors / low-pass 1 / low-pass 2 |")
    print("|---|---|---|")
    for k, r in e12b.items():
        print(f"| {k} | {r['arp_lowpass_0']:.3f} / {r['arp_lowpass_1']:.3f} / {r['arp_lowpass_2']:.3f} | "
              f"{r['arp_noHF_lowpass_0']:.3f} / {r['arp_noHF_lowpass_1']:.3f} / {r['arp_noHF_lowpass_2']:.3f} |")
p16 = d / "e16_vpred.json"
if p16.is_file():
    e16 = json.loads(p16.read_text())
    print("\n### E16 trained bidirectional nets: ε vs v prediction, linear vs log-spaced schedule\n")
    print("| net, schedule | run | level bias (sd) | variance ratio | first-difference variance ratio | start | end |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for k, r in e16.items():
        a, b = k.split(" | ")
        print(f"| {a} | {b} | {r['level_bias_in_sd']:+.2f} | {r['var_ratio']:.2f} | {r['diff_var_ratio']:.2f} | "
              f"{r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} |")
p17 = d / "e17_guidance_decay.json"
if p17.is_file():
    e17 = json.loads(p17.read_text())
    print("\n### E17 guidance weight ∝ σ_t^p (mean 1 over the chain), level-shift prototype slice\n")
    print("| denoiser | final scale, decay p, recurrences | diff p99.9 | start | end | proto dist. |")
    print("|---|---|---:|---:|---:|---:|")
    for k, r in e17.items():
        a, b = (k.split(" | ") + ["(target)"])[:2]
        pdist = r.get("proto_dist")
        print(f"| {a} | {b} | {r['diff_p999']:.3f} | {r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} | "
              f"{'' if pdist is None else f'{pdist:.3f}'} |")
p17b = d / "e17b_decay_diversity.json"
if p17b.is_file():
    e17b = json.loads(p17b.read_text())
    print("\n### E17b does decay collapse samples onto their prototype? (spread of samples sharing a prototype / spread of the donors)\n")
    print("| denoiser | proto weight, decay, recurrences | proto dist. | within-prototype spread / donor spread | diff p99.9 |")
    print("|---|---|---:|---:|---:|")
    for k, r in e17b.items():
        if not isinstance(r, dict):
            continue
        a, b = k.split(" | ")
        print(f"| {a} | {b} | {r['proto_dist']:.3f} | {r['spread_vs_donors']:.3f} | {r['diff_p999']:.3f} |")
p18 = d / "e18_metrics.json"
if p18.is_file():
    e18 = json.loads(p18.read_text())
    print("\n### E18 which metric sees which defect (real anomaly queries against a gallery; C2ST grouped by event)\n")
    print("| gallery | C2ST AUC raw (linear) | C2ST AUC φ | C2ST AUC ROCKET | ARP φ | ARP ROCKET |")
    print("|---|---:|---:|---:|---:|---:|")
    for k, r in e18.items():
        print(f"| {k} | {r['c2st_auc_raw']:.3f} | {r['c2st_auc_phi']:.3f} | {r['c2st_auc_rocket']:.3f} | "
              f"{r['arp_phi']:.3f} | {r['arp_rocket']:.3f} |")
p19 = d / "e19_sigma_max.json"
if p19.is_file():
    e19 = json.loads(p19.read_text())
    print("\n### E19 σ_max from the data: from-noise sampling with the exact bidirectional denoiser\n")
    print("| data | schedule | level bias (sd) | variance ratio | first-difference variance ratio |")
    print("|---|---|---:|---:|---:|")
    for data, rows in e19.items():
        pr = rows.get("predicted", {})
        for k, r in rows.items():
            if k == "predicted":
                continue
            print(f"| {data} | {k} | {r['level_bias_in_sd']:+.3f} | {r['var_ratio']:.2f} | {r['diff_var_ratio']:.2f} |")
        print(f"| {data} | rule (texture.schedule_bounds): σ_min {pr.get('sigma_min', float('nan')):.2g}, "
              f"σ_max {pr.get('sigma_max', float('nan')):.3g} (without DC {pr.get('sigma_max_without_dc', float('nan')):.3g}) | | | |")
p20 = d / "e20_detect_utility.json"
if p20.is_file():
    e20 = json.loads(p20.read_text())
    su = e20["setup"]
    for dt, summ in e20["summary"].items():
        print(f"\n### E20 detector utility ({dt}): 18 real anomalies + up to 255 generated, ROCKET features, "
              f"recall at 1 % FAR ({su['reps']} repetitions × {3 * su['n_test_per_kind']} test anomalies)\n")
        print("| training positives | n generated kept | recall @ 1 % FAR | spike | level shift | drift | AUROC | AP |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm, r in summ.items():
            arm_md = arm.replace("|", BAR)
            print(f"| {arm_md} | {r.get('n_synth', 0):.0f} | {r['recall@1%FAR']:.3f} ± {r['recall_sd_over_reps']:.3f} | "
                  f"{r['recall spike']:.3f} | {r['recall level shift']:.3f} | {r['recall drift']:.3f} | "
                  f"{r['auroc']:.3f} | {r['ap']:.3f} |")
        print(f"\nPaired differences in recall ({dt}). Test-side CI: bootstrap over the pooled test anomalies; "
              "across repetitions: t-interval over the per-repetition differences (also covers the training draw):\n")
        print("| A − B | mean | test-side 95 % CI | across repetitions 95 % CI | spike | level shift | drift | per repetition |")
        print("|---|---:|---|---|---:|---:|---:|---|")
        for name, p in e20["pairs"][dt].items():
            a = p["all"]
            ar = p.get("across_reps") or {}
            ci = f"[{ar['lo']:+.3f}, {ar['hi']:+.3f}]" if ar else ""
            name_md = name.replace("|", BAR)
            print(f"| {name_md} | {a['diff']:+.3f} | [{a['lo']:+.3f}, {a['hi']:+.3f}] | {ci} | "
                  f"{p['spike']['diff']:+.3f} | {p['level shift']['diff']:+.3f} | {p['drift']['diff']:+.3f} | "
                  f"{' '.join(f'{v:+.2f}' for v in p['per_rep'])} |")
    pr = e20.get("pass_rate") or {}
    if pr:
        print("\nShare of generated windows the kNN novelty filter keeps (outside the nominal 99 % set), per kind:\n")
        print("| generator | spike | level shift | drift |")
        print("|---|---:|---:|---:|")
        for g, v in pr.items():
            print(f"| {g} | " + " | ".join(f"{v[k]:.2f}" for k in ("spike", "level shift", "drift")) + " |")
p20b = d / "e20b_shape_checks.json"
if p20b.is_file():
    e20b = json.loads(p20b.read_text())
    print(f"\n### E20b more anomalous is not more useful (logistic, {e20b['reps']} repetitions, recall at 1 % FAR)\n")
    print("| training positives | all | spike | level shift | drift | drift windows outside the nominal set |")
    print("|---|---:|---:|---:|---:|---:|")
    for arm, r in e20b["summary"].items():
        kept = r.get("drift kept by kNN filter")
        print(f"| {arm} | {r['recall@1%FAR']:.3f} | {r['recall spike']:.3f} | {r['recall level shift']:.3f} | "
              f"{r['recall drift']:.3f} | {'' if kept is None else f'{kept:.2f}'} |")
p20c = d / "e20c_detector_strength.json"
if p20c.is_file():
    e20c = json.loads(p20c.read_text())
    print(f"\n### E20c the gain against a better-tuned detector (logistic, penalty C swept, {e20c['reps']} repetitions)\n")
    print("| C, training positives | recall @ 1 % FAR | spike | level shift | drift | AUROC |")
    print("|---|---:|---:|---:|---:|---:|")
    for arm, r in e20c["summary"].items():
        arm_md = arm.replace("|", BAR)
        print(f"| {arm_md} | {r['recall@1%FAR']:.3f} | {r['recall spike']:.3f} | {r['recall level shift']:.3f} | "
              f"{r['recall drift']:.3f} | {r['auroc']:.3f} |")
p20d = d / "e20d_repetitions.json"
if p20d.is_file():
    e20d = json.loads(p20d.read_text())
    print(f"\n### E20d the headline across {e20d['reps']} repetitions (recall at 1 % FAR)\n")
    dets = list(e20d["arms"])
    print("| training positives | " + " | ".join(dets) + " |")
    print("|---|" + "---:|" * len(dets))
    for arm in next(iter(e20d["arms"].values())):
        print(f"| {arm} | " + " | ".join(f"{e20d['arms'][dt][arm]['recall_mean']:.3f} ± {e20d['arms'][dt][arm]['recall_sd_over_reps']:.3f}"
                                       for dt in dets) + " |")
    print("\nPaired differences: mean over repetitions, t-interval and two-level bootstrap (repetitions, then test anomalies), share of repetitions with a gain:\n")
    print("| detector | A − B | mean | t 95 % CI | two-level bootstrap 95 % CI | repetitions with A > B |")
    print("|---|---|---:|---|---|---:|")
    for dt in dets:
        for name, p in e20d["pairs"][dt].items():
            print(f"| {dt} | {name} | {p['mean']:+.3f} | [{p['t_lo']:+.3f}, {p['t_hi']:+.3f}] | "
                  f"[{p['boot_lo']:+.3f}, {p['boot_hi']:+.3f}] | {p['share_of_reps_positive']:.0%} |")
