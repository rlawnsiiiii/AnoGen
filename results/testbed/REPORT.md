### E1 unguided

| start | denoiser | peak@start | peak@end | std ratio start / mid / end | Δ-std ratio start / mid / end |
|---|---|---:|---:|---|---|
| ν=1.0 | bidir | 0.111 | 0.066 | 0.95 / 0.96 / 0.98 | 0.61 / 0.60 / 0.60 |
| ν=1.0 | causal | 0.607 | 0.006 | 0.68 / 0.62 / 0.63 | 1.92 / 0.96 / 0.90 |
| ν=1.0 | flip_avg | 0.254 | 0.232 | 0.59 / 0.64 / 0.61 | 1.22 / 0.89 / 1.22 |
| ν=1.0 | flip_ramp | 0.115 | 0.100 | 0.78 / 0.76 / 0.80 | 0.92 / 0.90 / 0.91 |
| ν=1.0 | causal + 256-bin burn-in | 0.141 | 0.096 | 0.67 / 0.59 / 0.66 | 0.95 / 0.92 / 0.90 |
| ν=0.2 | bidir | 0.100 | 0.072 | 1.02 / 0.95 / 1.06 | 0.68 / 0.69 / 0.69 |
| ν=0.2 | causal | 0.412 | 0.014 | 0.84 / 0.87 / 0.97 | 1.77 / 1.09 / 1.05 |
| ν=0.2 | flip_avg | 0.191 | 0.164 | 0.90 / 0.90 / 0.94 | 1.26 / 1.04 / 1.27 |
| ν=0.2 | flip_ramp | 0.145 | 0.102 | 0.99 / 0.92 / 1.03 | 1.06 / 1.04 / 1.05 |
| ν=0.2 | causal + 256-bin burn-in | 0.115 | 0.086 | 0.94 / 0.84 / 0.98 | 1.08 / 1.06 / 1.06 |
| ν=1.0 | causal + 128-bin burn-in | 0.348 | 0.051 | | |
| ν=0.2 | causal + 128-bin burn-in | 0.176 | 0.090 | | |
| — | real nominal (target) | 0.096 | 0.094 | 1 / 1 / 1 | 1 / 1 / 1 |

x̂₀ error at t=40 (ν=0.2 edit time), MSE first 16 / middle 16 / last 16 bins:

- bidir: 0.0012 / 0.0012 / 0.0011
- causal: 0.0068 / 0.0016 / 0.0012
- anticausal: 0.0011 / 0.0016 / 0.0062
- flip_avg: 0.0025 / 0.0012 / 0.0023
- flip_ramp: 0.0013 / 0.0011 / 0.0013

### E3a guidance VJP

| denoiser | t | flat ∇: mass in first 10 % | centre ∇: centroid of landed edit |
|---|---:|---:|---:|
| bidir | 40 | 0.127 | 0.499 |
| bidir | 20 | 0.137 | 0.500 |
| bidir | 5 | 0.102 | 0.500 |
| causal | 40 | 0.344 | 0.250 |
| causal | 20 | 0.360 | 0.259 |
| causal | 5 | 0.172 | 0.341 |
| flip_ramp | 40 | 0.086 | 0.499 |
| flip_ramp | 20 | 0.126 | 0.499 |
| flip_ramp | 5 | 0.104 | 0.500 |

### E3b band-only slice

| sampler | env exit | diff p99.9 | peak@start | peak@end | position entropy | occupancy |
|---|---:|---:|---:|---:|---:|---:|
| real_nominal | 0.00 | 0.031 | 0.10 | 0.07 | 0.99 | nan |
| repo x final1 | 0.05 | 0.048 | 0.45 | 0.02 | 0.76 | 0.48 |
| x final0 | 0.07 | 0.048 | 0.43 | 0.03 | 0.77 | 0.43 |
| x0 | 0.04 | 0.050 | 0.47 | 0.02 | 0.74 | 0.44 |
| flip_ramp x0 | 0.00 | 0.040 | 0.13 | 0.10 | 1.00 | 0.49 |
| bidir x final1 | 0.01 | 0.028 | 0.09 | 0.07 | 1.00 | 0.41 |
| bidir x0 | 0.02 | 0.029 | 0.10 | 0.10 | 1.00 | 0.38 |

### E2 level-shift proto slice

| sampler | env exit | diff p99.9 | CUSUM | peak@start | peak@end | shelf rate | step pos. mean ± sd | proto dist. |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| real_level_shift | 0.16 | 0.050 | -2.53 | 0.07 | 0.10 | 0.86 | 0.50 ± 0.24 | 0.00 |
| repo: causal, x-space, final kick | 0.08 | 0.163 | -2.51 | 0.27 | 0.04 | 1.00 | 0.46 ± 0.19 | 0.17 |
| causal, x-space, no final kick | 0.07 | 0.118 | -2.52 | 0.18 | 0.03 | 1.00 | 0.47 ± 0.19 | 0.21 |
| causal, x-space, final 0.25 | 0.00 | 0.065 | -2.52 | 0.18 | 0.03 | 1.00 | 0.46 ± 0.19 | 0.13 |
| causal, x0-space | 0.05 | 0.079 | -2.57 | 0.04 | 0.02 | 1.00 | 0.50 ± 0.21 | 0.18 |
| causal, x0-space, final 0.25 | 0.00 | 0.064 | -2.51 | 0.08 | 0.02 | 1.00 | 0.47 ± 0.19 | 0.09 |
| causal, eps-space | 0.00 | 0.042 | +0.04 | 0.41 | 0.03 | 0.00 | 0.56 ± 0.37 | 1.78 |
| causal, eps-space lam x3 | 0.00 | 0.042 | +0.04 | 0.41 | 0.03 | 0.00 | 0.57 ± 0.37 | 1.76 |
| flip_ramp, x0-space | 0.20 | 0.063 | -2.62 | 0.03 | 0.17 | 1.00 | 0.48 ± 0.20 | 0.19 |
| flip_ramp, x0-space, final 0.25 | 0.00 | 0.058 | -2.49 | 0.04 | 0.14 | 1.00 | 0.48 ± 0.20 | 0.09 |
| causal+burnin, x0-space | 0.14 | 0.081 | -2.57 | 0.05 | 0.20 | 1.00 | 0.48 ± 0.20 | 0.19 |
| bidir, x-space, final kick | 0.01 | 0.041 | -2.58 | 0.08 | 0.18 | 1.00 | 0.48 ± 0.20 | 0.18 |
| bidir, x0-space | 0.13 | 0.045 | -2.73 | 0.00 | 0.15 | 1.00 | 0.49 ± 0.20 | 0.22 |
| bidir, eps-space lam x3 | 0.00 | 0.020 | +0.00 | 0.10 | 0.09 | 0.00 | 0.49 ± 0.36 | 1.80 |
| bidir, x0-space, proto shift | 0.12 | 0.043 | -2.07 | 0.10 | 0.21 | 0.70 | 0.45 ± 0.23 | 0.19 |
| flip_ramp, x0-space, proto shift | 0.13 | 0.058 | -2.04 | 0.09 | 0.17 | 0.71 | 0.45 ± 0.23 | 0.18 |

### E5 from-noise start

| data mean | denoiser | start | generated mean | level bias (sd) |
|---:|---|---|---:|---:|
| 0.5 | bidir | N(0,I) | 0.477 | -0.20 |
| 0.5 | bidir | matched | 0.500 | +0.00 |
| 0.5 | causal | N(0,I) | 0.483 | -0.16 |
| 0.5 | causal | matched | 0.500 | -0.00 |
| 0.8 | bidir | N(0,I) | 0.764 | -0.32 |
| 0.8 | bidir | matched | 0.800 | +0.00 |
| 0.8 | causal | N(0,I) | 0.772 | -0.25 |
| 0.8 | causal | matched | 0.800 | -0.00 |

### E6 level shifts: 6 shared prototypes vs a parametric step contrast

| sampler | target | shelf rate | step amp mean | step pos sd | pos W1 to real | diff p99.9 | start | end |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| real level shifts | (target) | 0.90 | 0.280 | 0.315 | 0.000 | 0.050 | 0.08 | 0.09 |
| repo (causal, x, final 1) | proto (6 shared) | 1.00 | 0.262 | 0.261 | 0.075 | 0.150 | 0.28 | 0.03 |
| repo (causal, x, final 1) | step contrast | 0.84 | 0.325 | 0.305 | 0.132 | 0.056 | 0.27 | 0.01 |
| repo (causal, x, final 1) | proto + step contrast | 0.99 | 0.302 | 0.262 | 0.074 | 0.138 | 0.10 | 0.00 |
| fixed (flip_ramp, x0, final 0.25) | proto (6 shared) | 1.00 | 0.271 | 0.261 | 0.076 | 0.068 | 0.02 | 0.07 |
| fixed (flip_ramp, x0, final 0.25) | step contrast | 0.75 | 0.302 | 0.301 | 0.021 | 0.047 | 0.14 | 0.10 |
| fixed (flip_ramp, x0, final 0.25) | proto + step contrast | 1.00 | 0.293 | 0.261 | 0.075 | 0.083 | 0.02 | 0.03 |

### E7 what φ sees

Share of the pairwise squared φ distance between real nominal windows, by component:

| z_mean | z_std | z_min | z_max | dz_mean | dz_std | peak_abs_z | longest_excursion | fft_band1 | fft_band2 | fft_band3 | fft_band4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.000 | 0.080 | 0.079 | 0.000 | 0.001 | 0.060 | 0.000 | 0.000 | 0.262 | 0.262 | 0.256 |

Unguided regeneration of real nominal windows (no steering): mean shift of the three
high-frequency band energies, in real-nominal standard deviations:

| gallery | fft_band2 | fft_band3 | fft_band4 | ARP vs held-out real nominals |
|---|---:|---:|---:|---:|
| bidir nu=1.0 steps=20 | -1.12 | -1.37 | -1.55 | 0.756 |
| bidir nu=1.0 steps=100 | -0.99 | -1.36 | -1.57 | 0.756 |
| bidir nu=0.2 steps=50 | -0.89 | -1.28 | -1.48 | 0.744 |
| causal nu=1.0 steps=20 | +1.20 | +1.38 | +1.42 | 0.757 |
| causal nu=1.0 steps=100 | +1.30 | +1.46 | +1.49 | 0.750 |
| causal nu=0.2 steps=50 | +0.55 | +0.70 | +0.72 | 0.793 |
| donor (real nominal) | +0.00 | +0.00 | +0.00 | 0.811 |

### E10 texture the schedule can reproduce (closed form, exact bidirectional denoiser)

Generated / true variance, total and of first differences (texture), deterministic DDIM from t = T−1:

| texture sd | schedule | steps | variance | first-difference variance |
|---:|---|---:|---:|---:|
| 0.02 | linear (frozen; sigma 0.010..2.56) | 20 | 0.89 | 0.36 |
| 0.02 | linear (frozen; sigma 0.010..2.56) | 50 | 0.95 | 0.42 |
| 0.02 | linear (frozen; sigma 0.010..2.56) | 200 | 0.99 | 0.48 |
| 0.02 | geometric sigma 0.001..2.56 | 20 | 0.83 | 0.80 |
| 0.02 | geometric sigma 0.001..2.56 | 50 | 0.93 | 0.91 |
| 0.02 | geometric sigma 0.001..2.56 | 200 | 0.98 | 0.96 |
| 0.02 | geometric sigma 0.001..10 | 20 | 0.78 | 0.77 |
| 0.02 | geometric sigma 0.001..10 | 50 | 0.91 | 0.89 |
| 0.02 | geometric sigma 0.001..10 | 200 | 0.98 | 0.96 |
| 0.02 | geometric sigma 0.001..80 | 20 | 0.74 | 0.73 |
| 0.02 | geometric sigma 0.001..80 | 50 | 0.89 | 0.88 |
| 0.02 | geometric sigma 0.001..80 | 200 | 0.97 | 0.95 |
| 0.05 | linear (frozen; sigma 0.010..2.56) | 20 | 0.87 | 0.37 |
| 0.05 | linear (frozen; sigma 0.010..2.56) | 50 | 0.94 | 0.52 |
| 0.05 | linear (frozen; sigma 0.010..2.56) | 200 | 0.98 | 0.68 |
| 0.05 | geometric sigma 0.001..2.56 | 20 | 0.83 | 0.81 |
| 0.05 | geometric sigma 0.001..2.56 | 50 | 0.93 | 0.92 |
| 0.05 | geometric sigma 0.001..2.56 | 200 | 0.98 | 0.98 |
| 0.05 | geometric sigma 0.001..10 | 20 | 0.78 | 0.78 |
| 0.05 | geometric sigma 0.001..10 | 50 | 0.91 | 0.91 |
| 0.05 | geometric sigma 0.001..10 | 200 | 0.98 | 0.97 |
| 0.05 | geometric sigma 0.001..80 | 20 | 0.74 | 0.74 |
| 0.05 | geometric sigma 0.001..80 | 50 | 0.89 | 0.89 |
| 0.05 | geometric sigma 0.001..80 | 200 | 0.97 | 0.97 |

### E8 trained nonlinear denoisers (numpy dilated CNN, causal vs 'same' padding, same size)

| denoiser | x̂₀ MSE first 16 / middle / last 16 (t=40) | guidance VJP centroid | unguided ν=1 start / end | ν=0.2 start / end |
|---|---|---:|---|---|
| causal net | 0.0064 / 0.0034 / 0.0030 | 0.42 | 0.746 / 0.047 | 0.871 / 0.020 |
| bidir net | 0.0041 / 0.0026 / 0.0036 | 0.50 | 0.352 / 0.316 | 0.344 / 0.324 |
| causal net + flip ramp | 0.0032 / 0.0021 / 0.0036 |  | 0.102 / 0.133 | 0.098 / 0.105 |

Level-shift prototype slice with the trained nets:

| sampler | start | end | diff p99.9 |
|---|---:|---:|---:|
| causal net, x-space, final 1.0 | 0.34 | 0.03 | 0.083 |
| causal net, x0-space, final 0.25 | 0.07 | 0.05 | 0.128 |
| causal net + flip ramp, x0-space, final 0.25 | 0.02 | 0.06 | 0.114 |
| bidir net, x-space, final 1.0 | 0.06 | 0.08 | 0.075 |
| bidir net, x0-space, final 0.25 | 0.07 | 0.07 | 0.087 |

Label-free detection by denoising error (spikes of 0.15 at uniform positions vs nominal):

| denoiser | AUROC | spike in first 10 % | spike elsewhere |
|---|---:|---:|---:|
| causal net | 0.634 | 0.660 | 0.632 |
| causal net, skip first 32 | 0.637 | 0.605 | 0.641 |
| bidir net | 0.854 | 0.865 | 0.853 |

### E8c trained nets with real context around the window (generate W + context, keep W)

| configuration | start | end |
|---|---:|---:|
| causal net, context 0 bins (none), nu=1.0 | 0.766 | 0.047 |
| causal net, context 0 bins (none), nu=0.2 | 0.870 | 0.021 |
| bidir net, context 0 bins (none), nu=1.0 | 0.349 | 0.292 |
| bidir net, context 0 bins (none), nu=0.2 | 0.354 | 0.359 |
| causal net, context 32 bins (start), nu=1.0 | 0.135 | 0.151 |
| causal net, context 32 bins (start), nu=0.2 | 0.125 | 0.125 |
| causal net, context 32 bins (both), nu=1.0 | 0.115 | 0.073 |
| causal net, context 32 bins (both), nu=0.2 | 0.094 | 0.089 |
| bidir net, context 32 bins (both), nu=1.0 | 0.104 | 0.078 |
| bidir net, context 32 bins (both), nu=0.2 | 0.104 | 0.104 |
| causal net, context 64 bins (start), nu=1.0 | 0.068 | 0.083 |
| causal net, context 64 bins (start), nu=0.2 | 0.062 | 0.115 |
| causal net, context 64 bins (both), nu=1.0 | 0.120 | 0.130 |
| causal net, context 64 bins (both), nu=0.2 | 0.068 | 0.120 |
| bidir net, context 64 bins (both), nu=1.0 | 0.089 | 0.125 |
| bidir net, context 64 bins (both), nu=0.2 | 0.073 | 0.109 |
| causal net, context 128 bins (start), nu=1.0 | 0.089 | 0.109 |
| causal net, context 128 bins (start), nu=0.2 | 0.094 | 0.125 |
| causal net, context 128 bins (both), nu=1.0 | 0.104 | 0.068 |
| causal net, context 128 bins (both), nu=0.2 | 0.089 | 0.099 |
| bidir net, context 128 bins (both), nu=1.0 | 0.120 | 0.104 |
| bidir net, context 128 bins (both), nu=0.2 | 0.130 | 0.104 |

### E2b leaving the end of the trajectory unguided (level-shift slice, exact denoisers, x̂₀-space)

| denoiser | final scale | guided t / t* window | diff p99.9 | start | end | proto dist. |
|---|---:|---|---:|---:|---:|---:|
| flip_ramp | 1.0 | (0, 1) | 0.069 | 0.06 | 0.15 | 0.18 |
| flip_ramp | 0.25 | (0, 1) | 0.069 | 0.08 | 0.14 | 0.11 |
| flip_ramp | 1.0 | (0.1, 1) | 0.099 | 0.02 | 0.14 | 0.24 |
| flip_ramp | 1.0 | (0.2, 1) | 0.112 | 0.00 | 0.14 | 0.49 |
| flip_ramp | 0.0 | (0.1, 1) | 0.099 | 0.02 | 0.14 | 0.24 |
| bidir | 1.0 | (0, 1) | 0.044 | 0.11 | 0.03 | 0.17 |
| bidir | 0.25 | (0, 1) | 0.045 | 0.14 | 0.03 | 0.12 |
| bidir | 1.0 | (0.1, 1) | 0.070 | 0.06 | 0.01 | 0.23 |
| bidir | 1.0 | (0.2, 1) | 0.094 | 0.00 | 0.00 | 0.41 |
| bidir | 0.0 | (0.1, 1) | 0.070 | 0.06 | 0.01 | 0.23 |

### E6b ±64-bin step contrast instead of whole-window segments

| sampler | target | shelf rate | step amp mean | pos W1 to real |
|---|---|---:|---:|---:|
| real level shifts | (target) | 0.90 | 0.280 | 0.000 |
| repo (causal, x, final 1) | proto (6 shared) | 1.00 | 0.262 | 0.075 |
| repo (causal, x, final 1) | step contrast | 0.05 | 0.042 | 0.130 |
| repo (causal, x, final 1) | proto + step contrast | 0.99 | 0.273 | 0.064 |
| fixed (flip_ramp, x0, final 0.25) | proto (6 shared) | 1.00 | 0.271 | 0.076 |
| fixed (flip_ramp, x0, final 0.25) | step contrast | 0.00 | 0.042 | 0.121 |
| fixed (flip_ramp, x0, final 0.25) | proto + step contrast | 0.99 | 0.271 | 0.066 |

### E10b trained bidirectional nets: linear vs geometric schedule (32 bins of context each side)

| net | run | variance ratio | first-difference variance ratio | high-band log10 power shift | start | end |
|---|---|---:|---:|---:|---:|---:|
| bidir net, linear schedule | from noise, 20 steps | 0.84 | 1.24 | +0.06 | 0.09 | 0.08 |
| bidir net, linear schedule | from noise, 50 steps | 0.91 | 1.43 | +0.11 | 0.11 | 0.08 |
| bidir net, linear schedule | edit at nu=0.2 noise level, 50 steps | 1.02 | 0.98 | +0.05 | 0.09 | 0.09 |
| bidir net, geometric schedule | from noise, 20 steps | 0.71 | 0.79 | -0.21 | 0.10 | 0.09 |
| bidir net, geometric schedule | from noise, 50 steps | 0.78 | 0.93 | -0.15 | 0.09 | 0.07 |
| bidir net, geometric schedule | edit at nu=0.2 noise level, 50 steps | 0.97 | 0.95 | -0.03 | 0.06 | 0.09 |

### E11 where the error goes: generated / true variance by position (closed form, exact denoisers)

| denoiser, edit | variance first 16 / middle / last 16 | first-difference variance first 16 / middle / last 16 | deviation from donor first 16 / middle |
|---|---|---|---|
| bidir nu=1.0 | 0.92 / 0.92 / 0.92 | 0.37 / 0.37 / 0.37 | 1.01 / 1.01 |
| bidir nu=0.2 | 0.99 / 0.99 / 0.99 | 0.48 / 0.48 / 0.48 | 0.11 / 0.10 |
| causal nu=1.0 | 0.47 / 0.41 / 0.40 | 4.19 / 0.92 / 0.83 | 1.32 / 1.01 |
| causal nu=0.2 | 0.69 / 0.83 / 0.84 | 3.44 / 1.19 / 1.13 | 0.61 / 0.17 |
| flip_ramp nu=1.0 | 0.62 / 0.59 / 0.62 | 0.86 / 0.82 / 0.86 | 0.87 / 0.86 |
| flip_ramp nu=0.2 | 0.93 / 0.93 / 0.93 | 1.14 / 1.09 / 1.14 | 0.10 / 0.10 |

### E12 what ARP in φ rewards: galleries without anomaly content vs an oracle

| gallery | ARP, frozen φ | ARP, standardized φ | ARP, φ without HF bands | diff sd vs donors |
|---|---:|---:|---:|---:|
| donors, low-pass sd 0 bins | 0.301 | 0.094 | 0.339 | 1.00× |
| donors, low-pass sd 0.5 bins | 0.289 | 0.087 | 0.337 | 0.83× |
| donors, low-pass sd 1 bins | 0.263 | 0.077 | 0.330 | 0.62× |
| donors, low-pass sd 1.5 bins | 0.256 | 0.074 | 0.326 | 0.55× |
| donors, low-pass sd 2 bins | 0.251 | 0.073 | 0.323 | 0.52× |
| donors, low-pass sd 3 bins | 0.251 | 0.073 | 0.317 | 0.48× |
| donors + white texture sd 0.01 | 0.279 | 0.105 | 0.346 | 1.73× |
| donors + white texture sd 0.02 | 0.249 | 0.100 | 0.346 | 2.98× |
| oracle: fresh anomalies, same generator | 0.748 | 0.356 | 0.881 | 3.37× |
| oracle, low-pass sd 1.5 bins | 0.341 | 0.105 | 0.821 | 1.28× |

### E13 x̂₀-space guidance with trained nets: self-recurrence vs smoothing the edit (level-shift slice)

| denoiser | variant | diff p99.9 | start | end | proto dist. |
|---|---|---:|---:|---:|---:|
| real level shifts | (target) | 0.051 | 0.04 | 0.07 |  |
| bidir net | x0, final 0.25 | 0.076 | 0.18 | 0.05 | 0.10 |
| bidir net | x0, final 0 | 0.076 | 0.17 | 0.05 | 0.19 |
| bidir net | x0, recurrence k=2 | 0.060 | 0.11 | 0.05 | 0.10 |
| bidir net | x0, recurrence k=4 | 0.057 | 0.14 | 0.03 | 0.10 |
| bidir net | x0, edit low-pass sd 4 | 0.056 | 0.33 | 0.39 | 0.10 |
| bidir net | x0, edit low-pass sd 8 | 0.052 | 0.37 | 0.42 | 0.12 |
| bidir net | x0, recurrence k=2 + low-pass sd 4 | 0.050 | 0.32 | 0.41 | 0.10 |
| causal net + flip ramp | x0, final 0.25 | 0.107 | 0.06 | 0.07 | 0.10 |
| causal net + flip ramp | x0, final 0 | 0.106 | 0.07 | 0.06 | 0.18 |
| causal net + flip ramp | x0, recurrence k=2 | 0.078 | 0.05 | 0.12 | 0.10 |
| causal net + flip ramp | x0, recurrence k=4 | 0.073 | 0.10 | 0.09 | 0.09 |
| causal net + flip ramp | x0, edit low-pass sd 4 | 0.073 | 0.05 | 0.20 | 0.10 |
| causal net + flip ramp | x0, edit low-pass sd 8 | 0.069 | 0.05 | 0.26 | 0.12 |
| causal net + flip ramp | x0, recurrence k=2 + low-pass sd 4 | 0.062 | 0.06 | 0.17 | 0.11 |

### E14 masked (RePaint) level shifts vs a whole-window contrast (exact bidirectional denoiser)

| sampler | onset error (bins, median) | amplitude / target | seam: max diff near onset | texture inside | change outside the mask |
|---|---:|---:|---:|---:|---:|
| real level shifts (ramp 6 bins) | 4.5 | 0.93 | 0.051 | 1.00 | 0.000 |
| whole-window contrast, nu 0.2 (E6 style) | 5.0 | 1.10 | 0.022 | 0.73 | 0.062 |
| masked, nu 0.2 | 4.0 | 1.07 | 0.080 | 0.79 | 0.000 |
| masked, nu 0.2, resample 3 | 6.5 | 1.12 | 0.065 | 0.86 | 0.000 |
| masked + 8-bin dilation, nu 0.2 | 14.0 | 0.99 | 0.074 | 0.89 | 0.000 |
| masked + 8-bin dilation, nu 0.5, resample 3 | 18.0 | 1.00 | 0.058 | 1.03 | 0.000 |
| masked, nu 0.5 | 5.0 | 1.12 | 0.079 | 0.86 | 0.000 |
| masked, nu 0.5, resample 3 | 7.0 | 1.13 | 0.068 | 0.91 | 0.000 |
| masked, from t_max | 6.0 | 1.14 | 0.079 | 0.87 | 0.000 |
| masked, from t_max, resample 3 | 6.0 | 1.13 | 0.085 | 0.88 | 0.000 |

### E15 pooling the per-bin denoising error (trained bidirectional net, t = 10, 20, 40; AUROC)

| anomaly | mean | max16 | top16 | MSMA (mean) | MSMA (max16) |
|---|---:|---:|---:|---:|---:|
| spike 0.15, 3 bins | 0.855 | 0.938 | 0.997 | 0.748 | 0.937 |
| level shift 0.10 | 0.628 | 0.619 | 0.661 | 0.500 | 0.504 |
| drift 0.15 over 128 bins | 0.583 | 0.629 | 0.603 | 0.497 | 0.581 |

### E15 the same at t = 60, 100, 150

| anomaly | mean | max16 | top16 | MSMA (mean) | MSMA (max16) |
|---|---:|---:|---:|---:|---:|
| spike 0.15, 3 bins | 0.572 | 0.569 | 0.660 | 0.502 | 0.509 |
| level shift 0.10 | 0.647 | 0.679 | 0.680 | 0.486 | 0.582 |
| drift 0.15 over 128 bins | 0.743 | 0.774 | 0.774 | 0.666 | 0.734 |

### E15b one score for every kind: top-16 error per t, z-scored on held-out nominals (AUROC)

| anomaly | t=10 | t=20 | t=40 | t=60 | t=100 | t=150 | z-max over t | z-mean over t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spike 0.15, 3 bins | 0.999 | 0.979 | 0.776 | 0.658 | 0.603 | 0.586 | 0.995 | 0.947 |
| level shift 0.10 | 0.647 | 0.644 | 0.572 | 0.587 | 0.694 | 0.710 | 0.667 | 0.694 |
| drift 0.15 over 128 bins | 0.575 | 0.586 | 0.590 | 0.640 | 0.790 | 0.832 | 0.749 | 0.744 |

### E12b level-shift queries only: ARP of the real donors, unchanged or low-passed

| queries | ARP frozen φ: donors / low-pass 1 / low-pass 2 | ARP φ without HF: donors / low-pass 1 / low-pass 2 |
|---|---|---|
| level shifts only, size 0.25 | 0.761 / 0.725 / 0.766 | 0.906 / 0.911 / 0.906 |
| level shifts only, size 0.5 | 0.714 / 0.711 / 0.740 | 0.846 / 0.855 / 0.858 |

### E16 trained bidirectional nets: ε vs v prediction, linear vs log-spaced schedule

| net, schedule | run | level bias (sd) | variance ratio | first-difference variance ratio | start | end |
|---|---|---:|---:|---:|---:|---:|
| ε-net, linear schedule | from noise, 20 steps | -1.69 | 0.84 | 1.24 | 0.09 | 0.08 |
| ε-net, linear schedule | from noise, 50 steps | -1.75 | 0.91 | 1.43 | 0.11 | 0.08 |
| ε-net, linear schedule | from matched noise, 50 steps | -0.23 | 0.78 | 0.81 | 0.11 | 0.09 |
| ε-net, linear schedule | edit at the ν = 0.2 noise level, 50 steps | -0.06 | 1.02 | 0.98 | 0.09 | 0.09 |
| ε-net, log-spaced schedule | from noise, 20 steps | -0.65 | 0.71 | 0.79 | 0.10 | 0.09 |
| ε-net, log-spaced schedule | from noise, 50 steps | -0.65 | 0.78 | 0.93 | 0.09 | 0.07 |
| ε-net, log-spaced schedule | from matched noise, 50 steps | -0.29 | 0.77 | 0.92 | 0.09 | 0.07 |
| ε-net, log-spaced schedule | edit at the ν = 0.2 noise level, 50 steps | -0.04 | 0.97 | 0.95 | 0.06 | 0.09 |
| v-net, log-spaced schedule | from noise, 20 steps | -0.48 | 0.65 | 0.79 | 0.11 | 0.08 |
| v-net, log-spaced schedule | from noise, 50 steps | -0.52 | 0.75 | 0.93 | 0.11 | 0.09 |
| v-net, log-spaced schedule | from matched noise, 50 steps | -0.16 | 0.74 | 0.92 | 0.11 | 0.09 |
| v-net, log-spaced schedule | edit at the ν = 0.2 noise level, 50 steps | -0.04 | 0.95 | 0.96 | 0.06 | 0.07 |

### E17 guidance weight ∝ σ_t^p (mean 1 over the chain), level-shift prototype slice

| denoiser | final scale, decay p, recurrences | diff p99.9 | start | end | proto dist. |
|---|---|---:|---:|---:|---:|
| real level shifts | (target) | 0.051 | 0.04 | 0.07 |  |
| exact flip ramp | final 1.0, decay 0.0, recur 1 | 0.073 | 0.09 | 0.08 | 0.183 |
| exact flip ramp | final 1.0, decay 0.5, recur 1 | 0.075 | 0.08 | 0.08 | 0.044 |
| exact flip ramp | final 1.0, decay 1.0, recur 1 | 0.074 | 0.11 | 0.12 | 0.025 |
| exact flip ramp | final 1.0, decay 2.0, recur 1 | 0.102 | 0.02 | 0.01 | 0.853 |
| exact flip ramp | final 0.25, decay 0.0, recur 1 | 0.069 | 0.10 | 0.09 | 0.092 |
| exact flip ramp | final 0.25, decay 0.0, recur 2 | 0.051 | 0.13 | 0.12 | 0.158 |
| exact flip ramp | final 0.25, decay 0.5, recur 1 | 0.071 | 0.11 | 0.08 | 0.021 |
| exact flip ramp | final 0.25, decay 0.5, recur 2 | 0.050 | 0.12 | 0.11 | 0.041 |
| exact flip ramp | final 0.25, decay 1.0, recur 1 | 0.078 | 0.09 | 0.10 | 0.022 |
| exact flip ramp | final 0.25, decay 1.0, recur 2 | 0.050 | 0.09 | 0.09 | 0.051 |
| exact flip ramp | final 0.25, decay 2.0, recur 1 | 0.102 | 0.02 | 0.01 | 0.854 |
| exact flip ramp | final 0.25, decay 2.0, recur 2 | 0.042 | 0.12 | 0.08 | 0.267 |
| exact bidirectional | final 1.0, decay 0.0, recur 1 | 0.050 | 0.00 | 0.09 | 0.174 |
| exact bidirectional | final 1.0, decay 0.5, recur 1 | 0.048 | 0.00 | 0.03 | 0.058 |
| exact bidirectional | final 1.0, decay 1.0, recur 1 | 0.050 | 0.00 | 0.05 | 0.011 |
| exact bidirectional | final 1.0, decay 2.0, recur 1 | 0.094 | 0.00 | 0.00 | 0.296 |
| exact bidirectional | final 0.25, decay 0.0, recur 1 | 0.048 | 0.00 | 0.09 | 0.113 |
| exact bidirectional | final 0.25, decay 0.0, recur 2 | 0.044 | 0.01 | 0.04 | 0.092 |
| exact bidirectional | final 0.25, decay 0.5, recur 1 | 0.050 | 0.00 | 0.10 | 0.013 |
| exact bidirectional | final 0.25, decay 0.5, recur 2 | 0.044 | 0.01 | 0.01 | 0.031 |
| exact bidirectional | final 0.25, decay 1.0, recur 1 | 0.053 | 0.00 | 0.11 | 0.017 |
| exact bidirectional | final 0.25, decay 1.0, recur 2 | 0.045 | 0.01 | 0.00 | 0.005 |
| exact bidirectional | final 0.25, decay 2.0, recur 1 | 0.094 | 0.00 | 0.00 | 0.297 |
| exact bidirectional | final 0.25, decay 2.0, recur 2 | 0.041 | 0.05 | 0.04 | 0.065 |
| trained bidirectional net | final 1.0, decay 0.0, recur 1 | 0.076 | 0.16 | 0.05 | 0.196 |
| trained bidirectional net | final 1.0, decay 0.5, recur 1 | 0.081 | 0.12 | 0.05 | 0.043 |
| trained bidirectional net | final 1.0, decay 1.0, recur 1 | 0.080 | 0.13 | 0.05 | 0.009 |
| trained bidirectional net | final 1.0, decay 2.0, recur 1 | 0.087 | 0.12 | 0.03 | 0.026 |
| trained bidirectional net | final 0.25, decay 0.0, recur 1 | 0.076 | 0.18 | 0.05 | 0.100 |
| trained bidirectional net | final 0.25, decay 0.0, recur 2 | 0.056 | 0.10 | 0.08 | 0.110 |
| trained bidirectional net | final 0.25, decay 0.5, recur 1 | 0.079 | 0.12 | 0.06 | 0.024 |
| trained bidirectional net | final 0.25, decay 0.5, recur 2 | 0.058 | 0.08 | 0.07 | 0.018 |
| trained bidirectional net | final 0.25, decay 1.0, recur 1 | 0.081 | 0.12 | 0.03 | 0.006 |
| trained bidirectional net | final 0.25, decay 1.0, recur 2 | 0.059 | 0.09 | 0.06 | 0.002 |
| trained bidirectional net | final 0.25, decay 2.0, recur 1 | 0.087 | 0.12 | 0.03 | 0.027 |
| trained bidirectional net | final 0.25, decay 2.0, recur 2 | 0.047 | 0.36 | 0.27 | 0.081 |
| trained causal net + flip ramp | final 1.0, decay 0.0, recur 1 | 0.109 | 0.07 | 0.06 | 0.185 |

### E17b does decay collapse samples onto their prototype? (spread of samples sharing a prototype / spread of the donors)

| denoiser | proto weight, decay, recurrences | proto dist. | within-prototype spread / donor spread | diff p99.9 |
|---|---|---:|---:|---:|
| exact flip ramp | proto weight 1.0, decay 0.0, recur 1 | 0.092 | 0.106 | 0.069 |
| exact flip ramp | proto weight 1.0, decay 0.0, recur 2 | 0.158 | 0.150 | 0.051 |
| exact flip ramp | proto weight 1.0, decay 1.0, recur 1 | 0.022 | 0.088 | 0.078 |
| exact flip ramp | proto weight 1.0, decay 1.0, recur 2 | 0.051 | 0.078 | 0.050 |
| exact flip ramp | proto weight 0.5, decay 0.0, recur 1 | 0.061 | 0.094 | 0.062 |
| exact flip ramp | proto weight 0.5, decay 0.0, recur 2 | 0.055 | 0.081 | 0.049 |
| exact flip ramp | proto weight 0.5, decay 1.0, recur 1 | 0.056 | 0.091 | 0.064 |
| exact flip ramp | proto weight 0.5, decay 1.0, recur 2 | 0.227 | 0.096 | 0.042 |
| trained bidirectional net | proto weight 1.0, decay 0.0, recur 1 | 0.100 | 0.131 | 0.076 |
| trained bidirectional net | proto weight 1.0, decay 0.0, recur 2 | 0.110 | 0.131 | 0.056 |
| trained bidirectional net | proto weight 1.0, decay 1.0, recur 1 | 0.006 | 0.087 | 0.081 |
| trained bidirectional net | proto weight 1.0, decay 1.0, recur 2 | 0.002 | 0.071 | 0.059 |
| trained bidirectional net | proto weight 0.5, decay 0.0, recur 1 | 0.063 | 0.084 | 0.069 |
| trained bidirectional net | proto weight 0.5, decay 0.0, recur 2 | 0.057 | 0.077 | 0.055 |
| trained bidirectional net | proto weight 0.5, decay 1.0, recur 1 | 0.016 | 0.089 | 0.077 |
| trained bidirectional net | proto weight 0.5, decay 1.0, recur 2 | 0.047 | 0.079 | 0.053 |

### E18 which metric sees which defect (real anomaly queries against a gallery; C2ST grouped by event)

| gallery | C2ST AUC raw (linear) | C2ST AUC φ | C2ST AUC ROCKET | ARP φ | ARP ROCKET |
|---|---:|---:|---:|---:|---:|
| null: same generator, new events | 0.501 | 0.538 | 0.515 | 0.726 | 0.028 |
| nominal windows, no anomaly | 0.511 | 0.973 | 0.993 | 0.454 | 0.006 |
| texture smoothed (low-pass sd 1.5 bins) | 0.517 | 0.979 | 1.000 | 0.475 | 0.009 |
| amplitude x1.6 | 0.468 | 0.701 | 0.789 | 0.708 | 0.027 |
| all events in the first 10 % of the window | 0.507 | 0.765 | 0.975 | 0.715 | 0.029 |

### E19 σ_max from the data: from-noise sampling with the exact bidirectional denoiser

| data | schedule | level bias (sd) | variance ratio | first-difference variance ratio |
|---|---|---:|---:|---:|
| min/max-style data (mean 0.5) | linear (frozen, sigma_max 2.56) | -0.260 | 0.95 | 0.46 |
| min/max-style data (mean 0.5) | geometric sigma_max 2.56 | -0.267 | 0.92 | 1.03 |
| min/max-style data (mean 0.5) | geometric sigma_max 10 | -0.071 | 0.91 | 0.90 |
| min/max-style data (mean 0.5) | geometric sigma_max 30 | -0.024 | 0.90 | 0.89 |
| min/max-style data (mean 0.5) | geometric sigma_max 100 | -0.007 | 0.89 | 0.87 |
| min/max-style data (mean 0.5) | rule (texture.schedule_bounds): σ_min 0.0016, σ_max 80.1 (without DC 9.7) | | | |
| centred data (mean 0) | linear (frozen, sigma_max 2.56) | +0.000 | 0.95 | 0.46 |
| centred data (mean 0) | geometric sigma_max 2.56 | +0.000 | 0.92 | 1.03 |
| centred data (mean 0) | geometric sigma_max 10 | +0.000 | 0.91 | 0.90 |
| centred data (mean 0) | geometric sigma_max 30 | +0.000 | 0.90 | 0.89 |
| centred data (mean 0) | geometric sigma_max 100 | +0.000 | 0.89 | 0.87 |
| centred data (mean 0) | rule (texture.schedule_bounds): σ_min 0.0016, σ_max 9.7 (without DC 9.7) | | | |

### E20 detector utility (ridge): 18 real anomalies + up to 255 generated, ROCKET features, recall at 1 % FAR (5 repetitions × 900 test anomalies)

| training positives | n generated kept | recall @ 1 % FAR | spike | level shift | drift | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| real_only | 0 | 0.594 ± 0.058 | 0.888 | 0.395 | 0.499 | 0.781 | 0.769 |
| +posthoc | 255 | 0.436 ± 0.033 | 0.865 | 0.139 | 0.303 | 0.798 | 0.716 |
| +posthoc | kNN filter | 201 | 0.464 ± 0.032 | 0.831 | 0.208 | 0.352 | 0.810 | 0.734 |
| +posthoc | weighted | 255 | 0.489 ± 0.024 | 0.872 | 0.219 | 0.377 | 0.818 | 0.752 |
| +unguided | 255 | 0.444 ± 0.091 | 0.859 | 0.229 | 0.242 | 0.719 | 0.671 |
| +unguided | kNN filter | 41 | 0.519 ± 0.067 | 0.835 | 0.364 | 0.359 | 0.771 | 0.738 |
| +unguided | weighted | 255 | 0.467 ± 0.076 | 0.849 | 0.263 | 0.289 | 0.739 | 0.693 |
| +C1-like | 255 | 0.569 ± 0.045 | 0.829 | 0.397 | 0.482 | 0.781 | 0.760 |
| +C1-like | kNN filter | 253 | 0.566 ± 0.049 | 0.827 | 0.393 | 0.477 | 0.782 | 0.759 |
| +C1-like | weighted | 255 | 0.562 ± 0.050 | 0.816 | 0.390 | 0.481 | 0.782 | 0.760 |
| +twin | 255 | 0.288 ± 0.059 | 0.710 | 0.033 | 0.120 | 0.615 | 0.535 |
| +twin | kNN filter | 1 | 0.552 ± 0.105 | 0.868 | 0.347 | 0.440 | 0.752 | 0.734 |
| +twin | weighted | 255 | 0.362 ± 0.075 | 0.773 | 0.082 | 0.231 | 0.654 | 0.598 |
| +recipe | 255 | 0.492 ± 0.069 | 0.803 | 0.337 | 0.337 | 0.805 | 0.754 |
| +recipe | kNN filter | 203 | 0.505 ± 0.059 | 0.817 | 0.343 | 0.354 | 0.791 | 0.742 |
| +recipe | weighted | 255 | 0.536 ± 0.055 | 0.823 | 0.372 | 0.414 | 0.808 | 0.769 |
| +recipe/rel | 255 | 0.514 ± 0.064 | 0.839 | 0.343 | 0.361 | 0.804 | 0.757 |
| +recipe/rel | kNN filter | 203 | 0.508 ± 0.069 | 0.838 | 0.341 | 0.344 | 0.792 | 0.745 |
| +recipe/rel | weighted | 255 | 0.548 ± 0.063 | 0.849 | 0.374 | 0.421 | 0.807 | 0.770 |
| +recipe/rel+jitter | 255 | 0.479 ± 0.076 | 0.842 | 0.277 | 0.319 | 0.793 | 0.740 |
| +recipe/rel+jitter | kNN filter | 202 | 0.495 ± 0.065 | 0.840 | 0.305 | 0.341 | 0.772 | 0.734 |
| +recipe/rel+jitter | weighted | 255 | 0.531 ± 0.068 | 0.857 | 0.338 | 0.397 | 0.804 | 0.764 |
| +recipe/rel+shift | 255 | 0.520 ± 0.052 | 0.822 | 0.346 | 0.393 | 0.814 | 0.764 |
| +recipe/rel+shift | kNN filter | 208 | 0.525 ± 0.053 | 0.833 | 0.352 | 0.391 | 0.816 | 0.768 |
| +recipe/rel+shift | weighted | 255 | 0.543 ± 0.049 | 0.833 | 0.363 | 0.435 | 0.817 | 0.774 |
| +recipe/rel+div | 255 | 0.483 ± 0.074 | 0.815 | 0.286 | 0.348 | 0.803 | 0.746 |
| +recipe/rel+div | kNN filter | 207 | 0.496 ± 0.071 | 0.819 | 0.309 | 0.362 | 0.804 | 0.753 |
| +recipe/rel+div | weighted | 255 | 0.534 ± 0.063 | 0.843 | 0.341 | 0.420 | 0.809 | 0.765 |
| +recipe/rel/bidir | 255 | 0.461 ± 0.077 | 0.725 | 0.320 | 0.337 | 0.787 | 0.728 |
| +recipe/rel/bidir | kNN filter | 227 | 0.467 ± 0.075 | 0.741 | 0.315 | 0.344 | 0.788 | 0.730 |
| +recipe/rel/bidir | weighted | 255 | 0.517 ± 0.075 | 0.759 | 0.372 | 0.421 | 0.799 | 0.753 |
| +oracle | 255 | 0.678 ± 0.028 | 0.870 | 0.508 | 0.657 | 0.934 | 0.890 |
| +oracle | kNN filter | 132 | 0.693 ± 0.042 | 0.912 | 0.501 | 0.667 | 0.925 | 0.884 |
| +oracle | weighted | 255 | 0.690 ± 0.037 | 0.873 | 0.513 | 0.683 | 0.936 | 0.894 |

Paired differences in recall (ridge; same test anomalies, pooled over repetitions, bootstrap 95 % CI):

| A − B | all kinds | spike | level shift | drift | per repetition |
|---|---|---:|---:|---:|---|
| +recipe - real_only | -0.102 [-0.114, -0.090] | -0.085 | -0.059 | -0.163 | -0.05 -0.22 -0.11 -0.05 -0.08 |
| +recipe/rel - real_only | -0.080 [-0.091, -0.068] | -0.049 | -0.052 | -0.138 | -0.04 -0.17 -0.09 -0.03 -0.07 |
| +recipe/rel+div - real_only | -0.111 [-0.123, -0.099] | -0.073 | -0.109 | -0.151 | -0.09 -0.14 -0.13 -0.12 -0.07 |
| +recipe/rel - +recipe | +0.022 [+0.016, +0.029] | +0.035 | +0.007 | +0.025 | +0.01 +0.04 +0.02 +0.03 +0.01 |
| +recipe/rel+jitter - +recipe/rel | -0.035 [-0.043, -0.028] | +0.003 | -0.067 | -0.042 | -0.03 +0.01 -0.04 -0.10 -0.01 |
| +recipe/rel+shift - +recipe/rel | +0.006 [-0.002, +0.013] | -0.017 | +0.003 | +0.032 | -0.03 +0.08 +0.01 -0.01 -0.02 |
| +recipe/rel+div - +recipe/rel | -0.031 [-0.040, -0.023] | -0.023 | -0.057 | -0.013 | -0.05 +0.04 -0.04 -0.10 -0.01 |
| +recipe/rel+div - +twin | +0.196 [+0.182, +0.209] | +0.105 | +0.253 | +0.228 | +0.27 +0.26 +0.18 +0.06 +0.21 |
| +recipe/rel+div - +C1-like | -0.086 [-0.098, -0.075] | -0.014 | -0.111 | -0.134 | -0.08 -0.07 -0.13 -0.11 -0.03 |
| +recipe/rel+div - +posthoc | +0.048 [+0.034, +0.062] | -0.049 | +0.147 | +0.045 | +0.11 +0.05 -0.08 -0.04 +0.19 |
| +recipe/rel/bidir - +recipe/rel | -0.054 [-0.064, -0.043] | -0.113 | -0.023 | -0.024 | -0.02 -0.08 +0.04 -0.15 -0.06 |
| +recipe - +twin | +0.205 [+0.191, +0.218] | +0.093 | +0.304 | +0.217 | +0.31 +0.17 +0.20 +0.13 +0.21 |
| +recipe - +C1-like | -0.077 [-0.088, -0.066] | -0.026 | -0.060 | -0.145 | -0.05 -0.15 -0.11 -0.04 -0.04 |
| +C1-like - real_only | -0.025 [-0.033, -0.017] | -0.059 | +0.001 | -0.017 | -0.01 -0.06 +0.00 -0.01 -0.04 |
| +posthoc - real_only | -0.159 [-0.174, -0.144] | -0.023 | -0.257 | -0.196 | -0.20 -0.19 -0.05 -0.09 -0.26 |
| +unguided - real_only | -0.151 [-0.163, -0.139] | -0.029 | -0.166 | -0.257 | -0.12 -0.27 -0.20 -0.09 -0.07 |
| +twin - real_only | -0.307 [-0.321, -0.292] | -0.178 | -0.363 | -0.379 | -0.36 -0.39 -0.31 -0.19 -0.29 |
| +oracle - real_only | +0.084 [+0.068, +0.100] | -0.018 | +0.113 | +0.157 | +0.05 +0.01 +0.20 +0.16 -0.00 |
| +oracle - +recipe/rel+div | +0.195 [+0.180, +0.211] | +0.055 | +0.222 | +0.309 | +0.14 +0.15 +0.33 +0.29 +0.07 |
| +twin | kNN filter - +twin | +0.264 [+0.250, +0.278] | +0.158 | +0.314 | +0.320 | +0.36 +0.34 +0.16 +0.17 +0.29 |
| +unguided | kNN filter - +unguided | +0.076 [+0.066, +0.086] | -0.024 | +0.135 | +0.117 | +0.04 +0.16 +0.08 +0.10 +0.01 |
| +recipe/rel+div | kNN filter - +recipe/rel+div | +0.013 [+0.006, +0.021] | +0.003 | +0.023 | +0.014 | +0.04 +0.01 +0.03 +0.00 -0.01 |
| +oracle | kNN filter - +oracle | +0.015 [+0.004, +0.026] | +0.042 | -0.007 | +0.010 | +0.06 +0.01 +0.03 -0.03 +0.00 |
| +recipe/rel+div | weighted - +recipe/rel+div | +0.051 [+0.044, +0.058] | +0.027 | +0.055 | +0.072 | +0.02 +0.08 +0.05 +0.08 +0.02 |
| +oracle | weighted - +oracle | +0.012 [+0.005, +0.018] | +0.003 | +0.005 | +0.026 | +0.04 +0.01 +0.01 +0.01 -0.01 |

### E20 detector utility (logistic): 18 real anomalies + up to 255 generated, ROCKET features, recall at 1 % FAR (5 repetitions × 900 test anomalies)

| training positives | n generated kept | recall @ 1 % FAR | spike | level shift | drift | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| real_only | 0 | 0.489 ± 0.080 | 0.877 | 0.283 | 0.307 | 0.744 | 0.705 |
| +posthoc | 255 | 0.511 ± 0.049 | 0.962 | 0.289 | 0.282 | 0.844 | 0.774 |
| +posthoc | kNN filter | 201 | 0.523 ± 0.033 | 0.953 | 0.319 | 0.298 | 0.824 | 0.768 |
| +posthoc | weighted | 255 | 0.529 ± 0.049 | 0.963 | 0.302 | 0.321 | 0.849 | 0.784 |
| +unguided | 255 | 0.470 ± 0.063 | 0.968 | 0.281 | 0.160 | 0.754 | 0.699 |
| +unguided | kNN filter | 41 | 0.495 ± 0.071 | 0.939 | 0.330 | 0.217 | 0.768 | 0.725 |
| +unguided | weighted | 255 | 0.485 ± 0.069 | 0.965 | 0.291 | 0.198 | 0.757 | 0.710 |
| +C1-like | 255 | 0.498 ± 0.076 | 0.920 | 0.293 | 0.281 | 0.755 | 0.717 |
| +C1-like | kNN filter | 253 | 0.499 ± 0.073 | 0.919 | 0.295 | 0.283 | 0.755 | 0.717 |
| +C1-like | weighted | 255 | 0.501 ± 0.075 | 0.904 | 0.299 | 0.299 | 0.752 | 0.715 |
| +twin | 255 | 0.316 ± 0.058 | 0.804 | 0.035 | 0.107 | 0.622 | 0.551 |
| +twin | kNN filter | 1 | 0.483 ± 0.084 | 0.883 | 0.275 | 0.291 | 0.745 | 0.706 |
| +twin | weighted | 255 | 0.354 ± 0.068 | 0.830 | 0.067 | 0.165 | 0.649 | 0.590 |
| +recipe | 255 | 0.507 ± 0.057 | 0.925 | 0.322 | 0.273 | 0.787 | 0.738 |
| +recipe | kNN filter | 203 | 0.499 ± 0.053 | 0.922 | 0.315 | 0.259 | 0.781 | 0.735 |
| +recipe | weighted | 255 | 0.507 ± 0.059 | 0.913 | 0.317 | 0.293 | 0.781 | 0.736 |
| +recipe/rel | 255 | 0.515 ± 0.059 | 0.938 | 0.326 | 0.281 | 0.784 | 0.741 |
| +recipe/rel | kNN filter | 203 | 0.513 ± 0.059 | 0.933 | 0.333 | 0.272 | 0.778 | 0.738 |
| +recipe/rel | weighted | 255 | 0.517 ± 0.060 | 0.925 | 0.327 | 0.300 | 0.780 | 0.741 |
| +recipe/rel+jitter | 255 | 0.522 ± 0.052 | 0.950 | 0.355 | 0.262 | 0.792 | 0.747 |
| +recipe/rel+jitter | kNN filter | 202 | 0.520 ± 0.054 | 0.944 | 0.334 | 0.281 | 0.780 | 0.741 |
| +recipe/rel+jitter | weighted | 255 | 0.527 ± 0.055 | 0.943 | 0.352 | 0.286 | 0.788 | 0.748 |
| +recipe/rel+shift | 255 | 0.525 ± 0.054 | 0.937 | 0.337 | 0.302 | 0.798 | 0.753 |
| +recipe/rel+shift | kNN filter | 208 | 0.519 ± 0.057 | 0.934 | 0.335 | 0.287 | 0.798 | 0.751 |
| +recipe/rel+shift | weighted | 255 | 0.527 ± 0.055 | 0.927 | 0.335 | 0.319 | 0.793 | 0.750 |
| +recipe/rel+div | 255 | 0.536 ± 0.051 | 0.953 | 0.363 | 0.293 | 0.804 | 0.759 |
| +recipe/rel+div | kNN filter | 207 | 0.527 ± 0.054 | 0.946 | 0.343 | 0.291 | 0.797 | 0.752 |
| +recipe/rel+div | weighted | 255 | 0.538 ± 0.055 | 0.943 | 0.360 | 0.313 | 0.799 | 0.758 |
| +recipe/rel/bidir | 255 | 0.502 ± 0.033 | 0.876 | 0.350 | 0.279 | 0.790 | 0.743 |
| +recipe/rel/bidir | kNN filter | 227 | 0.505 ± 0.034 | 0.881 | 0.351 | 0.283 | 0.791 | 0.744 |
| +recipe/rel/bidir | weighted | 255 | 0.508 ± 0.039 | 0.874 | 0.349 | 0.302 | 0.788 | 0.745 |
| +oracle | 255 | 0.742 ± 0.036 | 0.957 | 0.573 | 0.695 | 0.940 | 0.908 |
| +oracle | kNN filter | 132 | 0.664 ± 0.041 | 0.954 | 0.506 | 0.531 | 0.916 | 0.873 |
| +oracle | weighted | 255 | 0.739 ± 0.033 | 0.956 | 0.567 | 0.693 | 0.941 | 0.908 |

Paired differences in recall (logistic; same test anomalies, pooled over repetitions, bootstrap 95 % CI):

| A − B | all kinds | spike | level shift | drift | per repetition |
|---|---|---:|---:|---:|---|
| +recipe - real_only | +0.018 [+0.009, +0.026] | +0.048 | +0.039 | -0.034 | +0.03 +0.05 +0.05 +0.01 -0.04 |
| +recipe/rel - real_only | +0.026 [+0.018, +0.034] | +0.061 | +0.043 | -0.025 | +0.03 +0.05 +0.05 +0.02 -0.02 |
| +recipe/rel+div - real_only | +0.047 [+0.038, +0.057] | +0.076 | +0.080 | -0.014 | +0.03 +0.09 +0.10 +0.02 +0.00 |
| +recipe/rel - +recipe | +0.008 [+0.003, +0.014] | +0.013 | +0.004 | +0.009 | +0.00 +0.01 +0.00 +0.02 +0.02 |
| +recipe/rel+jitter - +recipe/rel | +0.007 [+0.001, +0.013] | +0.012 | +0.029 | -0.019 | +0.00 -0.00 +0.03 +0.00 +0.01 |
| +recipe/rel+shift - +recipe/rel | +0.010 [+0.004, +0.016] | -0.001 | +0.011 | +0.021 | -0.00 +0.03 +0.02 -0.00 +0.00 |
| +recipe/rel+div - +recipe/rel | +0.021 [+0.014, +0.028] | +0.015 | +0.037 | +0.011 | +0.00 +0.04 +0.05 -0.00 +0.02 |
| +recipe/rel+div - +twin | +0.221 [+0.208, +0.233] | +0.149 | +0.327 | +0.185 | +0.23 +0.27 +0.23 +0.14 +0.23 |
| +recipe/rel+div - +C1-like | +0.038 [+0.030, +0.047] | +0.033 | +0.069 | +0.012 | +0.03 +0.08 +0.08 +0.01 -0.01 |
| +recipe/rel+div - +posthoc | +0.025 [+0.012, +0.038] | -0.009 | +0.074 | +0.011 | +0.03 -0.01 -0.06 -0.02 +0.18 |
| +recipe/rel/bidir - +recipe/rel | -0.013 [-0.022, -0.005] | -0.062 | +0.024 | -0.002 | -0.02 -0.03 +0.04 -0.00 -0.05 |
| +recipe - +twin | +0.191 [+0.178, +0.203] | +0.121 | +0.287 | +0.165 | +0.23 +0.23 +0.18 +0.13 +0.19 |
| +recipe - +C1-like | +0.009 [+0.001, +0.016] | +0.005 | +0.029 | -0.008 | +0.03 +0.03 +0.03 +0.00 -0.05 |
| +C1-like - real_only | +0.009 [+0.004, +0.015] | +0.043 | +0.011 | -0.026 | -0.00 +0.01 +0.02 +0.01 +0.01 |
| +posthoc - real_only | +0.022 [+0.008, +0.037] | +0.085 | +0.006 | -0.025 | -0.00 +0.10 +0.16 +0.04 -0.18 |
| +unguided - real_only | -0.019 [-0.030, -0.009] | +0.091 | -0.002 | -0.147 | -0.05 -0.04 +0.04 -0.03 -0.02 |
| +twin - real_only | -0.173 [-0.186, -0.161] | -0.073 | -0.247 | -0.199 | -0.20 -0.18 -0.13 -0.12 -0.23 |
| +oracle - real_only | +0.253 [+0.237, +0.268] | +0.080 | +0.291 | +0.388 | +0.24 +0.26 +0.43 +0.26 +0.07 |
| +oracle - +recipe/rel+div | +0.206 [+0.191, +0.220] | +0.004 | +0.211 | +0.402 | +0.22 +0.17 +0.33 +0.24 +0.07 |
| +twin | kNN filter - +twin | +0.168 [+0.156, +0.180] | +0.079 | +0.239 | +0.184 | +0.20 +0.16 +0.13 +0.12 +0.23 |
| +unguided | kNN filter - +unguided | +0.026 [+0.018, +0.034] | -0.029 | +0.049 | +0.057 | +0.05 +0.03 -0.00 +0.02 +0.03 |
| +recipe/rel+div | kNN filter - +recipe/rel+div | -0.009 [-0.015, -0.003] | -0.007 | -0.019 | -0.001 | +0.01 -0.03 -0.02 +0.00 -0.00 |
| +oracle | kNN filter - +oracle | -0.078 [-0.088, -0.068] | -0.003 | -0.067 | -0.163 | -0.05 -0.03 -0.12 -0.11 -0.08 |
| +recipe/rel+div | weighted - +recipe/rel+div | +0.002 [-0.002, +0.006] | -0.011 | -0.003 | +0.020 | +0.00 +0.02 -0.01 +0.00 -0.00 |
| +oracle | weighted - +oracle | -0.003 [-0.006, +0.001] | -0.001 | -0.006 | -0.001 | -0.01 +0.00 -0.01 +0.00 +0.00 |

Share of generated windows the kNN novelty filter keeps (outside the nominal 99 % set), per kind:

| generator | spike | level shift | drift |
|---|---:|---:|---:|
| posthoc | 0.78 | 0.80 | 0.79 |
| unguided | 0.17 | 0.17 | 0.14 |
| C1-like | 1.00 | 0.99 | 0.99 |
| twin | 0.00 | 0.01 | 0.00 |
| recipe | 1.00 | 0.96 | 0.42 |
| recipe/rel | 0.99 | 0.97 | 0.42 |
| recipe/rel+jitter | 0.97 | 0.98 | 0.42 |
| recipe/rel+shift | 0.99 | 0.97 | 0.49 |
| recipe/rel+div | 0.97 | 0.98 | 0.49 |
| recipe/rel/bidir | 0.98 | 0.98 | 0.71 |
| oracle | 0.88 | 0.28 | 0.39 |

### E20b more anomalous is not more useful (logistic, 3 repetitions, recall at 1 % FAR)

| training positives | all | spike | level shift | drift | drift windows outside the nominal set |
|---|---:|---:|---:|---:|---:|
| real_only | 0.448 | 0.867 | 0.232 | 0.244 |  |
| +div | 0.520 | 0.957 | 0.353 | 0.250 | 0.56 |
| +div/ramp | 0.489 | 0.952 | 0.351 | 0.164 | 0.99 |
| +div/ramp/wide | 0.470 | 0.964 | 0.316 | 0.131 | 0.99 |
| +oracle | 0.760 | 0.956 | 0.598 | 0.727 | 0.49 |

### E20c the gain against a better-tuned detector (logistic, penalty C swept, 3 repetitions)

| C, training positives | recall @ 1 % FAR | spike | level shift | drift | AUROC |
|---|---:|---:|---:|---:|---:|
| C=0.01 | real_only | 0.413 | 0.861 | 0.202 | 0.177 | 0.765 |
| C=0.01 | +recipe/rel+div | 0.474 | 0.950 | 0.317 | 0.154 | 0.823 |
| C=0.01 | +oracle | 0.742 | 0.962 | 0.584 | 0.680 | 0.944 |
| C=0.1 | real_only | 0.448 | 0.867 | 0.232 | 0.244 | 0.772 |
| C=0.1 | +recipe/rel+div | 0.520 | 0.957 | 0.353 | 0.250 | 0.836 |
| C=0.1 | +oracle | 0.760 | 0.956 | 0.598 | 0.727 | 0.944 |
| C=1 | real_only | 0.482 | 0.877 | 0.264 | 0.306 | 0.775 |
| C=1 | +recipe/rel+div | 0.551 | 0.960 | 0.383 | 0.310 | 0.847 |
| C=1 | +oracle | 0.756 | 0.948 | 0.598 | 0.723 | 0.942 |
| C=10 | real_only | 0.563 | 0.893 | 0.349 | 0.448 | 0.775 |
| C=10 | +recipe/rel+div | 0.603 | 0.960 | 0.419 | 0.430 | 0.854 |
| C=10 | +oracle | 0.750 | 0.940 | 0.591 | 0.718 | 0.941 |
