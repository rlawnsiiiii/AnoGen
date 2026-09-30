# Raw-window realism audit

Readable in a plain editor. Code: `src/anogen/shell/realism.py` (measures) and
`src/anogen/phases/realism.py` (phase, tables, plots). Plan:
`realism_audit_metrics_d1e88891.plan.md`. **Findings from the first run are in
§17** — start there if you only want the result.

This audit is **parallel** to the locked evaluation protocol, not a replacement:

- `feature_pack_v1`, ARP, Coverage@τ, EDI, Div, and `results/shell_s4/` are
  untouched. Nothing here re-chooses τ and nothing here retrains a generator.
- The audit only reads existing raw windows and writes to
  `results/shell_realism/` and `docs/realism/`.
- Published fidelity measures and project-specific diagnostics go to separate
  tables (`metrics.csv` carries a `tier` column), so a morphology check can
  never be mistaken for a standard realism metric.

The motivation is that every S4 number looks at φ(x) = `feature_pack_v1`, which
z-scores each window and therefore cannot see DC level or absolute amplitude.
This audit measures the **raw** windows instead.

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml realism
```

The frozen configuration is `shell.realism` in `configs/shell_mission1.yaml`.
`bins: 50` and `max_lag: 64` are the TSGBench defaults.

---

## 0. Notation and shared conventions

| Symbol | Meaning |
|---|---|
| `W` | window length in 30 s bins (512 by default) |
| `N` | number of windows in a set |
| `c` | channel index; every window belongs to exactly one channel |
| `C` | channels populated by **both** the real and the generated set |
| `u` | window after channel-span normalization (below) |
| `W1` | Wasserstein-1 distance between two 1-D samples |

### Channel-span normalization

Raw telemetry channels live on different physical scales, so a raw-value metric
needs a common unit. The audit uses the **frozen per-channel nominal training
span** from S0 (`ChannelMinMax`), the same affine the denoiser trains on, and
deliberately does **not** clip. Values outside `[0, 1]` are real information:
they mean the window left the training envelope.

```
u[i, t]  =  ( x[i, t] − lo_c )  /  ( hi_c − lo_c )
```

```python
def channel_span_normalize(
    x: np.ndarray,
    channel_idx: np.ndarray,
    scaler: ChannelMinMax,
) -> np.ndarray:
    """Map raw telemetry to channel training-envelope units without clipping."""
    a = _windows(x)
    ch = _channels(channel_idx, len(a))
    lo = scaler.lo[ch, None]
    span = (scaler.hi - scaler.lo)[ch, None]
    return (a - lo) / np.maximum(span, _EPS)
```

This is the one lens every raw metric looks through, and it is frozen. Unlike
φ it keeps level and amplitude: a window sitting at 0.80 when the channel
normally sits at 0.95 stays visibly different.

### Macro-averaging over channels

Every distributional measure is computed per channel and then averaged with
equal weight, so one heavily-sampled channel cannot dominate.

```
metric  =  ( 1 / |C| )  Σ_{c ∈ C}  metric_c
```

```python
def _macro_result(per_channel: dict[str, float]) -> dict[str, Any]:
    vals = list(per_channel.values())
    return {
        "value": float(np.mean(vals)) if vals else float("nan"),
        "per_channel": per_channel,
        "n_channels": len(per_channel),
    }
```

`n_real < n_min` (10) sets `insufficient_evidence` on the row instead of
producing a number that reads like evidence.

---

## Part 1 — Published fidelity scorecard

All six are `tier = published` in `metrics.csv`. Lower is better everywhere
except C2ST, where the target is 0.5 balanced accuracy.

## 1. Marginal Wasserstein-1 (`marginal_w1`)

Stenger et al. list "WD on marginals" as a distribution-level measure. Pool
every sample of every window in a channel and compare the two 1-D samples.

```
P_c  =  { u_real[i, t]  :  all i, all t }
Q_c  =  { u_gen [i, t]  :  all i, all t }

W1(P, Q)  =  ∫ | F_P(v) − F_Q(v) | dv          (F = empirical CDF)

marginal_w1  =  ( 1 / |C| )  Σ_c  W1(P_c, Q_c)
```

```python
def marginal_w1(
    real: np.ndarray,
    synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    scaler: ChannelMinMax,
) -> dict[str, Any]:
    """Macro channel Wasserstein-1 on raw values in channel-span units."""
    r = channel_span_normalize(real, real_ch, scaler)
    g = channel_span_normalize(synth, synth_ch, scaler)
    rc = _channels(real_ch, len(r))
    gc = _channels(synth_ch, len(g))
    per: dict[str, float] = {}
    for c in _shared_channels(rc, gc):
        per[str(c)] = float(wasserstein_distance(r[rc == c].ravel(), g[gc == c].ravel()))
    return _macro_result(per)
```

Because time is pooled away, this is **blind to ordering**: a time-permuted
window scores identically to the original. That is what ACD is for.

---

## 2. TSGBench Marginal Distribution Difference (`mdd`)

TSGBench M4. The paper text does not fix the normalization, so the formula is
taken from TSGBench's own `HistoLoss` / `calculate_mdd`. Unlike `marginal_w1`
this is **per time step**, so it does see position within the window.

For each channel `c` and each time step `t`, the real windows define the bins
and a histogram **density** (count divided by bin width *and* by sample count):

```
K    = bins = 50
lo   = min_i u_real[i, t]
hi   = max_i u_real[i, t]            (hi ← lo + 1e-5 when hi == lo)
Δ    = (hi − lo) / K
e_j  = lo + j Δ                      j = 0 … K

p_real[j]  =  #{ i : u_real[i,t] ∈ [e_j, e_{j+1}) }  /  ( Δ · N_real )
p_gen [j]  =  #{ i : u_gen [i,t] ∈ [e_j, e_{j+1}) }  /  ( Δ · N_gen  )

MDD_c  =  ( 1 / (W−1) )  Σ_{t=1}^{W−1}  ( 1 / K )  Σ_j | p_real[j] − p_gen[j] |
MDD    =  ( 1 / |C| )  Σ_c  MDD_c
```

Three details are TSGBench's, not ours, and all three matter:

1. **Density, not probability mass.** The extra `1/Δ` is not a constant
   rescaling, because `Δ` is set by the real range at each individual time step.
2. **The first time step is dropped.** `calculate_mdd` slices `[:, 1:, :]`.
3. **Generated mass outside the real range is discarded.** `HistoLoss.compute`
   counts a generated value only when it falls strictly inside some bin, yet
   still divides by the full generated count. A generated spike that *overshoots*
   the real range is therefore invisible to MDD rather than penalised. We keep
   the definition and report the discarded mass separately (§8).

```python
    start = 1 if skip_first else 0
    for c in _shared_channels(rc, gc):
        rv = r[rc == c]
        gv = g[gc == c]
        if rv.shape[1] <= start:
            continue
        step_diff = np.empty(rv.shape[1] - start, dtype=np.float64)
        step_drop = np.empty(rv.shape[1] - start, dtype=np.float64)
        for j, t in enumerate(range(start, rv.shape[1])):
            rt, gt = rv[:, t], gv[:, t]
            lo, hi = float(rt.min()), float(rt.max())
            if hi == lo:
                hi = lo + 1e-5
            edges = np.linspace(lo, hi, int(bins) + 1)
            width = float(edges[1] - edges[0])
            rh = np.histogram(rt, bins=edges)[0] / width / float(len(rt))
            gh = np.histogram(gt, bins=edges)[0] / width / float(len(gt))
            step_diff[j] = float(np.mean(np.abs(rh - gh)))
            step_drop[j] = float(np.mean((gt < lo) | (gt > hi)))
        per[str(c)] = float(step_diff.mean())
        outside[str(c)] = float(step_drop.mean())
```

MDD is **scale-dependent**, so the normalization above has to stay frozen for
the numbers to be comparable across methods.

---

## 3. TSGBench AutoCorrelation Difference (`acd`)

TSGBench M5, again read off `acf_torch` / `acf_diff` / `calculate_acd` rather
than the paper prose. One mean and one **biased** variance are pooled over all
windows *and* all time steps of the channel, and the lag-`k` product is averaged
over every window and all `W − k` valid offsets:

```
L      =  min(max_lag, W)                        max_lag = 64

μ_c    =  mean over (i, t)  of  u[i, t]
z      =  u − μ_c
v_c    =  mean over (i, t)  of  z[i, t]²         (biased variance)

ρ_c[0] =  1
ρ_c[k] =  ( mean over (i, t)  of  z[i, t+k] · z[i, t] )  /  v_c      k = 1 … L−1
```

The two ACF vectors are then compared by **Euclidean norm over lags** — that is
what `acf_diff = sqrt(sum(x², dim=0))` does — and macro-averaged:

```
ACD_c  =  sqrt(  Σ_{k=0}^{L−1} ( ρ_real,c[k] − ρ_gen,c[k] )²  )
ACD    =  ( 1 / |C| )  Σ_c  ACD_c
```

Lag 0 is 1 on both sides and always contributes 0.

```python
def pooled_acf(x: np.ndarray, max_lag: int = 64) -> np.ndarray:
    a = _windows(x)
    w = a.shape[1]
    lag = min(max(0, int(max_lag)), w)
    if len(a) == 0 or lag == 0:
        return np.zeros(lag, dtype=np.float64)
    z = a - a.mean()
    var = float(np.mean(z * z))
    out = np.zeros(lag, dtype=np.float64)
    out[0] = 1.0
    if var <= _EPS:
        return out
    for k in range(1, lag):
        out[k] = float(np.mean(z[:, k:] * z[:, :-k])) / var
    return out


    for c in _shared_channels(rc, gc):
        ra = pooled_acf(r[rc == c], max_lag)
        ga = pooled_acf(g[gc == c], max_lag)
        n = min(len(ra), len(ga))
        per[str(c)] = float(np.sqrt(np.sum((ra[:n] - ga[:n]) ** 2)))
```

Pooling is not the same as averaging per-window ACF curves, which is what a
naive reading produces: pooled centring lets **between-window** level spread
contribute to `v_c`, so a set of flat windows at different levels has a very
different pooled ACF from a set of windows that each drift.

The pooled ACF is exactly invariant to a per-channel affine map, so `acd` takes
the raw windows and needs no scaler. The ACF figures use `pooled_acf` too, so
the plots and the score show the same quantity.

---

## 4. COSCI-GAN Average Wasserstein Distance (`fft_awd_amplitude`)

Seyfi et al. define AWD as the channel average of the Wasserstein distance
between real and generated **amplitude** distributions. Their toy study knew
its own generating amplitudes, so the original paper never specifies how to
*extract* an amplitude, and their repository contains no AWD implementation.
The extraction step therefore follows Stenger et al.'s survey description —
"FFT determines the most likely period in each channel of each sample, then the
amplitudes are compared" — and is frozen here.

Per window, after subtracting its own mean:

```
X[k]  =  Σ_t  u[t] · exp( −2πi k t / W )                 (rFFT)
a[k]  =  2 |X[k]| / W                    k = 1 … ⌊W/2⌋   (DC bin dropped)

k*    =  argmax_k a[k]                   (lowest k on ties)
A     =  a[k*]                           amplitude
F     =  k*                              dominant bin
```

The `2 |X_k| / W` scaling is chosen so a pure sinusoid returns its own
amplitude: for `u[t] = A sin(2π k t / W)` we get `|X[k]| = A W / 2`, hence
`a[k] = A`. That identity is pinned by a test.

```
AWD_c  =  W1( { A : real windows of c },  { A : generated windows of c } )
AWD    =  ( 1 / |C| )  Σ_c  AWD_c
```

```python
def dominant_fft_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _windows(x)
    if a.shape[1] < 3:
        return np.zeros(len(a)), np.zeros(len(a))
    centered = a - a.mean(axis=1, keepdims=True)
    amp = 2.0 * np.abs(np.fft.rfft(centered, axis=1)) / float(a.shape[1])
    amp = amp[:, 1:]
    idx = np.argmax(amp, axis=1)
    peak = amp[np.arange(len(a)), idx]
    return (idx + 1).astype(np.float64), peak.astype(np.float64)


    for c in _shared_channels(rc, gc):
        rf, ra = dominant_fft_features(r[rc == c])
        gf, ga = dominant_fft_features(g[gc == c])
        if len(ra) == 0 or float(np.max(ra)) <= _EPS:
            skipped.append(c)
            continue
        amp_per[str(c)] = float(wasserstein_distance(ra, ga))
```

The survey warns that AWD "may fail on time series without any seasonality".
Channels whose real amplitudes all vanish are **skipped and listed** in
`skipped_nonperiodic_channels` rather than scored, which is our handling of that
caveat. `frequency_w1` (§8) is a project diagnostic, not part of AWD.

---

## 5. Interpretable feature distances (`feature_distance_macro`)

Six per-window statistics, each compared as a 1-D distribution. These are the
elementary statistics both the Stenger survey and TSGBench M6/M7 use, made
explicit so a failure is readable: "the std distribution is off" is actionable
in a way that a single scalar is not.

```
mean      =  ( 1 / W )  Σ_t  u[t]
std       =  sqrt( ( 1 / W )  Σ_t ( u[t] − mean )² )
trend     =  Σ_t ( u[t] − mean )( τ_t − τ̄ )  /  Σ_t ( τ_t − τ̄ )²
             with  τ = linspace(−0.5, 0.5, W)
skewness  =  ( 1 / W )  Σ_t  z_t³            z = ( u − mean ) / std
kurtosis  =  ( 1 / W )  Σ_t  z_t⁴  −  3      (excess)
acf1      =  Σ_t ( u_t − mean )( u_{t+1} − mean )  /  Σ_t ( u_t − mean )²
```

Note `acf1` is the per-window lag-1 autocorrelation (`acf_rows`), which is a
different estimator from the pooled `ρ_c[1]` used by ACD.

```python
def interpretable_features(x: np.ndarray) -> np.ndarray:
    """Per-window statistics used by feature-distribution fidelity."""
    a = _windows(x)
    if len(a) == 0:
        return np.zeros((0, len(INTERPRETABLE_FEATURES)), dtype=np.float64)
    w = a.shape[1]
    t = np.linspace(-0.5, 0.5, w, dtype=np.float64)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    centered = a - a.mean(axis=1, keepdims=True)
    trend = (centered @ tc) / max(denom, _EPS)
    sd = a.std(axis=1, keepdims=True)
    z = np.divide(centered, sd, out=np.zeros_like(centered), where=sd > _EPS)
    sk = np.mean(z**3, axis=1)
    ku = np.mean(z**4, axis=1) - 3.0
    acf1 = acf_rows(a, 1)[:, 1] if w > 1 else np.zeros(len(a))
    return np.column_stack([a.mean(axis=1), a.std(axis=1), trend, sk, ku, acf1])
```

Each feature is macro-averaged over channels, and the headline is the mean of
the six. All seven numbers are written out (`feature_w1_mean`,
`feature_w1_std`, …) so the headline never hides which feature moved.

```
feature_w1[f]           =  ( 1 / |C| )  Σ_c  W1( real f of c,  generated f of c )
feature_distance_macro  =  mean over the six features
```

---

## 6. Classifier two-sample test (`c2st.csv`)

Lopez-Paz and Oquab. Train a classifier to tell real from generated; if it
cannot, the two sets are hard to distinguish. Reported over 5 seeds.

```
per seed s:
  balance   : sample equal real / generated counts per channel
  split     : GroupShuffleSplit, test_size = 0.3
              real groups  = event_id      (no event spans train and test)
              synth groups = donor id
  fit       : LogisticRegression(C = 1.0, liblinear) on the raw u windows
  acc       : balanced accuracy on the held-out groups
  score     : discriminative_score = | acc − 0.5 |
  p-value   : (1 + #{ π : balacc(π(y_test), pred) ≥ acc }) / (1 + n_perm)
```

```python
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=int(seed))
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        acc = float(balanced_accuracy_score(y_test, pred))
        null = np.empty(max(0, int(n_perm)), dtype=np.float64)
        for i in range(len(null)):
            null[i] = balanced_accuracy_score(rng.permutation(y_test), pred)
        p = (
            float((1 + np.sum(null >= acc)) / (1 + len(null)))
            if len(null)
            else float("nan")
        )
```

`acc ≈ 0.5` with a large p-value means indistinguishable. `acc ≈ 1.0` means
trivially separable. Three choices keep this honest: the **grouped** split stops
an event from appearing on both sides, the **balanced** accuracy stops a class
imbalance from faking a score, and the classifier is **frozen** (a linear model
on raw windows) so nothing here can be tuned into a better-looking number.

---

## Part 2 — Diagnostics

These are `tier = diagnostic`. They are **not** standard realism metrics and
must never be reported as such. They exist because the four ESA anomaly kinds
have specific morphology that no published distributional measure isolates:
polarity, persistence, and where in the window the event sits.

Diagnostics are summarised per set (`real` / `synthetic`) as mean, median, p05
and p95 in `diagnostics.csv`, with per-window arrays in `per_window.npz`. They
are meant to be **compared between real and synthetic**, not thresholded.

## 7. Training-envelope violations (`diag_envelope`)

The one measure that is deliberately *not* invariant to leaving the envelope.

```
violation[i,t]   =  u[i,t] < 0  or  u[i,t] > 1
excess[i,t]      =  max( −u[i,t],  u[i,t] − 1,  0 )

sample_fraction  =  mean over (i, t)  of  violation
window_fraction  =  fraction of windows with any violation
max_excess       =  max over (i, t)  of  excess
```

```python
    u = channel_span_normalize(x, channel_idx, scaler)
    bad = (u < 0.0) | (u > 1.0)
    excess = np.maximum(-u, np.maximum(u - 1.0, 0.0))
```

Real Point/Global anomalies exceed the channel's nominal range by definition, so
a generator whose `max_excess` is far below the real one is not producing global
anomalies, whatever its shape metrics say.

## 8. Metrics the published definitions discard

Two numbers that exist only because the published metrics throw information
away. Both are written to `metrics.csv` with `tier = diagnostic`.

```
mdd_out_of_real_range_fraction  =  mean over (c, t) of
                                   #{ i : u_gen[i,t] ∉ [lo, hi] } / N_gen

fft_dominant_frequency_w1_c     =  W1( { F : real },  { F : generated } ) / ⌊W/2⌋
```

The first is the generated mass MDD silently drops (§2). The second is the
dominant-bin shift, normalized by the bin count; AWD compares amplitudes only,
so a generator with correct amplitudes at the wrong period scores well on AWD
and badly here.

## 9. Signed peak severity and position (`diag_signed_peak`)

Peak **polarity** distinguishes a spike from a dropout, which `|z|`-style
features collapse together.

```
r[t]      =  u[t]  −  rolling_median( u, width = 9 )[t]
t*        =  argmax_t | r[t] |
severity  =  r[t*]                     signed: + is a spike, − is a dropout
position  =  t* / ( W − 1 )
```

```python
def rolling_residual(x: np.ndarray, width: int = 9) -> np.ndarray:
    a = _windows(x)
    k = int(width)
    if k < 1 or k % 2 == 0:
        raise ValueError("rolling median width must be a positive odd integer")
    return a - median_filter(a, size=(1, k), mode="nearest")


def diag_signed_peak(x: np.ndarray, width: int = 9) -> dict[str, np.ndarray]:
    residual = rolling_residual(x, width)
    if residual.shape[1] == 0:
        return {"severity": np.zeros(len(residual)), "position": np.zeros(len(residual))}
    idx = np.abs(residual).argmax(axis=1)
    severity = residual[np.arange(len(residual)), idx]
    denom = max(1, residual.shape[1] - 1)
    return {"severity": severity, "position": idx.astype(np.float64) / denom}
```

The rolling median (not mean) is used so the baseline is not itself dragged by
the excursion being measured.

## 10. CUSUM change contrast and position (`diag_cusum`)

The standard single-change-point statistic (Truong et al., §2), used here purely
descriptively: how step-like is the window, and where is the step.

```
m  =  cusum_min_segment = 8

for each split  s = m … W−m :
    left   =  ( 1 / s )       Σ_{t < s}  u[t]
    right  =  ( 1 / (W−s) )   Σ_{t ≥ s}  u[t]
    C(s)   =  sqrt( s (W−s) / W )  ·  ( left − right )

s*        =  argmax_s | C(s) |
contrast  =  C(s*)                     signed
position  =  s* / W
```

The `sqrt(s(W−s)/W)` factor is the usual variance normalization: without it a
split one sample from the edge wins on noise alone.

```python
    cs = np.cumsum(a, axis=1)
    total = cs[:, -1]
    best_abs = np.full(n, -np.inf)
    best_signed = np.zeros(n)
    best_s = np.full(n, m)
    for s in range(m, w - m + 1):
        left = cs[:, s - 1] / float(s)
        right = (total - cs[:, s - 1]) / float(w - s)
        val = np.sqrt(s * (w - s) / float(w)) * (left - right)
        take = np.abs(val) > best_abs
        best_abs[take] = np.abs(val[take])
        best_signed[take] = val[take]
        best_s[take] = s
```

## 11. Quarter-level difference (`diag_persistence`)

```
q            =  max( 1, ⌊ W / 4 ⌋ )
persistence =  mean( u[ last q ] )  −  mean( u[ first q ] )
```

```python
def diag_persistence(x: np.ndarray) -> np.ndarray:
    a = _windows(x)
    q = max(1, a.shape[1] // 4)
    return a[:, -q:].mean(axis=1) - a[:, :q].mean(axis=1)
```

A level shift ends somewhere other than where it started; a transient returns.
**Limitation:** this cannot distinguish an early transient from an early
permanent shift when both sit inside the first quarter. It is descriptive
end-to-start evidence, not proof of persistence. Read it together with the
CUSUM position.

## 12. Contiguous excursion count (`diag_excursion_count`)

Counts *events*, not samples, so one wide excursion is 1 and not its width.

```
r      =  u  −  rolling_median( u, 9 )
med    =  median_t  r
mad    =  median_t  | r − med |
mask   =  | r − med |  >  mad_scale · mad           mad_scale = 4
count  =  number of runs of consecutive True in mask
```

```python
    residual = rolling_residual(x, width)
    med = np.median(residual, axis=1, keepdims=True)
    mad = np.median(np.abs(residual - med), axis=1, keepdims=True)
    threshold = np.maximum(float(mad_scale) * mad, _EPS)
    mask = np.abs(residual - med) > threshold
    starts = mask.copy()
    if mask.shape[1] > 1:
        starts[:, 1:] &= ~mask[:, :-1]
    return starts.sum(axis=1).astype(np.int64)
```

Median and MAD are used because the threshold must not be inflated by the
excursions it is supposed to find. A run start is a `True` whose predecessor is
`False`, which is the `starts[:, 1:] &= ~mask[:, :-1]` line.

## 13. First-difference texture (`diag_first_difference`)

Step-injection baselines produce instantaneous jumps; real telemetry rarely
does. This is the cheapest way to see an implausibly sharp edge.

```
d[t]       =  u[t+1] − u[t]
diff_std   =  std_t  d
diff_p99   =  quantile_0.99   | d |
diff_p999  =  quantile_0.999  | d |
```

```python
    d = np.abs(np.diff(a, axis=1))
    return {
        "std": np.diff(a, axis=1).std(axis=1),
        "p99": np.quantile(d, 0.99, axis=1),
        "p999": np.quantile(d, 0.999, axis=1),
    }
```

## 14. Mean periodogram (`mean_psd`, plots only)

```
psd[k]  =  mean over windows of  | rFFT( u − mean(u) )[k] |²  /  W
```

```python
def mean_psd(x: np.ndarray) -> np.ndarray:
    """Mean one-sided periodogram after subtracting each window mean."""
    a = _windows(x)
    if len(a) == 0:
        return np.zeros(a.shape[1] // 2 + 1, dtype=np.float64)
    z = a - a.mean(axis=1, keepdims=True)
    return (np.abs(np.fft.rfft(z, axis=1)) ** 2 / max(1, a.shape[1])).mean(axis=0)
```

Used for `psd_*.png` only. AWD reduces the spectrum to one bin; this shows the
whole thing, which is where an over-smoothed generator is most obvious.

---

## 15. Deviations from the sources

Recorded in `spec.json` on every run so a result is never separated from its
assumptions.

| Choice | Source behaviour | Here | Why |
|---|---|---|---|
| Normalization | TSGBench min-max scales the whole dataset to `[0, 1]` | frozen per-channel nominal training span | matches the denoiser's own affine; keeps out-of-envelope information |
| Channel aggregation | TSGBench averages one tensor whose channels share a sample count | macro-average over channels | each window here belongs to exactly one channel |
| AWD amplitude | unspecified by Seyfi et al. | `2 \|X_k\| / W` at the dominant non-DC bin | recovers `A` for a pure sinusoid; follows Stenger et al. |
| Aperiodic channels | survey notes AWD "may fail" | skipped and listed | a skipped channel is honest; a zero is not |
| `fft_dominant_frequency_w1` | not part of AWD | `tier = diagnostic` | our addition |
| `mdd_out_of_real_range_fraction` | discarded by MDD | `tier = diagnostic` | makes the discarded mass visible |

The audit is run **once** with these frozen. Tuning the measures until the
numbers improve would make them part of the model, which is exactly the failure
mode this phase exists to avoid.

---

## 16. Code map

| Formula | Code |
|---|---|
| `u = (x − lo_c)/(hi_c − lo_c)` | `channel_span_normalize` |
| macro average over channels | `_macro_result`, `_shared_channels` |
| `W1` on pooled values | `marginal_w1` |
| MDD densities | `marginal_distribution_difference` |
| pooled `ρ_c[k]` | `pooled_acf` |
| ACD Euclidean norm | `autocorrelation_difference` |
| `A`, `F` per window | `dominant_fft_features` |
| AWD | `fft_average_wasserstein` |
| six per-window statistics | `interpretable_features` |
| per-feature `W1` | `feature_distribution_distance` |
| per-window lag-1 ACF | `acf_rows` |
| grouped split | `grouped_c2st_split` |
| balanced C2ST + permutation p | `classifier_two_sample_test` |
| envelope violations | `diag_envelope` |
| rolling-median residual | `rolling_residual` |
| signed peak | `diag_signed_peak` |
| CUSUM | `diag_cusum` |
| quarter level difference | `diag_persistence` |
| excursion runs | `diag_excursion_count` |
| first-difference tails | `diag_first_difference` |
| periodogram | `mean_psd` |
| tables, plots, spec | `anogen/phases/realism.py` |

Outputs: `results/shell_realism/{spec,summary}.json`, `metrics.csv`,
`diagnostics.csv`, `c2st.csv`, `per_window.npz`, `plots/`, mirrored to
`docs/realism/`.

---

## 17. First run — findings

Run of 2026-09-30 over 14 galleries, 672 real anomaly windows, 3 event folds.
Tables in `results/shell_realism/`. Read with §15 in mind.

### 17.1 `genias_patched` generates nothing

`results/shell_s3/galleries.npz` holds `genias_patched` **bit-identical** to
`cond`, the donor windows (`np.array_equal` is `True`). GenIAS's raw deviation
from its parent peaks at 3.1e-2, so the Algorithm-2 patch condition never fires
at `patch_tau: 0.2` and the output is the untouched parent. Every
`genias_patched` row in this audit is therefore a `donor` row, which is why the
two lines agree to every decimal. Anywhere else that arm is reported it is
reporting the donor.

### 17.2 The distributional measures are dominated by the nominal bulk

`donor` — untouched real *nominal* windows — wins `marginal_w1`, `mdd` and
`fft_awd_amplitude` against real *anomaly* windows:

| method | marginal_w1 | mdd | acd | fft_awd | feature_macro |
|---|---|---|---|---|---|
| donor | **0.1123** | **3.198** | 5.634 | **0.0031** | 3.705 |
| taxonomy | 0.1122 | 3.375 | 5.411 | 0.0039 | 3.697 |
| cutaddpaste | 0.1123 | 3.449 | 5.446 | 0.0086 | 3.707 |
| posthoc | 0.2734 | 3.476 | **2.928** | 0.0040 | 3.751 |
| c1 | 0.1594 | 4.310 | 4.788 | 0.0274 | 5.334 |
| unguided | 0.1169 | 4.803 | 6.564 | 0.0147 | 3.531 |

A 512-bin window whose fault span is short is *mostly nominal*, so these four
measures largely answer "is this plausible telemetry", not "does this contain an
anomaly". **They are necessary, not sufficient:** good for catching artifacts,
worthless as a ranking of anomaly quality. Do not read the donor row as a win.

### 17.3 C2ST is the published measure that discriminates

Median balanced accuracy over 3 folds × 5 seeds, and the fraction of runs whose
permutation test rejects at 0.05:

| method | median acc | reject@0.05 |
|---|---|---|
| posthoc | 0.487 | 0.00 |
| cutaddpaste / taxonomy | 0.500 | 0.27 / 0.20 |
| donor / genias / shell | 0.509–0.514 | 0.27–0.40 |
| editor | 0.727 | 0.93 |
| c2 | 0.778 | 0.93 |
| **c1** | **0.795** | **0.93** |
| timeleash_l2 / timeleash_gm | 0.802 / 0.838 | 0.93 |

A linear model on raw windows tells `c1` from real anomalies in 14 of 15 runs.
The injection baselines are indistinguishable. Averaging the p-values is
misleading here: 14 rows sit on the 1/201 floor and one degenerate fold-2 split
(`n_test` 140 > `n_train` 112, fold 2 holds only 126 real anomaly windows)
returns 1.0. Report the median and the rejection rate.

### 17.4 Why `c1` is separable — and the mirror-image baseline failure

Diagnostic means, real versus synthetic, in channel-span units:

| | envelope max excess | envelope window frac | cusum contrast | diff p99.9 | excursions |
|---|---|---|---|---|---|
| **real** | 4.016 | **0.039** | **−0.005** | **0.169** | 24.7 |
| c1 | 4.580 | 0.678 | +1.067 | 0.522 | 36.4 |
| timeleash_gm | 4.153 | 0.678 | +0.991 | 0.535 | 37.4 |
| editor | 4.150 | 0.605 | +0.321 | 0.251 | 28.7 |
| posthoc | 1.981 | 0.298 | −0.001 | 0.059 | 23.5 |
| taxonomy | 0.363 | 0.170 | −0.001 | 0.090 | 24.1 |
| genias | 0.106 | 0.001 | −0.005 | 0.055 | 30.0 |
| cutaddpaste | **0.000** | **0.000** | −0.009 | 0.063 | 23.6 |
| donor | **0.000** | **0.000** | −0.004 | 0.058 | 23.2 |

Real anomalies are **mostly inside the envelope with rare extreme excursions**:
only 3.9% of windows leave it, but when they do they reach 4 spans out.

- `c1` has the **magnitude right and the frequency wrong**: max excess 4.58
  against a real 4.02, but it leaves the envelope in 68% of windows, 17× too
  often. It also injects a step that real anomalies do not have (CUSUM contrast
  +1.07 against −0.005) and edges 3× too sharp (p99.9 0.52 against 0.17).
- The injection family fails the other way: texture, CUSUM and excursion counts
  match real closely, but `cutaddpaste` and `donor` **never** leave the envelope
  and `genias` and `taxonomy` barely do, so they cannot produce a Point/Global
  anomaly at all. `posthoc` gets halfway (1.98).

This is the trade-off in one line: **`c1` produces out-of-envelope excursions
that real anomalies need but produces them constantly and too sharply, while the
injection baselines are statistically indistinguishable from real windows and
never leave the envelope.**

It also explains why post-hoc injection wins Coverage@τ on `feature_pack_v1`:
φ z-scores each window, so it cannot see the level and envelope information that
separates these two failure modes, and post-hoc's locally correct shapes win on
the shape similarity that is all φ can measure.

### 17.5 The discard diagnostics earned their place

`mdd_out_of_real_range_fraction`: `c1` **0.386**, `timeleash_l2` 0.373,
`c2` 0.369, `posthoc` 0.265, `donor` 0.036, `unguided` 0.007.

`c1`'s MDD of 4.31 is computed after TSGBench's definition silently discards
39% of its generated mass. Without this diagnostic the MDD figure would be
badly misleading, and the same applies to `posthoc` at 27%.

### 17.6 Per-kind evidence does not exist except for global subsequence

Of 672 real anomaly windows, 636 are ESA global subsequence. The rest:
Point/Global 18, local subsequence 12, level shift 6 — that is ≤ 6 per event
fold, below the `n_min: 10` floor, so **no per-kind published metric exists for
them**. The only per-kind rows in `metrics.csv` are global subsequence.

Any claim that separates "local subsequence" from "global subsequence"
behaviour is unsupported by this label set: there are 12 local-subsequence
windows in total. Fixing that needs more labelled events, not more metrics.

### 17.7 Plot limitation

`marginal_*.png` is dominated by the mode at 0 with a tail to −4 and +2 spans,
so the linear density axis hides everything that matters. The envelope and
CUSUM diagnostics in `diagnostics.csv` carry the actual signal; a log density
axis or a tail-only inset would be the fix.

---

## 18. References

1. Stenger et al., *Evaluation is key: a survey on evaluation measures for
   synthetic time series*, Journal of Big Data 11:66 (2024).
   <https://doi.org/10.1186/s40537-024-00924-7> — WD on marginals, AWD
   (pp. 10–11), elementary statistics.
2. Ang et al., *TSGBench: Time Series Generation Benchmark*, PVLDB 17(3),
   305–318 (2023). <https://doi.org/10.14778/3632093.3632097> — M4 (MDD) and
   M5 (ACD) in §4.2; the formulas used here are from
   [`src/feature_based_measures.py`](https://github.com/YihaoAng/TSGBench/blob/main/src/feature_based_measures.py)
   (`HistoLoss`, `calculate_mdd`, `acf_torch`, `acf_diff`, `calculate_acd`).
3. Seyfi, Rajotte and Ng, *Generating multivariate time series with COmmon
   Source CoordInated GAN (COSCI-GAN)*, NeurIPS (2022) — AWD is defined in the
   toy-data paragraph immediately before Table 1.
4. Lopez-Paz and Oquab, *Revisiting Classifier Two-Sample Tests*, ICLR (2017).
5. Yoon et al., *TimeGAN*, NeurIPS (2019) — origin of the train-on-synthetic
   evaluation family that TSGBench M1/M2 formalise.
6. Truong, Oudre and Vayatis, *Selective review of offline change point
   detection methods*, Signal Processing 167 (2020) — the CUSUM statistic.
7. Darban et al., *GenIAS* (2025), [arXiv:2502.08262](https://arxiv.org/abs/2502.08262).
8. Wang et al., *TSAGen*, IEEE TNSM 19(1), 130–145 (2022).
