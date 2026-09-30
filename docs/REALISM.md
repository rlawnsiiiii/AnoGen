# Raw-window realism audit

Implementation checkpoint only. The full audit has **not been run yet**.

The phase is intentionally parallel to the locked evaluation protocol:

- `feature_pack_v1`, ARP, Coverage@τ, EDI, Div, and `results/shell_s4/`
  remain unchanged.
- The audit reads existing raw windows and writes only to
  `results/shell_realism/` and `docs/realism/`.
- Published fidelity measures and project-specific anomaly diagnostics are
  stored in separate tables.

Before the first run, review the frozen configuration under `shell.realism`
in `configs/shell_mission1.yaml`. Its `bins: 50` and `max_lag: 64` are the
TSGBench defaults.

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml realism
```

## Published fidelity scorecard

- Channel-macro raw marginal Wasserstein-1
- TSGBench Marginal Distribution Difference (MDD)
- TSGBench AutoCorrelation Difference (ACD)
- COSCI-GAN Average Wasserstein Distance over FFT amplitudes (AWD)
- Distribution distances for mean, standard deviation, trend, skewness,
  kurtosis, and lag-1 autocorrelation
- Grouped classifier two-sample test

### Where the formulas come from

MDD and ACD are implemented from TSGBench's own
[`src/feature_based_measures.py`](https://github.com/YihaoAng/TSGBench/blob/main/src/feature_based_measures.py),
because the paper text for M4 and M5 does not fix the normalization:

- MDD (`HistoLoss`, `calculate_mdd`) compares histogram *densities*, so counts
  are divided by the bin width as well as by the number of windows. The real
  windows set the edges, generated mass outside the real range is discarded,
  the first time step is dropped, and 50 bins are used.
- ACD (`acf_torch`, `acf_diff`, `calculate_acd`) pools one mean and one biased
  variance over windows and time per channel, averages the lag-`k` product over
  all windows and all `W - k` offsets, and aggregates the lag vector by its
  Euclidean norm. Lags run `0 .. min(max_lag, W) - 1`.

AWD is only defined to the channel-averaged Wasserstein level by Seyfi et al.,
whose toy study knew its own generating amplitudes. The FFT extraction step
follows Stenger et al.'s description, with `2 |X_k| / W` at the dominant non-DC
bin so that a pure sinusoid returns its amplitude. This is a frozen reading of
an underspecified metric, not a reproduction of COSCI-GAN's code.

Two deviations from TSGBench are unavoidable and are recorded in `spec.json`:
the normalization is the frozen per-channel nominal training span rather than a
global min-max to `[0, 1]`, and because every window belongs to exactly one
channel, MDD and ACD are macro-averaged over channels.

## Diagnostics (not standard realism metrics)

- generated mass discarded by the MDD definition
- dominant-frequency Wasserstein distance
- training-envelope violations
- signed peak severity and position
- CUSUM change contrast and position
- first-quarter to last-quarter level difference
- contiguous extreme-excursion count
- first-difference scale and tails

The quarter-level diagnostic cannot distinguish an early transient from an
early permanent shift when both occupy the first quarter. It is descriptive
end-to-start evidence, not proof of persistence.

## References

1. Stenger et al., *Evaluation is key*, Journal of Big Data 11:66 (2024).
   <https://doi.org/10.1186/s40537-024-00924-7> (AWD, pp. 10–11)
2. Ang et al., *TSGBench*, PVLDB 17(3), 305–318 (2023).
   <https://doi.org/10.14778/3632093.3632097> (M4, M5 in §4.2);
   code at <https://github.com/YihaoAng/TSGBench>
3. Yoon et al., *TimeGAN*, NeurIPS (2019).
4. Seyfi et al., *COSCI-GAN*, NeurIPS (2022). AWD is defined in the toy-data
   paragraph immediately before Table 1.
5. Lopez-Paz and Oquab, *Revisiting Classifier Two-Sample Tests*, ICLR (2017).
6. Darban et al., *GenIAS* (2025), arXiv:2502.08262.
7. Wang et al., *TSAGen*, IEEE TNSM 19(1), 130–145 (2022).
8. Truong et al., *Selective review of offline change point detection
   methods*, Signal Processing 167 (2020).

