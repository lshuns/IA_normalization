# 15IA_SIM

Plotting scripts for IA normalization tests in the DarkQuest N-body simulations.

## Scripts

| Script | What it plots | Output in `DarkQuest/plots/` |
|---|---|---|
| `Fig_1_compare_ratio.py` | The two ratios for th and obs, with all redshifts in each panel. One figure per shape type. | `P0pP2_P4_over_P0_<shape>.png` |
| `Fig_2_compare_mu.py` | P<sub>hE</sub>(k, μ<sub>i</sub>) / (1 − μ<sub>i</sub>²) relative to the first μ bin. th and obs, RIAC, one figure per redshift. | `r_mu_RIAC_<snap>.png` |
| `Fig_3_compare_random_AIA.py` | A<sub>IA</sub> against halo mass, from P<sub>dE</sub><sup>0</sup> / P<sub>dd</sub> at k < 0.05 h/Mpc (obs normalization, RIAC). Un-randomized against σ<sub>MF</sub> = 0.37 and 0.64. One figure per redshift. | `AIA_randomization_RIAC_<snap>.png` |
| `Fig_4_compare_random_ratio.py` | The two ratios for un-randomized, σ<sub>MF</sub> = 0.37 and σ<sub>MF</sub> = 0.64 (RIAC), with separate figures for th, obs and th − obs. One set per redshift. | `P0pP2_P4_over_P0_random_<th\|obs\|th-obs>_RIAC_<snap>.png` |
| `Fig_999_compare_shapes_ratio.py` | The th − obs residual of the two ratios for the six shape types. One figure per redshift. | `P0pP2_P4_over_P0_shapes_th-obs_<snap>.png` |
| `test_compare_discreteness.py` | Tests whether the small residual in th comes from grid discreteness. It sums the (1 − μ²) template exactly over the FFT modes, on the measurement's grid and k bins, and compares the result with the measured th ratios. Three figures: full view, zoom-in on the mode sum, and the mode sum per fine k bin. | `test_th_modesum_RIAC.png`, `test_th_modesum_zoom_RIAC.png`, `test_th_modesum_fine.png` |
| `thomasbakx_measurements.py` | Original script by thomasbakx. | none |