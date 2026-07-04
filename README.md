# ECI Optimization — CO2 & Employment Extension (complete simulation)

Extension of **Stojkoski & Hidalgo, "Optimizing Economic Complexity"**.
The forecast model (paper Eq. 2) and the 0-1 integer optimization (the LP)
are the **authors' code, unchanged**. The extension enters only through the
growth regression (paper Eq. 3), which is augmented with CO2-emissions
growth and the unemployment rate. Inverting the two regressions at the same
3.5 % growth target produces two different ECI targets, and therefore two
different optimal specialization portfolios.

## Folder layout

```
extension_simulation/
├── config.py                  ← all paths & parameters (paper spec)
├── run_simulation.py          ← main entry point:  python run_simulation.py
├── data/
│   ├── bilateral_2022.csv     ← OEC HS4 trade, 2022 (authors' data)
│   ├── Data-ECI-Trade.csv     ← published ECI panel 1995–2022
│   ├── gdp_per_capita.csv     ← WDI GDP pc (NY.GDP.PCAP.KD)
│   ├── merged_eci_wdi_co2.csv ← WDI CO2 + employment panel (extension data)
│   └── beta_eq2_entry/exit.csv← Eq. 2 OLS betas (Δt=10, τ=5, averaged over
│                                12 windows — reproduces paper Table 1;
│                                regenerate with new_approach/script/03)
├── src/
│   ├── ecioptimization.py     ← authors' library, VERBATIM copy (the LP)
│   ├── complexity.py          ← vectorized RCA/ECI/PCI/relatedness (≡ authors)
│   ├── forward_model.py       ← notebook cell 28: 2022 → predicted 2032 state
│   ├── growth_model.py        ← Eq. 3 baseline + extended, inversion
│   └── optimization.py        ← LP wrapper + Figure-5 target sweep
└── results/                   ← all outputs (see below)
```

## Method (per country: Thailand, Mexico)

1. **Forward model** (authors'): 2022 exports → RCA, relatedness → Eq. 2
   with steppingstone = current values → predicted RCA in 2032 → predicted
   specialization matrix → **future PCI** ("Estimated PCI in 2032") and the
   future proximity matrix used by the LP.
2. **Growth regression** on 10-year windows (1999-2009, 2009-2019),
   year FE, HC1:
   - *baseline* (paper): `growth ~ gdpz * eci + C(Year)`
   - *extended*: `growth ~ gdpz * eci + co2g + unemp + eci:co2g + eci:unemp + C(Year)`
3. **Inversion** at 3.5 % target growth. The extended model is inverted with
   CO2 growth fixed at **2 %/yr** and unemployment at **5 %** (config).
4. **Authors' LP** at each ECI target → optimal product portfolio; a
   Figure-5-style sweep records the first ECI target at which each product
   is recommended.

## Sustainability scenario (extension) — policymaker's convention

Both margins enter the model as **annualized reduction rates, positive =
policy success**: `co2red` = pace at which CO2 emissions fall (Paris
direction), `unred` = pace at which the ILO unemployment rate falls
(SDG 8). Central scenario: both at **+2 %/yr** (≈ one-fifth lower after a
decade; Thailand's unemployment 0.73 → 0.60 %, Mexico's 2.77 → 2.26 %).
Parameters live in `config.py` (`CO2_REDUCTION_TARGET`,
`UNEMP_REDUCTION_TARGET`).

## Results (this run)

| | Thailand | Mexico |
|---|---|---|
| Predicted 2032 ECI (no optimization) | **0.933** (paper 0.933) | 0.921 (paper 0.926) |
| ECI target, baseline Eq. 3 | 1.153 (paper 1.225) | 1.724 (paper 1.287) |
| ECI target, extended Eq. 3 (raw) | 2.223 | 2.682 |
| Max feasible ECI (any portfolio) | 1.806 | 2.045 |
| ECI target used (capped) | **1.786** | **2.025** |
| Attainable growth under scenario | **3.03 %** | **2.82 %** |
| Products, baseline | 40 ($19.3B) | 100 ($62.7B) |
| Products, extended | 257 ($184.8B) | 221 ($242.8B) |

**Key findings.** Both margins are statistically significant with opposite
signs: each pp/yr of emission-reduction pace costs ≈0.29 pp of growth
(co2red −0.285***), each pp/yr of unemployment-reduction pace adds
≈0.10 pp (unred +0.104***), amplified by complexity (eci:unred +0.050*).
Under the central scenario the 3.5 % target is infeasible (required ECI
2.2-2.7 vs frontier 1.8-2.0); attainable growth is 3.03 %/2.82 %. In the
robustness grid the two commitments trade almost one-for-one (~0.4 pp per
+2 pp of pace on either margin), 3.5 % survives flat emissions if
unemployment falls ≥2 %/yr, and in every capped cell the product list and
investment are identical — the strategy is scenario-robust.

**Validation of the old approach** (LP run at the paper's published targets
1.225 / 1.287 — `products_*_paper_target.csv`): Thailand matches **14/15**
products of paper Figure 5c (all of 7320, 8501, 2927, 8523, 8532, 4006,
8525, 8455, 8481, 8428, 8101, 3819, 8477, 9031; only 4008 enters just above
the target), Mexico matches **13/15** of Figure 5d (7211 and 8480 belong to
the paper's boundary tier ECI = 1.29 > 1.287 and appear in our sweep just
above the target). Product-level statistics (RCA_cp, relative relatedness,
PCI 2032) match the paper's Figure-5 tables to ~2-3 decimals. Our own
inverted targets differ slightly from the paper's because the local GDP
series is constant-2015 USD while the paper uses PPP constant-2021 (not
re-downloadable in this environment); the machinery is identical.

**How the extension changes specialization.** In the extended regression,
CO2 growth is strongly pro-growth (historical carbon-intensity of growth)
and both interactions with ECI are positive. Fixing CO2 growth at 2 % and
unemployment at 5 % therefore *raises* the marginal growth return of
complexity (larger dGrowth/dECI), so the same 3.5 % growth target requires
**less complexity**: the extended ECI target is lower for both countries.
Consequently the extended portfolio is a **smaller, cheaper, more feasible
subset** of the baseline: it keeps the efficient core (30 of 31 Thai
extended products are also baseline products; 57 of 58 for Mexico) and
drops the highest-effort marginal products (baseline-only products have
~2x the average effort: 0.54 vs 0.27 for Thailand, 1.42 vs 0.65 for
Mexico). Under sustainability constraints, the model recommends
consolidating near-existing capabilities rather than stretching to distant
high-complexity products.

## Robustness

`python robustness.py` (run after the main simulation) re-solves the whole
pipeline over 12 scenarios: CO2 growth ∈ {+2, 0, −2, −4} %/yr ×
unemployment reduction ∈ {0, 20, 40} %. Outputs `results/robustness.csv`
and `results/figures/robustness.png`. Findings: the emissions path drives
everything (~0.45 pp of attainable growth lost per pp of emissions
decline; unemployment effects ≤ 0.1 pp); 3.5 % growth is attainable only
if emissions keep growing (+2 %/yr row); and once the feasibility frontier
binds, the product list and the investment are identical in every cell —
the recommendation is scenario-robust, only the return varies.

## Output files (results/)

- `run.log` — full console log of the last run
- `growth_regressions.csv` — baseline + extended Eq. 3 coefficients
- `products_{tha,mex}_{paper_target,baseline,extended}.csv` — recommended
  products with the first ECI target at which each enters (Figure-5c style),
  implied growth, RCA 2022, relatedness, PCI 2032, effort, added volume
- `comparison_{tha,mex}.csv` — product-by-product baseline vs extended
- `lp_full_*.csv` — raw LP output for every product
- `validation_vs_paper.csv` — overlap with paper Figure 5
- `figures/effort_complexity_{tha,mex}.png` — Figure-5a/b style diagrams
- `cache/state.pkl` — cached forward model (delete to force a full rebuild)

## Policy simulator (Streamlit app)

`app/` contains an interactive simulator for policymakers:

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Pick a country (128 available), set the growth objective and the CO2 and
unemployment growth paths (constrained to ≤ 0 — reductions only, within
validated ranges), and run. The app shows a dynamic country dashboard
(map, flag, latest indicators, ECI/CO2/unemployment/employment-structure
time series), the feasibility verdict, the recommended product list with
each product's required added exports and share of GDP, treemap and
effort-complexity charts, and a one-click **Excel export** whose workbook
includes a README sheet with metadata, method notes and full data sources.
All logic lives in `app/engine.py` (UI-free, testable); the app reuses the
same forward-model cache and growth models as the paper.

## Requirements

`pip install numpy pandas scipy statsmodels pulp matplotlib`
Run time ≈ 3-5 min on first run (builds the forward model), seconds after.
