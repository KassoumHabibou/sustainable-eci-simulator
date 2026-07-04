"""
robustness.py — Sensitivity of the constrained recommendations to the scenario
==============================================================================
Varies the two scenario parameters over a grid:

  * CO2 emissions growth:      +2, 0, −2, −4  % per year
  * unemployment reduction:     0, 20, 40     % of the latest observed rate

and reports, for each country and each cell of the grid:

  * the required ECI implied by the extended growth equation,
  * whether it exceeds the feasibility bound (and the capped target used),
  * the number of recommended products and the added export volume
    (the required investment),
  * the growth attainable at the target (the possible return), and the
    growth gain relative to the no-optimization forecast under the same
    scenario.

Run AFTER run_simulation.py (uses its cached forward model):
    python robustness.py
Outputs: results/robustness.csv and results/figures/robustness.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src import growth_model as gm
from src import optimization as opt
from run_simulation import get_state, max_feasible_eci_z

# Scenario grid — both margins are annualized REDUCTION rates (+ = success)
CO2_GRID = [-2.0, 0.0, 2.0, 4.0]         # %/yr CO2-emission reduction
RED_GRID = [0.0, 2.0, 4.0]               # %/yr unemployment-rate reduction


def main():
    state = get_state()
    be, bx = state["beta_entry"], state["beta_exit"]

    panel, gdpz_2022 = gm.build_panel(cfg.ECI_PANEL_FILE, cfg.GDP_PC_FILE,
                                      cfg.WDI_PANEL_FILE, cfg.GROWTH_T,
                                      cfg.GROWTH_YEARS)
    m_base, m_ext = gm.fit(panel)
    last_year = int(panel["Year"].max())

    rows = []
    for c, iso3 in cfg.TARGET_COUNTRIES.items():
        gdpz = gdpz_2022.get(iso3, 0.0)
        z_max = max_feasible_eci_z(state, c)
        # no-optimization predicted 2032 ECI (z units)
        from src import forward_model as fm
        eci_2032 = fm.eci_z(state, c)

        for co2 in CO2_GRID:
            for red in RED_GRID:
                t_raw = gm.invert_for_eci(m_ext, cfg.TARGET_GROWTH, gdpz,
                                          last_year, co2red=co2, unred=red)
                capped = t_raw > z_max - 0.02
                t_used = min(t_raw, z_max - 0.02)
                g_att = gm.predict_growth(m_ext, t_used, gdpz, last_year,
                                          co2red=co2, unred=red)
                g_noopt = gm.predict_growth(m_ext, eci_2032, gdpz, last_year,
                                            co2red=co2, unred=red)
                target_nn = state["mean_nn"] + state["sd_nn"] * t_used
                df = opt.run_lp(state, c, target_nn, be, bx)
                sel = df[df["Added_vol"] > 0]
                rows.append({
                    "country": c, "co2_reduction": co2,
                    "unemp_reduction": red,
                    "eci_required": round(t_raw, 3),
                    "max_feasible": round(z_max, 3), "capped": capped,
                    "eci_target_used": round(t_used, 3),
                    "n_products": int(len(sel)),
                    "investment_usd_bn": round(sel["Added_vol"].sum() / 1e9, 1),
                    "attainable_growth_pct": round(g_att, 2),
                    "growth_no_optimization_pct": round(g_noopt, 2),
                    "optimization_gain_pp": round(g_att - g_noopt, 2),
                })
                print(f"{c} CO2red={co2:+.0f} Ured={red:+.0f}  "
                      f"ECI*={t_raw:+.2f}{' (capped)' if capped else ''}"
                      f"  n={len(sel):>3}"
                      f"  inv=${sel['Added_vol'].sum()/1e9:,.0f}B"
                      f"  g={g_att:.2f}%")

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(cfg.RESULTS_DIR, "robustness.csv"), index=False)

    # ── Figure: attainable growth and investment across the grid ──────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for j, (c, cname) in enumerate([("tha", "Thailand"), ("mex", "Mexico")]):
        sub = tab[tab["country"] == c]
        for i, (col, lab, fmt) in enumerate([
                ("attainable_growth_pct", "Attainable growth (%/yr)", "{:.2f}"),
                ("investment_usd_bn", "Required investment (USD bn)", "{:,.0f}")]):
            ax = axes[i, j]
            M = sub.pivot(index="co2_reduction", columns="unemp_reduction",
                          values=col).sort_index(ascending=True)
            im = ax.imshow(M.values, cmap="RdYlGn" if i == 0 else "YlOrRd",
                           aspect="auto")
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    ax.text(b, a, fmt.format(M.values[a, b]), ha="center",
                            va="center", fontsize=9)
            ax.set_xticks(range(M.shape[1]),
                          [f"+{v:.0f}%/yr" for v in M.columns])
            ax.set_yticks(range(M.shape[0]),
                          [f"{v:+.0f}%/yr" for v in M.index])
            ax.set_xlabel("Unemployment reduction (+ = falling)")
            ax.set_ylabel("CO2 emission reduction (+ = falling)")
            ax.set_title(f"{cname} — {lab}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIG_DIR, "robustness.png"), dpi=200)
    print("\nSaved results/robustness.csv and figures/robustness.png")


if __name__ == "__main__":
    main()
