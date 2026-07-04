"""
optimization.py — Thin wrapper around the authors' LP (unchanged)
=================================================================
The 0-1 integer program is the authors' `ecioptimization.eci_optimization`
verbatim (see src/ecioptimization.py — an exact copy of their library).
This module only:

  * computes the "effort" Ycp (paper Eq. 2 inverted at the steppingstone),
  * runs the LP for a given ECI target,
  * reproduces the Figure-5 target sweep (which product enters the optimal
    portfolio at which ECI target / implied growth rate).
"""

import io
import contextlib

import numpy as np
import pandas as pd
import pulp

from . import ecioptimization as eciopt
from . import forward_model as fm

# Silence the CBC solver (the LP is solved dozens of times in the sweep).
pulp.LpSolverDefault.msg = 0


def effort_ycp(pr, beta_entry, beta_exit):
    """Ycp — additional RCA needed at the steppingstone year (paper's 'Effort')."""
    r, rel, rr = (pr["RCA_start"].values, pr["Relatedness_start"].values,
                  pr["Relative_relatedness_start"].values)
    out = np.full(r.shape, np.nan)
    for beta, mask in ((beta_entry, r < 1), (beta_exit, r >= 1)):
        out[mask] = np.exp(
            (np.log(2) - (beta[0] + beta[2] * np.log1p(r[mask])
                          + beta[3] * rel[mask] + beta[4] * rr[mask])) / beta[1]
        ) - r[mask] - 1
    return out


def run_lp(state, country, target_nn, beta_entry, beta_exit):
    """Authors' LP at a target expressed on the mean-PCI ('not normalized') scale."""
    pr = fm.product_table(state, country)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = eciopt.eci_optimization(
            target_country=country, ECI_target=target_nn,
            CountryRankings=state["CountryRankings"], ProductRankings=pr,
            indices_to_exclude=[], beta_entry=beta_entry, beta_exit=beta_exit,
            PHIpp=state["PHIpp_future"],
        )
    df["PCI_future"] = pr["PCI"].values
    df["RCA_start"] = pr["RCA_start"].values
    df["Relatedness_start"] = pr["Relatedness_start"].values
    df["Relative_relatedness_start"] = pr["Relative_relatedness_start"].values
    df["Effort_Ycp"] = effort_ycp(pr, beta_entry, beta_exit)
    return df


def sweep_targets(state, country, target_z_final, beta_entry, beta_exit,
                  growth_of_target=None, step=0.05):
    """
    Figure-5 sweep (notebook cell 39): raise the ECI target from
    ECI_initial + step·σ up to the final target in increments of step·σ,
    rerun the LP each time, and record the FIRST target at which every
    product is recommended (Added_vol > 0).

    growth_of_target: optional callable z-target → implied growth (%).
    Returns (tier table, dict of raw LP results by z-target).
    """
    mean_nn, sd_nn = state["mean_nn"], state["sd_nn"]
    cr = state["CountryRankings"]
    eci_init_nn = cr.loc[cr["Country"] == country, "ECI_not_normalized"].values[0]
    threshold_nn = mean_nn + sd_nn * target_z_final

    first_target = {}
    runs = {}
    target_nn = eci_init_nn + step * sd_nn
    final_run_done = False
    while True:
        z = (target_nn - mean_nn) / sd_nn
        df = run_lp(state, country, target_nn, beta_entry, beta_exit)
        runs[round(z, 4)] = df
        for code in df.loc[df["Added_vol"] > 0, "Code"]:
            first_target.setdefault(code, z)

        nxt = target_nn + step * sd_nn
        if final_run_done:
            break
        if nxt > threshold_nn:
            target_nn, final_run_done = threshold_nn, True
        else:
            target_nn = nxt

    rows = []
    final_df = runs[round(target_z_final, 4)]
    for code, z in first_target.items():
        r = final_df.loc[final_df["Code"] == code].iloc[0]
        rows.append({
            "Product": code, "first_target_ECI": round(z, 3),
            "growth_at_target": (round(growth_of_target(z), 3)
                                 if growth_of_target else np.nan),
            "RCA_2022": round(r["RCA_start"], 3),
            "RelativeRelatedness": round(r["Relative_relatedness_start"], 3),
            "PCI_2032": round(r["PCI_future"], 3),
            "Effort": round(r["Effort_Ycp"], 3),
            "Added_volume_USD": r["Added_vol"],
        })
    tiers = (pd.DataFrame(rows)
             .sort_values(["first_target_ECI", "Effort"])
             .reset_index(drop=True))
    return tiers, runs
