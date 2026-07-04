"""
forward_model.py — Authors' forward model (notebook cell 28), faithful
======================================================================
From the 2022 trade matrix, build everything the LP needs:

  1. 2022 RCA, specialization matrix M(2022), relatedness, PCI(2022).
  2. Apply Eq. 2 with RCA_mid = RCA_start = RCA(2022) ("W = 0" forecast)
     → predicted RCA in 2032 → predicted specialization matrix M(2032).
  3. Recompute complexity on M(2032) → FUTURE PCI (the "Estimated PCI in
     2032" of paper Figure 5) and the future proximity matrix PHIpp.
  4. ECI_not_normalized (mean future PCI of predicted specializations)
     and its cross-country mean/std, used to convert the growth-model
     ECI target (z units) into the LP target (mean-PCI units).

Everything mirrors the authors' reproducibility notebook exactly,
including the use of ddof=0 z-scores computed separately within the
entry (RCA<1) and exit (RCA≥1) samples of each country.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

from . import complexity as cx


# ---------------------------------------------------------------------------
# Trade matrix
# ---------------------------------------------------------------------------
def load_trade_matrix(path, min_country=1e9, min_product=5e5):
    """Pivot bilateral trade into an exporter × HS4 matrix, authors' filters."""
    df = pd.read_csv(path, usecols=["exporter_id", "hs_code", "value"],
                     dtype={"hs_code": str})
    X = df.pivot_table(values="value", index="exporter_id",
                       columns="hs_code", aggfunc="sum", fill_value=0.0)
    X = X[X.sum(axis=1) >= min_country]
    X = X.loc[:, X.sum(axis=0) >= min_product]
    return X


# ---------------------------------------------------------------------------
# Entry/exit relative relatedness (within-country z-scores, ddof=0)
# ---------------------------------------------------------------------------
def _relative_relatedness(RCA, Rel):
    """z-score of relatedness within each country, separately for the
    entry (RCA<1) and exit (RCA≥1) product sets — as in the notebook."""
    C, P = RCA.shape
    out = np.full((C, P), np.nan)
    for i in range(C):
        for mask in (RCA[i] < 1, RCA[i] >= 1):
            vals = Rel[i, mask]
            if vals.size > 1 and np.nanstd(vals) > 0:
                out[i, mask] = zscore(vals)          # ddof = 0, notebook default
            else:
                out[i, mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# Full state construction
# ---------------------------------------------------------------------------
def build_state(X_end, beta_entry, beta_exit):
    """
    Returns a dict with the current (2022) and predicted (2032) state.
    beta_* order: [Intercept, log_RCA_mid, log_RCA_start, Relatedness, RelRelatedness]
    """
    countries = list(X_end.index)
    products  = list(X_end.columns)
    X = X_end.values.astype(float)

    # ── 2022 state ─────────────────────────────────────────────────────────
    RCA_2022 = cx.rca(X)
    M_2022   = (RCA_2022 > 1).astype(float)                      # notebook: strict >
    CR22, PR22, Rel_2022, _ = cx.cplex_rank(M_2022, countries, products)
    PCI_2022 = PR22["PCI"].values
    ECI_nn_2022 = cx.eci_not_normalized(M_2022, PCI_2022)

    relrel = _relative_relatedness(RCA_2022, Rel_2022)

    # ── Eq. 2 forward prediction, W = 0 (RCA_mid = RCA_start = 2022) ──────
    l = np.log1p(RCA_2022)
    lp_entry = (beta_entry[0] + beta_entry[1] * l + beta_entry[2] * l
                + beta_entry[3] * Rel_2022 + beta_entry[4] * relrel)
    lp_exit  = (beta_exit[0]  + beta_exit[1]  * l + beta_exit[2]  * l
                + beta_exit[3]  * Rel_2022 + beta_exit[4]  * relrel)
    pred_RCA_2032 = np.where(RCA_2022 < 1, np.expm1(lp_entry), np.expm1(lp_exit))

    M_2032 = (pred_RCA_2032 > 1).astype(float)                   # notebook: strict >

    # ── 2032 (predicted) complexity ────────────────────────────────────────
    CR_f, PR_f, _, PHIpp_f = cx.cplex_rank(M_2032, countries, products)
    PCI_future = PR_f["PCI"].values
    ECI_nn_future = cx.eci_not_normalized(M_2032, PCI_future)
    mean_nn = np.nanmean(ECI_nn_future)
    sd_nn   = np.nanstd(ECI_nn_future)

    # CountryRankings table consumed by the authors' `eci_optimization`
    CountryRankings = pd.DataFrame({
        "Country": countries,
        "ECI": CR_f["ECI"].values,                    # future eigen-ECI (z)
        "ECI_2022": CR22["ECI"].values,
        "ECI_not_normalized": ECI_nn_future,          # future mean-PCI scale
        "ECI_not_normalized_2022": ECI_nn_2022,
    })

    W_p = X.sum(axis=0) / X.sum()

    return {
        "countries": countries, "products": products, "X": X,
        "RCA_2022": RCA_2022, "M_2022": M_2022, "Rel_2022": Rel_2022,
        "relrel_2022": relrel, "PCI_2022": PCI_2022,
        "pred_RCA_2032": pred_RCA_2032, "M_2032": M_2032,
        "PCI_future": PCI_future, "PHIpp_future": PHIpp_f,
        "CountryRankings": CountryRankings,
        "mean_nn": mean_nn, "sd_nn": sd_nn, "W_p": W_p,
        "X_p": X.sum(axis=0),
    }


def product_table(state, country):
    """Per-country ProductRankings table expected by `eci_optimization`."""
    i = state["countries"].index(country)
    pr = pd.DataFrame({
        "Product": state["products"],
        "PCI": state["PCI_future"],                            # future PCI!
        "X_start": state["X"][i, :],
        "Relatedness_start": state["Rel_2022"][i, :],
        "Relative_relatedness_start": state["relrel_2022"][i, :],
        "predicted_prob": state["pred_RCA_2032"][i, :],
        "X_p_start": state["X_p"],
        "W_p": state["W_p"],
        "RCA_start": state["RCA_2022"][i, :],
        "M_start": state["M_2032"][i, :],                      # PREDICTED 2032 M
    })
    return pr


def eci_z(state, country):
    """Country's predicted-2032 ECI in z units of the future nn-ECI distribution."""
    row = state["CountryRankings"]
    nn = row.loc[row["Country"] == country, "ECI_not_normalized"].values[0]
    return (nn - state["mean_nn"]) / state["sd_nn"]
