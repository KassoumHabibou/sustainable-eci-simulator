"""
growth_model.py — Paper Eq. 3 (baseline) and the EXTENSION (CO2 + employment)
=============================================================================
Baseline (paper, Table S1 / notebook cell 36):

    growth_ct = a0 + a1·ECI_ct + a2·gdpz_ct + a3·ECI_ct·gdpz_ct + γ_t + u_ct

  * growth = 100·((GDPpc_{t+10}/GDPpc_t)^{1/10} − 1), windows 1999-2009, 2009-2019
  * gdpz   = per-year z-score of log GDP per capita (across the ECI sample)
  * year fixed effects, HC1 robust standard errors

EXTENSION — two sustainability margins, both expressed as POSITIVE POLICY
OUTCOMES (a policymaker wants both numbers to be large):

    … + a4·CO2red_ct + a5·Ured_ct + a6·ECI·CO2red + a7·ECI·Ured

  * CO2red = −100·((CO2_{t+10}/CO2_t)^{1/10} − 1)
             annualized rate of CO2-emission REDUCTION (positive = falling)
  * Ured   = −100·((U_{t+10}/U_t)^{1/10} − 1)
             annualized rate of UNEMPLOYMENT REDUCTION over the same window
             (positive = unemployment falling), U = ILO unemployment rate

Inversion: Eq. 3 is linear in ECI → growth = A(x) + B(x)·ECI.
Fixing the controls (GDP at the country's 2022 value; the two reduction
rates at the policy targets) gives ECI* = (g_target − A) / B: the
complexity a country needs to reach the SAME growth target while
delivering the chosen emission and unemployment reductions.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import zscore


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
def build_panel(eci_file, gdp_file, wdi_file, T, window_years):
    """Return (panel DataFrame, gdpz_2022 dict ISO3 → z-scored log GDPpc 2022)."""
    # ── ECI (published, wide) ───────────────────────────────────────────────
    eci = pd.read_csv(eci_file)
    eci.columns = [str(c) for c in eci.columns]

    # ── GDP per capita (long → wide) ────────────────────────────────────────
    gdp = pd.read_csv(gdp_file)
    gdp_w = gdp.pivot_table(index="Country", columns="Year",
                            values="GDP_pc").reset_index()
    gdp_w.columns = [str(c) for c in gdp_w.columns]
    gdp_w = gdp_w[gdp_w["Country"].isin(eci["Country"])].reset_index(drop=True)
    eci = eci[eci["Country"].isin(gdp_w["Country"])].reset_index(drop=True)
    eci = eci.set_index("Country").reindex(gdp_w["Country"]).reset_index()

    years_avail = [c for c in gdp_w.columns if c.isdigit()]

    # z-score of log GDPpc per year, restricted to countries with ECI that
    # year (the notebook's NaN alignment step).
    gdpz_w = gdp_w.copy()
    for y in years_avail:
        if y in eci.columns:
            gdpz_w[y] = gdp_w[y].where(~eci[y].isna())
            gdpz_w[y] = zscore(np.log(gdpz_w[y]), nan_policy="omit")
        else:
            gdpz_w[y] = np.nan

    # 10-year annualized GDPpc growth from year t
    def ann_growth(w, t, col):
        t2 = str(int(t) + T)
        if t2 not in w.columns:
            return pd.Series(np.nan, index=w.index)
        return 100.0 * ((w[t2] / w[t]) ** (1.0 / T) - 1.0)

    # ── CO2 and unemployment (WDI panel, long → wide) ──────────────────────
    wdi = pd.read_csv(wdi_file)
    co2_w = wdi.pivot_table(index="Country", columns="Year",
                            values="CO2_Emissions_Mt")
    co2_w.columns = [str(c) for c in co2_w.columns]
    un_w = wdi.pivot_table(index="Country", columns="Year",
                           values="Unemployment_Rate_ILO")
    un_w.columns = [str(c) for c in un_w.columns]

    # ── Assemble the stacked panel ─────────────────────────────────────────
    # Both sustainability margins are annualized REDUCTION rates over the
    # window (positive = improvement, the policymaker's convention).
    def ann_reduction(w, t):
        t2 = str(int(t) + T)
        if t2 not in w.columns:
            return pd.Series(np.nan, index=w.index)
        return -100.0 * ((w[t2] / w[t]) ** (1.0 / T) - 1.0)

    rows = []
    for t in window_years:
        t = str(t)
        block = pd.DataFrame({
            "Country": gdp_w["Country"],
            "Year": int(t),
            "eci": eci[t].values if t in eci.columns else np.nan,
            "gdpz": gdpz_w[t].values,
            "growth": ann_growth(gdp_w, t, t).values,
        })
        block["co2red"] = block["Country"].map(ann_reduction(co2_w, t))
        block["unred"]  = block["Country"].map(ann_reduction(un_w, t))
        rows.append(block)
    panel = pd.concat(rows, ignore_index=True)

    # 2022 z-scored log GDPpc (for prediction/inversion at the current state)
    gdpz_2022 = dict(zip(gdpz_w["Country"], gdpz_w.get("2022", np.nan)))

    return panel, gdpz_2022


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
FORMULA_BASE = "growth ~ gdpz * eci + C(Year)"
FORMULA_EXT  = ("growth ~ gdpz * eci + co2red + unred"
                " + eci:co2red + eci:unred + C(Year)")


def fit(panel):
    """Fit baseline and extended Eq. 3 on the SAME estimation sample."""
    sample_b = panel.dropna(subset=["growth", "eci", "gdpz"])
    sample_e = panel.dropna(subset=["growth", "eci", "gdpz", "co2red", "unred"])
    m_base = smf.ols(FORMULA_BASE, data=sample_b).fit(cov_type="HC1")
    m_ext  = smf.ols(FORMULA_EXT,  data=sample_e).fit(cov_type="HC1")
    return m_base, m_ext


def coef_table(model, label):
    return pd.DataFrame({
        "model": label, "term": model.params.index,
        "coef": model.params.values, "se": model.bse.values,
        "pvalue": model.pvalues.values,
    })


# ---------------------------------------------------------------------------
# Linear decomposition  growth = A + B·ECI  →  prediction & inversion
# ---------------------------------------------------------------------------
def _p(model, name):
    return float(model.params.get(name, 0.0))


def _AB_base(model, gdpz, last_year):
    A = _p(model, "Intercept") + _p(model, f"C(Year)[T.{last_year}]") \
        + _p(model, "gdpz") * gdpz
    B = _p(model, "eci") + _p(model, "gdpz:eci") * gdpz
    return A, B


def _AB_ext(model, gdpz, co2red, unred, last_year):
    A = (_p(model, "Intercept") + _p(model, f"C(Year)[T.{last_year}]")
         + _p(model, "gdpz") * gdpz
         + _p(model, "co2red") * co2red + _p(model, "unred") * unred)
    B = (_p(model, "eci") + _p(model, "gdpz:eci") * gdpz
         + _p(model, "eci:co2red") * co2red + _p(model, "eci:unred") * unred)
    return A, B


def predict_growth(model, eci, gdpz, last_year, co2red=None, unred=None):
    if co2red is None:
        A, B = _AB_base(model, gdpz, last_year)
    else:
        A, B = _AB_ext(model, gdpz, co2red, unred, last_year)
    return A + B * eci


def invert_for_eci(model, g_target, gdpz, last_year, co2red=None, unred=None):
    if co2red is None:
        A, B = _AB_base(model, gdpz, last_year)
    else:
        A, B = _AB_ext(model, gdpz, co2red, unred, last_year)
    if abs(B) < 1e-12:
        raise ValueError("dGrowth/dECI is zero — cannot invert Eq. 3.")
    return (g_target - A) / B
