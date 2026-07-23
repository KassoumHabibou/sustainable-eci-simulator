"""
engine.py — Simulation logic behind the policy simulator (UI-free)
==================================================================
Everything here is plain Python/pandas so it can be tested without
Streamlit. The Streamlit app (streamlit_app.py) only handles display.

Conventions
-----------
The policymaker states objectives as GROWTH RATES of CO2 emissions and of
the unemployment rate, both constrained to be ≤ 0 (emissions and
unemployment must not increase). Internally the growth model works with
REDUCTION rates (positive = improvement): reduction = −growth.
"""

import io
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT)

import config as cfg                              # noqa: E402
from src import forward_model as fm              # noqa: E402
from src import growth_model as gm               # noqa: E402
from src import optimization as opt              # noqa: E402

try:
    import pycountry
except ImportError:                               # pragma: no cover
    pycountry = None

# Built-in ISO3 → country name table (fallback if pycountry is missing,
# and preferred for short, policymaker-friendly names).
ISO3_NAMES = {
    "AFG": "Afghanistan", "AGO": "Angola", "ALB": "Albania",
    "ARE": "United Arab Emirates", "ARG": "Argentina", "ARM": "Armenia",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan",
    "BEL": "Belgium", "BEN": "Benin", "BFA": "Burkina Faso",
    "BGD": "Bangladesh", "BGR": "Bulgaria",
    "BIH": "Bosnia and Herzegovina", "BLR": "Belarus", "BOL": "Bolivia",
    "BRA": "Brazil", "BWA": "Botswana", "CAN": "Canada",
    "CHE": "Switzerland", "CHL": "Chile", "CHN": "China",
    "CIV": "Côte d'Ivoire", "CMR": "Cameroon",
    "COD": "DR Congo", "COG": "Congo", "COL": "Colombia",
    "CRI": "Costa Rica", "CUB": "Cuba", "CZE": "Czechia",
    "DEU": "Germany", "DNK": "Denmark", "DOM": "Dominican Republic",
    "DZA": "Algeria", "ECU": "Ecuador", "EGY": "Egypt", "ESP": "Spain",
    "ETH": "Ethiopia", "FIN": "Finland", "FRA": "France", "GAB": "Gabon",
    "GBR": "United Kingdom", "GEO": "Georgia", "GHA": "Ghana",
    "GIN": "Guinea", "GRC": "Greece", "GTM": "Guatemala",
    "HKG": "Hong Kong", "HND": "Honduras", "HRV": "Croatia",
    "HUN": "Hungary", "IDN": "Indonesia", "IND": "India",
    "IRL": "Ireland", "IRN": "Iran", "IRQ": "Iraq", "ISR": "Israel",
    "ITA": "Italy", "JAM": "Jamaica", "JOR": "Jordan", "JPN": "Japan",
    "KAZ": "Kazakhstan", "KEN": "Kenya", "KGZ": "Kyrgyzstan",
    "KHM": "Cambodia", "KOR": "South Korea", "KWT": "Kuwait",
    "LAO": "Laos", "LBN": "Lebanon", "LBR": "Liberia", "LBY": "Libya",
    "LKA": "Sri Lanka", "LTU": "Lithuania", "MAR": "Morocco",
    "MDA": "Moldova", "MDG": "Madagascar", "MEX": "Mexico",
    "MKD": "North Macedonia", "MLI": "Mali", "MMR": "Myanmar",
    "MNG": "Mongolia", "MOZ": "Mozambique", "MRT": "Mauritania",
    "MWI": "Malawi", "MYS": "Malaysia", "NAM": "Namibia", "NER": "Niger",
    "NGA": "Nigeria", "NIC": "Nicaragua", "NLD": "Netherlands",
    "NOR": "Norway", "NZL": "New Zealand", "OMN": "Oman",
    "PAK": "Pakistan", "PAN": "Panama", "PER": "Peru",
    "PHL": "Philippines", "PNG": "Papua New Guinea", "POL": "Poland",
    "PRT": "Portugal", "PRY": "Paraguay", "QAT": "Qatar",
    "ROU": "Romania", "RUS": "Russia", "SAU": "Saudi Arabia",
    "SDN": "Sudan", "SEN": "Senegal", "SGP": "Singapore",
    "SLV": "El Salvador", "SRB": "Serbia", "SVK": "Slovakia",
    "SVN": "Slovenia", "SWE": "Sweden", "TCD": "Chad", "TGO": "Togo",
    "THA": "Thailand", "TJK": "Tajikistan", "TKM": "Turkmenistan",
    "TUN": "Tunisia", "TUR": "Türkiye", "TZA": "Tanzania",
    "UGA": "Uganda", "UKR": "Ukraine", "URY": "Uruguay",
    "USA": "United States", "UZB": "Uzbekistan", "VEN": "Venezuela",
    "VNM": "Vietnam", "YEM": "Yemen", "ZAF": "South Africa",
    "ZMB": "Zambia", "ZWE": "Zimbabwe",
}

# Acceptable input ranges (enforced again by the UI sliders)
RANGES = {
    "growth_target": (0.5, 8.0),     # % annualized GDP-pc growth objective
    "co2_growth": (-8.0, 0.0),       # %/yr CO2-emissions growth (≤ 0)
    "unemp_growth": (-8.0, 0.0),     # %/yr unemployment-rate growth (≤ 0)
}

SOURCES = [
    ("Trade data", "Observatory of Economic Complexity (oec.world), bilateral exports, HS4 rev. 1992, year 2022"),
    ("ECI series", "OEC published Economic Complexity Index, 1995-2022 (Data-ECI-Trade.csv)"),
    ("GDP per capita", "World Bank WDI, NY.GDP.PCAP.KD (constant 2015 USD)"),
    ("Population", "World Bank WDI, SP.POP.TOTL"),
    ("CO2 emissions", "World Bank WDI, CO2 emissions (Mt), 1990-2023"),
    ("Unemployment", "World Bank WDI / ILO modelled estimate, SL.UEM.TOTL.ZS"),
    ("Optimization engine", "Stojkoski, V. & Hidalgo, C. A., Optimizing economic complexity, Research Policy 55, 105454 (2026)"),
    ("Sustainability extension", "Ibrahim Kassoum, H. & Boly, M., Optimizing Economic Complexity Under Environmental and Employment Constraints (working paper, CERDI / World Bank)"),
]


# ---------------------------------------------------------------------------
# Cached data loaders (wrapped by st.cache_* in the UI layer)
# ---------------------------------------------------------------------------
def load_state():
    with open(os.path.join(cfg.CACHE_DIR, "state.pkl"), "rb") as f:
        return pickle.load(f)


def load_models():
    panel, gdpz_2022 = gm.build_panel(cfg.ECI_PANEL_FILE, cfg.GDP_PC_FILE,
                                      cfg.WDI_PANEL_FILE, cfg.GROWTH_T,
                                      cfg.GROWTH_YEARS)
    m_base, m_ext = gm.fit(panel)
    return {"m_base": m_base, "m_ext": m_ext, "gdpz_2022": gdpz_2022,
            "last_year": int(panel["Year"].max()),
            "n_base": int(m_base.nobs), "n_ext": int(m_ext.nobs),
            "r2_base": float(m_base.rsquared), "r2_ext": float(m_ext.rsquared)}


def load_indicators():
    wdi = pd.read_csv(cfg.WDI_PANEL_FILE)
    gdp = pd.read_csv(cfg.GDP_PC_FILE)
    pop = pd.read_csv(os.path.join(cfg.DATA_DIR, "population.csv"))
    pop.columns = ["Country", "Year", "Population"]
    eci = pd.read_csv(cfg.ECI_PANEL_FILE)
    cls = pd.read_csv(cfg.HS4_CLASS_FILE, dtype=str)
    return {"wdi": wdi, "gdp": gdp, "pop": pop, "eci": eci, "cls": cls}


def country_name(iso3):
    iso3 = iso3.upper()
    if iso3 in ISO3_NAMES:
        return ISO3_NAMES[iso3]
    if pycountry:
        rec = pycountry.countries.get(alpha_3=iso3)
        if rec:
            return getattr(rec, "common_name", None) or rec.name
    return iso3


ISO3_TO_ISO2 = {
    "AFG": "AF", "AGO": "AO", "ALB": "AL", "ARE": "AE", "ARG": "AR",
    "ARM": "AM", "AUS": "AU", "AUT": "AT", "AZE": "AZ", "BEL": "BE",
    "BEN": "BJ", "BFA": "BF", "BGD": "BD", "BGR": "BG", "BIH": "BA",
    "BLR": "BY", "BOL": "BO", "BRA": "BR", "BWA": "BW", "CAN": "CA",
    "CHE": "CH", "CHL": "CL", "CHN": "CN", "CIV": "CI", "CMR": "CM",
    "COD": "CD", "COG": "CG", "COL": "CO", "CRI": "CR", "CUB": "CU",
    "CZE": "CZ", "DEU": "DE", "DNK": "DK", "DOM": "DO", "DZA": "DZ",
    "ECU": "EC", "EGY": "EG", "ESP": "ES", "ETH": "ET", "FIN": "FI",
    "FRA": "FR", "GAB": "GA", "GBR": "GB", "GEO": "GE", "GHA": "GH",
    "GIN": "GN", "GRC": "GR", "GTM": "GT", "HKG": "HK", "HND": "HN",
    "HRV": "HR", "HUN": "HU", "IDN": "ID", "IND": "IN", "IRL": "IE",
    "IRN": "IR", "IRQ": "IQ", "ISR": "IL", "ITA": "IT", "JAM": "JM",
    "JOR": "JO", "JPN": "JP", "KAZ": "KZ", "KEN": "KE", "KGZ": "KG",
    "KHM": "KH", "KOR": "KR", "KWT": "KW", "LAO": "LA", "LBN": "LB",
    "LBR": "LR", "LBY": "LY", "LKA": "LK", "LTU": "LT", "MAR": "MA",
    "MDA": "MD", "MDG": "MG", "MEX": "MX", "MKD": "MK", "MLI": "ML",
    "MMR": "MM", "MNG": "MN", "MOZ": "MZ", "MRT": "MR", "MWI": "MW",
    "MYS": "MY", "NAM": "NA", "NER": "NE", "NGA": "NG", "NIC": "NI",
    "NLD": "NL", "NOR": "NO", "NZL": "NZ", "OMN": "OM", "PAK": "PK",
    "PAN": "PA", "PER": "PE", "PHL": "PH", "PNG": "PG", "POL": "PL",
    "PRT": "PT", "PRY": "PY", "QAT": "QA", "ROU": "RO", "RUS": "RU",
    "SAU": "SA", "SDN": "SD", "SEN": "SN", "SGP": "SG", "SLV": "SV",
    "SRB": "RS", "SVK": "SK", "SVN": "SI", "SWE": "SE", "TCD": "TD",
    "TGO": "TG", "THA": "TH", "TJK": "TJ", "TKM": "TM", "TUN": "TN",
    "TUR": "TR", "TZA": "TZ", "UGA": "UG", "UKR": "UA", "URY": "UY",
    "USA": "US", "UZB": "UZ", "VEN": "VE", "VNM": "VN", "YEM": "YE",
    "ZAF": "ZA", "ZMB": "ZM", "ZWE": "ZW",
}


def country_flag(iso3):
    iso2 = ISO3_TO_ISO2.get(iso3.upper())
    if not iso2 and pycountry:
        rec = pycountry.countries.get(alpha_3=iso3.upper())
        iso2 = rec.alpha_2 if rec else None
    if iso2:
        return "".join(chr(0x1F1E6 + ord(ch) - 65) for ch in iso2)
    return ""


def available_countries(state, models):
    """Countries that can be simulated: in the trade state AND with a
    2022 GDP z-score (needed to invert the growth equation)."""
    out = []
    for c in state["countries"]:
        iso3 = c.upper()
        gz = models["gdpz_2022"].get(iso3, np.nan)
        if not (isinstance(gz, float) and np.isnan(gz)):
            out.append({"code": c, "iso3": iso3, "name": country_name(iso3)})
    return sorted(out, key=lambda x: x["name"])


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def run_policy_simulation(state, models, indicators, c,
                          growth_target, co2_growth, unemp_growth):
    """
    c              : lowercase iso3 (e.g. 'tha')
    growth_target  : desired annualized GDP-pc growth, %
    co2_growth     : desired CO2-emissions growth, %/yr (must be ≤ 0)
    unemp_growth   : desired unemployment-rate growth, %/yr (must be ≤ 0)
    """
    # ── validate ranges ────────────────────────────────────────────────────
    for key, val in [("growth_target", growth_target),
                     ("co2_growth", co2_growth),
                     ("unemp_growth", unemp_growth)]:
        lo, hi = RANGES[key]
        if not (lo <= val <= hi):
            raise ValueError(f"{key} = {val} outside acceptable range [{lo}, {hi}]")
    if co2_growth > 0 or unemp_growth > 0:
        raise ValueError("CO2 and unemployment growth rates must be 0 or negative.")

    co2red, unred = -co2_growth, -unemp_growth          # policymaker → model
    iso3 = c.upper()
    gdpz = float(models["gdpz_2022"][iso3])
    m_ext, m_base, ly = models["m_ext"], models["m_base"], models["last_year"]

    # ── targets ────────────────────────────────────────────────────────────
    eci_now = fm.eci_z(state, c)                        # predicted 2032, no action
    t_raw = gm.invert_for_eci(m_ext, growth_target, gdpz, ly,
                              co2red=co2red, unred=unred)
    z_max = _max_feasible(state, c)
    capped = t_raw > z_max - 0.02
    t_used = min(t_raw, z_max - 0.02)
    g_att = gm.predict_growth(m_ext, t_used, gdpz, ly, co2red=co2red, unred=unred)
    g_noopt = gm.predict_growth(m_ext, eci_now, gdpz, ly, co2red=co2red, unred=unred)
    g_base = gm.predict_growth(m_base, eci_now, gdpz, ly)

    # ── optimization ───────────────────────────────────────────────────────
    target_nn = state["mean_nn"] + state["sd_nn"] * t_used
    lp = opt.run_lp(state, c, target_nn,
                    state["beta_entry"], state["beta_exit"])
    sel = lp[lp["Added_vol"] > 0].copy()

    cls = indicators["cls"].set_index("code_trade")
    sel["Product"] = sel["Code"].astype(str)
    sel["Name"] = sel["Product"].map(cls["product_name"]).fillna("")
    sel["Sector (HS2)"] = sel["Product"].map(cls["hs2_name"]).fillna("")
    sel["Group"] = sel["Product"].map(cls["section"]).fillna("")

    # ── GDP scaling ────────────────────────────────────────────────────────
    gdp_pc, pop, gdp_total = _latest_gdp(indicators, iso3)
    sel["Added exports (USD M)"] = sel["Added_vol"] / 1e6
    sel["Share of GDP (%)"] = (sel["Added_vol"] / gdp_total * 100
                               if gdp_total else np.nan)
    sel = sel.sort_values("Effort_Ycp")
    sel["Priority"] = range(1, len(sel) + 1)

    products = sel[["Priority", "Product", "Name", "Sector (HS2)", "Group",
                    "RCA_start", "PCI_future", "Effort_Ycp",
                    "Added exports (USD M)", "Share of GDP (%)"]].rename(
        columns={"RCA_start": "RCA 2022", "PCI_future": "Estimated PCI 2032",
                 "Effort_Ycp": "Effort"}).reset_index(drop=True)

    invest = float(sel["Added_vol"].sum())
    summary = {
        "Country": country_name(iso3), "ISO3": iso3,
        "Growth objective (%/yr)": growth_target,
        "CO2 emissions growth objective (%/yr)": co2_growth,
        "Unemployment growth objective (%/yr)": unemp_growth,
        "Predicted 2032 ECI without action": round(eci_now, 3),
        "ECI required for the objective": round(t_raw, 3),
        "Maximum feasible ECI": round(z_max, 3),
        "Objective feasible?": "No — capped at the feasibility frontier" if capped else "Yes",
        "ECI target used": round(t_used, 3),
        "Attainable growth under the scenario (%/yr)": round(g_att, 2),
        "Growth without new products, same scenario (%/yr)": round(g_noopt, 2),
        "Growth without new products, no constraints (%/yr)": round(g_base, 2),
        "Gain from diversification (pp/yr)": round(g_att - g_noopt, 2),
        "Number of recommended products": int(len(sel)),
        "Required investment (USD bn added exports)": round(invest / 1e9, 2),
        "Investment as share of GDP (%)": round(invest / gdp_total * 100, 2) if gdp_total else None,
        "GDP (USD bn, latest)": round(gdp_total / 1e9, 1) if gdp_total else None,
    }
    return {"products": products, "summary": summary, "capped": capped,
            "lp_all": lp, "t_used": t_used, "t_raw": t_raw, "z_max": z_max,
            "g_att": g_att, "g_noopt": g_noopt, "invest": invest}


def _max_feasible(state, c):
    i = state["countries"].index(c)
    forced = state["M_2032"][i, :] > 0
    banned = (state["RCA_2022"][i, :] > 1) & (~forced)
    cand = np.sort(state["PCI_future"][~forced & ~banned])[::-1]
    pci_f = state["PCI_future"][forced]
    best, s, n = pci_f.mean(), pci_f.sum(), pci_f.size
    for v in cand:
        s, n = s + v, n + 1
        best = max(best, s / n)
    return (best - state["mean_nn"]) / state["sd_nn"]


def _latest_gdp(indicators, iso3):
    gdp = indicators["gdp"]
    pop = indicators["pop"]
    g = gdp[gdp["Country"] == iso3].dropna(subset=["GDP_pc"]).sort_values("Year")
    p = pop[pop["Country"] == iso3].dropna(subset=["Population"]).sort_values("Year")
    gdp_pc = float(g["GDP_pc"].iloc[-1]) if len(g) else None
    popv = float(p["Population"].iloc[-1]) if len(p) else None
    total = gdp_pc * popv if gdp_pc and popv else None
    return gdp_pc, popv, total


def latest_indicators(indicators, iso3):
    """Latest observed values for the country dashboard."""
    wdi = indicators["wdi"]
    sub = wdi[wdi["Country"] == iso3]
    gdp_pc, pop, gdp_total = _latest_gdp(indicators, iso3)

    def last(col):
        s = sub.dropna(subset=[col]).sort_values("Year")
        return (float(s[col].iloc[-1]), int(s["Year"].iloc[-1])) if len(s) else (None, None)

    eci, eci_y = last("ECI")
    co2, co2_y = last("CO2_Emissions_Mt")
    co2pc, _ = last("CO2_Emissions_Per_Capita")
    un, un_y = last("Unemployment_Rate_ILO")
    return {"gdp_pc": gdp_pc, "population": pop, "gdp_total": gdp_total,
            "eci": eci, "eci_year": eci_y, "co2": co2, "co2_year": co2_y,
            "co2_pc": co2pc, "unemp": un, "unemp_year": un_y}


def timeseries(indicators, iso3):
    wdi = indicators["wdi"]
    sub = wdi[wdi["Country"] == iso3].sort_values("Year")
    gdp = indicators["gdp"]
    g = gdp[gdp["Country"] == iso3].sort_values("Year")
    return {
        "eci": sub[["Year", "ECI"]].dropna(),
        "co2": sub[["Year", "CO2_Emissions_Mt"]].dropna(),
        "unemp": sub[["Year", "Unemployment_Rate_ILO"]].dropna(),
        "gdp_pc": g[["Year", "GDP_pc"]].dropna(),
        "sectors": sub[["Year", "Employment_Agriculture_Pct",
                        "Employment_Industry_Pct",
                        "Employment_Services_Pct"]].dropna(),
    }


# ---------------------------------------------------------------------------
# Trade structure (precomputed country × product × year aggregates)
# ---------------------------------------------------------------------------
def load_trade():
    """Compact export/import panels built by build_trade_aggregates.py."""
    exp = pd.read_parquet(os.path.join(cfg.DATA_DIR, "trade_exports.parquet"))
    imp = pd.read_parquet(os.path.join(cfg.DATA_DIR, "trade_imports.parquet"))
    years = sorted(int(y) for y in exp["year"].unique())
    return {"exports": exp, "imports": imp, "years": years}


def country_trade_table(trade, cls, flow, code, year):
    """All products a country trades in `year`, ordered by value.
    flow: 'exports' or 'imports'; code: lowercase iso3."""
    df = trade[flow]
    sub = df[(df["country"] == code) & (df["year"] == year)]
    total = float(sub["value"].sum())
    out = (sub.groupby("hs_code", observed=True, as_index=False)["value"]
           .sum().sort_values("value", ascending=False))
    out["hs_code"] = out["hs_code"].astype(str)
    m = cls.set_index("code_trade")
    out.insert(1, "Product", out["hs_code"].map(m["product_name"]).fillna(""))
    out.insert(2, "Sector", out["hs_code"].map(m["section"]).fillna(""))
    out["Value (USD M)"] = out["value"] / 1e6
    out["Share of country " + flow + " (%)"] = (
        out["value"] / total * 100 if total else np.nan)
    out = out.drop(columns="value").rename(columns={"hs_code": "Code"})
    return out.reset_index(drop=True), total


def product_world_shares(trade, flow, hs_code, year):
    """Each country's share (%) of WORLD trade in one product and year.
    Returns (DataFrame[iso3, value, share_pct], world_total_usd)."""
    df = trade[flow]
    sub = df[(df["hs_code"] == hs_code) & (df["year"] == year)]
    world = float(sub["value"].sum())
    out = (sub.groupby("country", observed=True, as_index=False)["value"]
           .sum())
    out["iso3"] = out["country"].astype(str).str.upper()
    out["share_pct"] = out["value"] / world * 100 if world else np.nan
    out["name"] = out["iso3"].map(lambda x: country_name(x))
    return out.sort_values("share_pct", ascending=False), world


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------
def build_excel(result, models):
    """Return an in-memory .xlsx with products, summary, metadata, sources."""
    buf = io.BytesIO()
    s = result["summary"]
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # 1 — README / metadata
        meta_rows = [
            ("Tool", "Sustainable Diversification Simulator"),
            ("Generated on", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Country", f"{s['Country']} ({s['ISO3']})"),
            ("Growth objective (%/yr)", s["Growth objective (%/yr)"]),
            ("CO2 emissions growth objective (%/yr)", s["CO2 emissions growth objective (%/yr)"]),
            ("Unemployment growth objective (%/yr)", s["Unemployment growth objective (%/yr)"]),
            ("", ""),
            ("METHOD", ""),
            ("Growth anchor", "Panel growth regression (10-year windows 1999-2009, 2009-2019) of GDP-pc growth on ECI, initial income, CO2-emission reduction pace, unemployment reduction pace, and interactions; period fixed effects, HC1 errors."),
            ("Model fit", f"baseline R2={models['r2_base']:.3f} (n={models['n_base']}), extended R2={models['r2_ext']:.3f} (n={models['n_ext']})"),
            ("Target setting", "The equation is inverted at the stated objectives to obtain the required ECI. If it exceeds the country's maximum feasible ECI, the target is capped at the feasibility frontier and the attainable growth is reported."),
            ("Product selection", "0-1 integer program (Stojkoski & Hidalgo 2026): minimize total steppingstone effort subject to the portfolio's average predicted 2032 PCI reaching the ECI target."),
            ("Effort", "Additional RCA the country must gain by 2027 for the model to predict competitive exports (RCA>=1) in 2032."),
            ("Share of GDP", "Added export volume required for the product, divided by latest GDP (GDP per capita x population, constant 2015 USD)."),
            ("Caveats", "The growth anchor is correlational; scenario inversion is an extrapolation, not a causal estimate. Recommendations should be read alongside country knowledge."),
            ("", ""),
            ("SOURCES", ""),
            *SOURCES,
        ]
        pd.DataFrame(meta_rows, columns=["Field", "Value"]).to_excel(
            xw, sheet_name="README & Sources", index=False)

        # 2 — Summary
        pd.DataFrame(list(s.items()), columns=["Indicator", "Value"]).to_excel(
            xw, sheet_name="Simulation summary", index=False)

        # 3 — Products
        result["products"].to_excel(xw, sheet_name="Recommended products", index=False)

        # column widths
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 90)
    buf.seek(0)
    return buf.getvalue()
