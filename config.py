"""
config.py — Single source of truth for paths and parameters
============================================================
Extension of Stojkoski & Hidalgo, "Optimizing Economic Complexity":
CO2-emissions growth and the unemployment rate are added to the growth
regression (paper Eq. 3). The forecast model (paper Eq. 2) and the
0-1 integer optimization (the LP) remain EXACTLY the authors' spec.

All parameters below follow the paper's main specification:
  * horizon Δt = 10 years, steppingstone τ = 5 years
  * 2022 trade data  →  predict 2032, steppingstone 2027
  * growth regression on 10-year windows 1999-2009 and 2009-2019
  * target: 3.5 % annualized GDP-per-capita growth (paper Figure 5)
"""

import os

# ---------------------------------------------------------------------------
# Paths (everything is relative to this folder → fully portable)
# ---------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "data")
SRC_DIR     = os.path.join(ROOT, "src")
RESULTS_DIR = os.path.join(ROOT, "results")
CACHE_DIR   = os.path.join(RESULTS_DIR, "cache")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")

TRADE_2022_FILE = os.path.join(DATA_DIR, "bilateral_2022.csv")       # OEC HS4 bilateral trade, 2022
ECI_PANEL_FILE  = os.path.join(DATA_DIR, "Data-ECI-Trade.csv")       # published ECI, 1995-2022 (wide)
GDP_PC_FILE     = os.path.join(DATA_DIR, "gdp_per_capita.csv")       # WDI NY.GDP.PCAP.KD (long)
WDI_PANEL_FILE  = os.path.join(DATA_DIR, "merged_eci_wdi_co2.csv")   # WDI CO2 + employment panel
BETA_ENTRY_FILE = os.path.join(DATA_DIR, "beta_eq2_entry.csv")       # Eq. 2 OLS betas (entry)
BETA_EXIT_FILE  = os.path.join(DATA_DIR, "beta_eq2_exit.csv")        # Eq. 2 OLS betas (exit)

for _d in (RESULTS_DIR, CACHE_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Paper parameters — forecast model / LP
# ---------------------------------------------------------------------------
END_YEAR = 2022          # "current" snapshot → forecast horizon 2032
DELTA_T  = 10            # Δt  (the Eq. 2 betas in data/ were estimated with this)
TAU      = 5             # τ   (steppingstone 2027)

MIN_COUNTRY_EXPORTS = 1e9   # authors' notebook filter: country total exports ≥ $1B
MIN_PRODUCT_EXPORTS = 5e5   # authors' notebook filter: product world exports ≥ $500K

# ---------------------------------------------------------------------------
# Paper parameters — growth regression (Eq. 3)
# ---------------------------------------------------------------------------
GROWTH_T     = 10                 # 10-year growth windows
GROWTH_YEARS = [1999, 2009]       # windows 1999-2009 and 2009-2019 (paper)
TARGET_GROWTH = 3.5               # % annualized GDP-pc growth target (paper Fig. 5)

# ---------------------------------------------------------------------------
# Extension parameters — sustainability scenario plugged into extended Eq. 3
# Both margins are expressed as REDUCTION RATES: positive = policy success.
# ---------------------------------------------------------------------------
# CO2: the Paris Agreement requires emissions to decline. We target an
# annualized emission REDUCTION of +2 %/yr over the 10-year horizon — an
# NDC-style pace for middle-income economies (the IPCC-implied global
# pace to 2030, ~+5 %/yr of reduction, is faster).
CO2_REDUCTION_TARGET = 2.0    # % annualized CO2-emission reduction (+ = falling)

# Employment (SDG 8): the unemployment rate must fall over the horizon.
# We target an annualized reduction of the rate of +2 %/yr, i.e. roughly
# 20 % lower unemployment after a decade.
UNEMP_REDUCTION_TARGET = 2.0  # % annualized reduction of the unemployment rate

# Full HS4 classification (name, HS2 chapter = type, HS section = group)
HS4_CLASS_FILE = os.path.join(DATA_DIR, "hs4_classification.csv")

# ---------------------------------------------------------------------------
# Simulation cases
# ---------------------------------------------------------------------------
TARGET_COUNTRIES = {"tha": "THA", "mex": "MEX"}
COUNTRY_NAMES    = {"tha": "Thailand", "mex": "Mexico"}

# Figure-5 sweep: increment of the ECI target, in std. dev. of the
# (future) not-normalized ECI distribution — exactly as the notebook.
SWEEP_STEP = 0.05

# ---------------------------------------------------------------------------
# Paper benchmarks used for VALIDATION ONLY (Figure 5 of the paper)
# ---------------------------------------------------------------------------
PAPER_ECI_TARGET   = {"tha": 1.225, "mex": 1.287}
PAPER_ECI_2032     = {"tha": 0.933, "mex": 0.926}   # forecast without optimization
PAPER_GROWTH_EXP   = {"tha": 3.23,  "mex": 3.15}    # expected growth at predicted ECI

# Products listed in paper Figure 5c (Thailand) and 5d (Mexico),
# in the order of the first ECI target at which they are recommended.
PAPER_PRODUCTS = {
    "tha": ["7320", "8501", "2927", "8523", "8532",           # first tier  (ECI ≈ 0.98)
            "4006", "8525", "8455", "8481", "8428",            # middle tiers (ECI ≈ 1.08-1.13)
            "8101", "4008", "3819", "8477", "9031"],           # last tier   (ECI ≈ 1.23)
    "mex": ["8547", "8485", "6810", "6815", "8484",            # first tier  (ECI ≈ 0.98)
            "8423", "8707", "8483", "3801", "7609",            # middle tier (ECI ≈ 1.13)
            "9030", "8428", "8480", "8433", "7211"],           # last tier   (ECI ≈ 1.28-1.29)
}

# Human-readable HS4 labels for products that appear in the results.
HS4_NAMES = {
    "2523": "Cement", "2927": "Diazo/azo compounds", "3801": "Artificial graphite",
    "3819": "Hydraulic brake fluids", "4006": "Unvulcanised rubber forms",
    "4008": "Rubber plates & sheets", "6810": "Cement/concrete articles",
    "6815": "Stone & mineral articles", "7211": "Flat-rolled iron (narrow)",
    "7320": "Iron springs", "7609": "Aluminium tube fittings", "8101": "Tungsten",
    "8423": "Weighing machinery", "8428": "Lifting/handling machinery",
    "8433": "Harvesting machinery", "8455": "Metal-rolling mills",
    "8477": "Rubberworking machinery", "8481": "Taps, cocks & valves",
    "8483": "Transmission shafts", "8484": "Gaskets",
    "8485": "Additive manufacturing machines", "8501": "Electric motors & generators",
    "8523": "Unrecorded media",
    "8525": "Transmission apparatus", "8532": "Electrical capacitors",
    "8547": "Insulating fittings (metal)", "8707": "Vehicle bodies",
    "9030": "Electrical measuring instruments", "9031": "Measuring instruments n.e.c.",
}
