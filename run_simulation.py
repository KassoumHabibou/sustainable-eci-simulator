"""
run_simulation.py — Complete simulation: paper baseline vs CO2/employment extension
===================================================================================
Reproduces the paper's Figure-5 exercise for Thailand and Mexico (the
"old approach") and runs the EXTENDED approach in which CO2-emissions
growth and the unemployment rate enter the growth regression (Eq. 3).

Pipeline
--------
  1. Forward model (authors', unchanged): 2022 trade → predicted 2032
     specialization → future PCI, future proximity matrix.
  2. Growth regression (Eq. 3):
       baseline  = paper spec  (ECI, GDPpc, interaction, year FE)
       extended  = + CO2 growth + unemployment + interactions with ECI
  3. Invert both regressions at a 3.5 % growth target → two ECI targets.
  4. Authors' LP (unchanged) at each target → two optimal product portfolios.
  5. Figure-5 style target sweep → the order in which products enter.
  6. Validation of the baseline against the paper's published results.

Run:  python run_simulation.py            (≈ 5-10 minutes)
Outputs land in results/ (tables) and results/figures/ (plots).
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src import forward_model as fm
from src import growth_model as gm
from src import optimization as opt


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Forward model (cached)
# ═══════════════════════════════════════════════════════════════════════════
def get_state():
    cache = os.path.join(cfg.CACHE_DIR, "state.pkl")
    if os.path.exists(cache):
        print("Step 1 — Forward model: loading cache")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("Step 1 — Forward model: building from 2022 trade data …")
    beta_entry = pd.read_csv(cfg.BETA_ENTRY_FILE)["beta"].values
    beta_exit = pd.read_csv(cfg.BETA_EXIT_FILE)["beta"].values
    X_end = fm.load_trade_matrix(cfg.TRADE_2022_FILE,
                                 cfg.MIN_COUNTRY_EXPORTS, cfg.MIN_PRODUCT_EXPORTS)
    print(f"  2022 matrix: {X_end.shape[0]} countries x {X_end.shape[1]} products")
    state = fm.build_state(X_end, beta_entry, beta_exit)
    state["beta_entry"], state["beta_exit"] = beta_entry, beta_exit
    with open(cache, "wb") as f:
        pickle.dump(state, f)
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Growth regressions (baseline + extended)
# ═══════════════════════════════════════════════════════════════════════════
def get_growth_models():
    print("\nStep 2 — Growth regression (Eq. 3), baseline vs extended")
    panel, gdpz_2022 = gm.build_panel(cfg.ECI_PANEL_FILE, cfg.GDP_PC_FILE,
                                      cfg.WDI_PANEL_FILE, cfg.GROWTH_T,
                                      cfg.GROWTH_YEARS)
    m_base, m_ext = gm.fit(panel)
    print(f"  baseline: n={int(m_base.nobs)}, R2={m_base.rsquared:.3f}   "
          f"extended: n={int(m_ext.nobs)}, R2={m_ext.rsquared:.3f}")

    tbl = pd.concat([gm.coef_table(m_base, "baseline"),
                     gm.coef_table(m_ext, "extended")])
    # Significance markers: *** p<0.01, ** p<0.05, * p<0.1
    tbl["signif"] = np.select(
        [tbl["pvalue"] < 0.01, tbl["pvalue"] < 0.05, tbl["pvalue"] < 0.10],
        ["***", "**", "*"], default="ns")
    tbl["significant_at_5pct"] = tbl["pvalue"] < 0.05
    tbl.to_csv(os.path.join(cfg.RESULTS_DIR, "growth_regressions.csv"),
               index=False)
    last_year = int(panel["Year"].max())
    return m_base, m_ext, gdpz_2022, last_year


# ═══════════════════════════════════════════════════════════════════════════
# Sustainability scenario — country-specific plug values
# ═══════════════════════════════════════════════════════════════════════════
def sustainability_targets():
    """Scenario in the policymaker's convention — both margins are
    annualized REDUCTION rates, positive = success: CO2 emissions falling
    (Paris-aligned) and the unemployment rate falling (SDG 8)."""
    wdi = pd.read_csv(cfg.WDI_PANEL_FILE)
    latest = (wdi.dropna(subset=["Unemployment_Rate_ILO"])
              .sort_values("Year").groupby("Country").last())
    rows = {}
    for c, iso3 in cfg.TARGET_COUNTRIES.items():
        u_now = float(latest.loc[iso3, "Unemployment_Rate_ILO"])
        rows[c] = {
            "co2_reduction": cfg.CO2_REDUCTION_TARGET,
            "unemp_reduction": cfg.UNEMP_REDUCTION_TARGET,
            "unemp_now": u_now,
            "unemp_in_10y": round(
                u_now * (1 - cfg.UNEMP_REDUCTION_TARGET / 100.0) ** 10, 3),
        }
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Steps 3-5 — Per-country simulation
# ═══════════════════════════════════════════════════════════════════════════
_CLS = pd.read_csv(cfg.HS4_CLASS_FILE, dtype=str).set_index("code_trade")


def add_names(df, code_col="Product"):
    """Attach product name, HS2 chapter (type) and HS section (group)."""
    df = df.drop(columns=["Name", "Type_HS2", "Group_Section"],
                 errors="ignore").copy()
    codes = df[code_col].astype(str)
    df.insert(1, "Name", codes.map(_CLS["product_name"]).fillna(""))
    df.insert(2, "Type_HS2", codes.map(_CLS["hs2_name"]).fillna(""))
    df.insert(3, "Group_Section", codes.map(_CLS["section"]).fillna(""))
    return df


def max_feasible_eci_z(state, c):
    """
    Highest average future PCI any admissible portfolio can reach:
    predicted-2032 specializations are forced IN by the LP, currently
    specialized products predicted to exit are forced OUT, and the best
    strategy is to add the highest-PCI entry candidates one by one.
    Returned in z units of the future nn-ECI distribution.
    """
    i = state["countries"].index(c)
    forced = state["M_2032"][i, :] > 0
    banned = (state["RCA_2022"][i, :] > 1) & (~forced)
    cand = np.sort(state["PCI_future"][~forced & ~banned])[::-1]
    pci_f = state["PCI_future"][forced]
    best = pci_f.mean()
    s, n = pci_f.sum(), pci_f.size
    for v in cand:
        s, n = s + v, n + 1
        best = max(best, s / n)
    return (best - state["mean_nn"]) / state["sd_nn"]


def simulate_country(c, state, m_base, m_ext, gdpz_2022, last_year, scen):
    iso3, name = cfg.TARGET_COUNTRIES[c], cfg.COUNTRY_NAMES[c]
    be, bx = state["beta_entry"], state["beta_exit"]
    print(f"\n{'=' * 66}\n{name} ({iso3})\n{'=' * 66}")

    # ── Current & predicted complexity ────────────────────────────────────
    eci_2032_z = fm.eci_z(state, c)
    gdpz = gdpz_2022.get(iso3, 0.0)
    g_pred = gm.predict_growth(m_base, eci_2032_z, gdpz, last_year)
    print(f"  Predicted 2032 ECI (no optimization): {eci_2032_z:+.3f}   "
          f"[paper: {cfg.PAPER_ECI_2032[c]:+.3f}]")
    print(f"  Expected growth at that ECI:          {g_pred:.2f} %  "
          f"[paper: {cfg.PAPER_GROWTH_EXP[c]:.2f} %]")

    # ── ECI targets from the two growth models ────────────────────────────
    co2_plug = scen[c]["co2_reduction"]
    unred_plug = scen[c]["unemp_reduction"]
    t_base = gm.invert_for_eci(m_base, cfg.TARGET_GROWTH, gdpz, last_year)
    t_ext_raw = gm.invert_for_eci(m_ext, cfg.TARGET_GROWTH, gdpz, last_year,
                                  co2red=co2_plug, unred=unred_plug)
    print(f"  ECI target, baseline Eq. 3:  {t_base:+.3f}  "
          f"[paper: {cfg.PAPER_ECI_TARGET[c]:+.3f}]")
    print(f"  ECI target, extended Eq. 3:  {t_ext_raw:+.3f}  "
          f"(CO2 reduction +{co2_plug:.1f}%/yr; unemployment reduction "
          f"+{unred_plug:.1f}%/yr: {scen[c]['unemp_now']:.2f}% -> "
          f"{scen[c]['unemp_in_10y']:.2f}% in 10y)")

    # Under a decarbonization scenario the required ECI can exceed what ANY
    # portfolio can deliver (growth was historically carbon-coupled). In
    # that case we cap the target at the country's maximum feasible average
    # future PCI and report the growth attainable under the scenario.
    z_max = max_feasible_eci_z(state, c)
    t_ext = min(t_ext_raw, z_max - 0.02)
    g_attain = gm.predict_growth(m_ext, t_ext, gdpz, last_year,
                                 co2red=co2_plug, unred=unred_plug)
    capped = t_ext < t_ext_raw
    if capped:
        print(f"  !! target unattainable (max feasible ECI = {z_max:+.3f});"
              f" capped at {t_ext:+.3f} -> attainable growth under the"
              f" scenario: {g_attain:.2f} % (vs {cfg.TARGET_GROWTH} % asked)")
    scen[c].update({"eci_target_raw": t_ext_raw, "eci_target_used": t_ext,
                    "max_feasible_eci": z_max, "capped": capped,
                    "attainable_growth": g_attain})

    # ── LP at the final targets ────────────────────────────────────────────
    # "paper_target" = the OLD approach run at the paper's PUBLISHED ECI
    # target (Figure 5). This is the validation run: it removes the only
    # source of divergence from the paper (our GDP series is constant-2015
    # USD, the paper's is PPP constant-2021, which shifts the inverted
    # target). "baseline" and "extended" use internally consistent targets
    # estimated from the same local data.
    res = {}
    for label, t_z, model, extra in (
            ("paper_target", cfg.PAPER_ECI_TARGET[c], m_base, {}),
            ("baseline", t_base, m_base, {}),
            ("extended", t_ext, m_ext,
             dict(co2red=co2_plug, unred=unred_plug))):
        target_nn = state["mean_nn"] + state["sd_nn"] * t_z
        df = opt.run_lp(state, c, target_nn, be, bx)
        add_names(df, code_col="Code").to_csv(
            os.path.join(cfg.RESULTS_DIR, f"lp_full_{c}_{label}.csv"),
            index=False)

        def g_of(z, model=model, extra=extra):
            return gm.predict_growth(model, z, gdpz, last_year, **extra)

        tiers, _ = opt.sweep_targets(state, c, t_z, be, bx,
                                     growth_of_target=g_of,
                                     step=cfg.SWEEP_STEP)
        tiers = add_names(tiers)
        tiers.to_csv(os.path.join(
            cfg.RESULTS_DIR, f"products_{c}_{label}.csv"), index=False)
        res[label] = {"target_z": t_z, "final": df, "tiers": tiers,
                      "suggested": set(df.loc[df["Added_vol"] > 0, "Code"])}
        print(f"  [{label:>8}] target ECI {t_z:+.3f} -> "
              f"{len(res[label]['suggested'])} products, "
              f"added volume USD {df['Added_vol'].sum() / 1e6:,.0f} M")
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Comparison + validation
# ═══════════════════════════════════════════════════════════════════════════
def compare(c, res, state):
    """Product-level comparison of the two portfolios."""
    fb, fe = res["baseline"]["final"], res["extended"]["final"]
    sb, se = res["baseline"]["suggested"], res["extended"]["suggested"]
    codes = sorted(sb | se)
    ref = fe.set_index("Code")
    rows = [{
        "Product": k,
        "Name": _CLS["product_name"].get(str(k), ""),
        "Type_HS2": _CLS["hs2_name"].get(str(k), ""),
        "Group_Section": _CLS["section"].get(str(k), ""),
        "in_baseline": k in sb, "in_extended": k in se,
        "status": ("both" if k in sb and k in se
                   else "baseline only" if k in sb else "extended only"),
        "RCA_2022": round(ref.loc[k, "RCA_start"], 3),
        "PCI_2032": round(ref.loc[k, "PCI_future"], 3),
        "Effort": round(ref.loc[k, "Effort_Ycp"], 3),
        "RelativeRelatedness": round(ref.loc[k, "Relative_relatedness_start"], 3),
        "Added_vol_baseline": fb.set_index("Code").loc[k, "Added_vol"],
        "Added_vol_extended": ref.loc[k, "Added_vol"],
    } for k in codes]
    cmp_df = pd.DataFrame(rows).sort_values(["status", "Effort"])
    cmp_df.to_csv(os.path.join(cfg.RESULTS_DIR, f"comparison_{c}.csv"),
                  index=False)
    return cmp_df


def validate(c, res):
    """Overlap of the old approach (at the paper's published ECI target)
    with the products listed in paper Figure 5c/5d."""
    paper = set(cfg.PAPER_PRODUCTS[c])
    ours = set(map(str, res["paper_target"]["suggested"]))
    inter = sorted(paper & ours)
    return {"country": c, "paper_n": len(paper), "ours_n": len(ours),
            "matched": len(inter), "match_pct": 100 * len(inter) / len(paper),
            "matched_codes": " ".join(inter),
            "missed_codes": " ".join(sorted(paper - ours))}


def scatter_figure(c, res, state):
    """Figure-5 style effort-complexity diagram, baseline vs extended."""
    fb, fe = res["baseline"]["final"], res["extended"]["final"]
    base = fb[(fb["RCA_start"] < 1)]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(base["Effort_Ycp"], base["PCI_future"], s=14, c="0.8",
               alpha=0.45, lw=0, label=None)
    for df, col, mk, lab in ((fb, "#d95f02", "s", "ECI (baseline Eq. 3)"),
                             (fe, "#1f77b4", "o", "ECI (extended Eq. 3)")):
        sel = df[df["Added_vol"] > 0]
        ax.scatter(sel["Effort_Ycp"], sel["PCI_future"], s=48, c=col,
                   marker=mk, edgecolor="k", lw=0.5, alpha=0.75, label=lab)
    cr = state["CountryRankings"]
    nn = cr.loc[cr["Country"] == c, "ECI_not_normalized"].values[0]
    ax.axhline(nn, color="k", ls="--", lw=1.5, alpha=0.4)
    ax.set_xlim(0, np.nanpercentile(base["Effort_Ycp"], 97))
    ax.set_xlabel("Estimated effort (added RCA at steppingstone, $Y_{cp}$)")
    ax.set_ylabel("Estimated PCI in 2032")
    ax.set_title(f"{cfg.COUNTRY_NAMES[c]} — target growth "
                 f"{cfg.TARGET_GROWTH}%:  baseline vs CO2/employment extension")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIG_DIR, f"effort_complexity_{c}.png"), dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
def main():
    state = get_state()
    m_base, m_ext, gdpz_2022, last_year = get_growth_models()
    scen = sustainability_targets()

    validations, all_res = [], {}
    for c in cfg.TARGET_COUNTRIES:
        res = simulate_country(c, state, m_base, m_ext, gdpz_2022,
                               last_year, scen)
        all_res[c] = res
        compare(c, res, state)
        validations.append(validate(c, res))
        scatter_figure(c, res, state)

    scen_df = pd.DataFrame(scen).T
    scen_df.index.name = "country"
    scen_df.to_csv(os.path.join(cfg.RESULTS_DIR, "scenario_targets.csv"))

    vdf = pd.DataFrame(validations)
    vdf.to_csv(os.path.join(cfg.RESULTS_DIR, "validation_vs_paper.csv"),
               index=False)
    print("\n" + "=" * 66)
    print("Validation of the baseline against paper Figure 5:")
    print(vdf.to_string(index=False))
    print("\nAll outputs saved in results/ — done.")


if __name__ == "__main__":
    main()
