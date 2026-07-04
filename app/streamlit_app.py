"""
Sustainable Diversification Simulator — Streamlit app
=====================================================
Run from the project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

# ════════════════════════════════════════════════════════════════════════════
# Page config & design (Claude-inspired palette: warm ivory, terracotta)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Sustainable Diversification Simulator",
                   page_icon="🧭", layout="wide")

ACCENT = "#C15F3C"      # terracotta
ACCENT_SOFT = "#E8CFC4"
INK = "#29261B"
PAPER = "#FAF9F5"
CARD = "#FFFFFF"
GOOD = "#2E7D5B"
WARN = "#B3261E"
SERIF = "Georgia, 'Times New Roman', serif"

st.markdown(f"""
<style>
  /* ── base: force dark ink on the warm paper, everywhere ─────────────── */
  .stApp {{ background-color: {PAPER}; }}
  .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
  .stMarkdown, .stCaption, div[data-testid="stCaptionContainer"] p {{
      color: {INK} !important; }}
  h1, h2, h3, h4 {{ font-family: {SERIF}; color: {INK} !important; }}
  .block-container {{ padding-top: 1.4rem; max-width: 1250px; }}

  /* sidebar */
  section[data-testid="stSidebar"] {{
      background-color: #F0EEE6; border-right: 1px solid #E2DCCC; }}
  section[data-testid="stSidebar"] * {{ color: {INK}; }}
  section[data-testid="stSidebar"] .stCaption p {{ color:#6E6A5E !important; }}

  /* metric cards */
  div[data-testid="stMetric"] {{
      background: {CARD}; border: 1px solid #E5E1D8; border-radius: 14px;
      padding: 14px 18px; box-shadow: 0 1px 4px rgba(41,38,27,.06); }}
  div[data-testid="stMetricLabel"] p {{ font-size:.82rem; color:#6E6A5E !important; }}
  div[data-testid="stMetricValue"] {{ color:{INK} !important;
      font-family:{SERIF}; }}

  /* ── tabs: full width, big, readable ────────────────────────────────── */
  div[data-testid="stTabs"] div[role="tablist"] {{
      display: flex; width: 100%; gap: 10px;
      border-bottom: none; margin-bottom: 6px; }}
  div[data-testid="stTabs"] button[role="tab"] {{
      flex: 1 1 0; justify-content: center;
      background: {CARD}; border: 1px solid #E5E1D8; border-radius: 12px;
      padding: 12px 0; font-size: 1.05rem; font-family: {SERIF};
      color: {INK} !important; box-shadow: 0 1px 3px rgba(41,38,27,.05); }}
  div[data-testid="stTabs"] button[role="tab"] p {{
      font-size: 1.05rem !important; color: {INK} !important; }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
      background: {ACCENT} !important; border-color: {ACCENT}; }}
  div[data-testid="stTabs"] button[aria-selected="true"] p {{
      color: #FFFFFF !important; font-weight: 700; }}
  div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
  div[data-testid="stTabs"] div[data-baseweb="tab-border"] {{ display:none; }}

  /* hero banner */
  .hero {{
      background: linear-gradient(135deg, #F3E8E1 0%, {PAPER} 70%);
      border: 1px solid #E8E2D5; border-radius: 18px;
      padding: 26px 32px; margin-bottom: 14px; }}
  .hero h1 {{ margin: 0 0 4px 0; font-size: 2.1rem; color:{INK}; }}
  .hero p  {{ margin: 0; color: #6E6A5E !important; }}
  .pill {{ display:inline-block; padding: 4px 14px; border-radius: 999px;
      background: {ACCENT_SOFT}; color:{INK} !important;
      font-size:.85rem; margin: 4px 6px 0 0; font-weight:600; }}

  /* callouts */
  .warnbox {{ background:#FBEAE8; border-left:5px solid {WARN};
      border-radius:10px; padding:14px 18px; margin:10px 0; color:{INK}; }}
  .okbox {{ background:#E9F3EE; border-left:5px solid {GOOD};
      border-radius:10px; padding:14px 18px; margin:10px 0; color:{INK}; }}
  .warnbox b, .okbox b {{ color:{INK}; }}

  /* buttons */
  .stButton>button {{ background:{ACCENT}; color:#FFFFFF !important;
      border:none; border-radius:10px; padding:.6rem 1.4rem;
      font-weight:700; font-size:1rem; }}
  .stButton>button:hover {{ background:#A94F30; }}
  .stButton>button * {{ color:#FFFFFF !important; }}
  .stDownloadButton>button {{ background:{INK}; color:#FFFFFF !important;
      border:none; border-radius:10px; font-weight:600; }}
  .stDownloadButton>button * {{ color:#FFFFFF !important; }}

  /* inputs */
  div[data-baseweb="select"] * {{ color:{INK}; }}
  div[data-testid="stSlider"] label p {{ font-weight:600; }}

  /* tables */
  div[data-testid="stDataFrame"] {{ border:1px solid #E5E1D8;
      border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Georgia, serif", color=INK, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ════════════════════════════════════════════════════════════════════════════
# Cached resources
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading the forward model …")
def _state():
    return engine.load_state()


@st.cache_resource(show_spinner="Estimating growth models …")
def _models():
    return engine.load_models()


@st.cache_data(show_spinner=False)
def _indicators():
    return engine.load_indicators()


state, models, indicators = _state(), _models(), _indicators()
countries = engine.available_countries(state, models)
by_name = {c["name"]: c for c in countries}


# ════════════════════════════════════════════════════════════════════════════
# Sidebar — policy levers
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ Policy levers")
    st.markdown("Set your objectives, then run the simulation.")

    labels = {f"{engine.country_flag(c['code'])}  {c['name']}": c
              for c in countries}
    keys = list(labels.keys())
    default_ix = next((i for i, k in enumerate(keys)
                       if k.endswith("Thailand")), 0)
    sel_label = st.selectbox("Country", keys, index=default_ix)
    sel = labels[sel_label]

    g_target = st.slider("📈 GDP per-capita growth objective (%/yr)",
                         *engine.RANGES["growth_target"], value=3.5, step=0.1)
    co2_g = st.slider("🌱 CO2 emissions growth (%/yr) — 0 or negative",
                      *engine.RANGES["co2_growth"], value=-2.0, step=0.5,
                      help="Negative = emissions decline. The Paris Agreement "
                           "requires a declining path; positive values are not allowed.")
    un_g = st.slider("🤝 Unemployment rate growth (%/yr) — 0 or negative",
                     *engine.RANGES["unemp_growth"], value=-2.0, step=0.5,
                     help="Negative = unemployment falls (SDG 8). "
                          "−2%/yr ≈ one fifth lower after a decade.")

    run = st.button("🚀  Run simulation", use_container_width=True)
    st.markdown("---")
    st.caption("Engine: ECI optimization (Stojkoski & Hidalgo, Research "
               "Policy 2026) with the CERDI–World Bank sustainability "
               "extension. Data: OEC trade (2022), World Bank WDI.")


# ════════════════════════════════════════════════════════════════════════════
# Header + country dashboard
# ════════════════════════════════════════════════════════════════════════════
flag = engine.country_flag(sel["code"])
ind = engine.latest_indicators(indicators, sel["iso3"])

pills = []
if ind["eci"] is not None:
    pills.append(f"ECI {ind['eci']:.2f} ({ind['eci_year']})")
if ind["gdp_pc"] is not None:
    pills.append(f"GDP pc ${ind['gdp_pc']:,.0f}")
if ind["unemp"] is not None:
    pills.append(f"Unemployment {ind['unemp']:.1f}% ({ind['unemp_year']})")
if ind["co2"] is not None:
    pills.append(f"CO2 {ind['co2']:,.0f} Mt ({ind['co2_year']})")
pills_html = "".join(f'<span class="pill">{p}</span>' for p in pills)

st.markdown(f"""
<div class="hero">
  <h1>{flag}&nbsp; {sel['name']}</h1>
  <p>Sustainable Diversification Simulator — growth targets under climate
     and employment commitments</p>
  <div style="margin-top:10px">{pills_html}</div>
</div>
""", unsafe_allow_html=True)

tab_profile, tab_sim, tab_method = st.tabs(
    ["🗺️  Country profile", "🎯  Simulation results", "📚  Method & sources"])

# ── Country profile ─────────────────────────────────────────────────────────
with tab_profile:
    c1, c2 = st.columns([1.05, 1])
    with c1:
        # dynamic map, zoomed on the country
        map_df = pd.DataFrame({"iso3": [sel["iso3"]], "v": [1.0],
                               "name": [sel["name"]]})
        figm = px.choropleth(map_df, locations="iso3", color="v",
                             hover_name="name",
                             color_continuous_scale=[[0, ACCENT], [1, ACCENT]])
        figm.update_traces(marker_line_color="white", marker_line_width=0.6,
                           showscale=False,
                           hovertemplate="<b>%{hovertext}</b><extra></extra>")
        figm.update_geos(fitbounds="locations", visible=True,
                         showcountries=True, countrycolor="#D8D2C4",
                         showland=True, landcolor="#EFEBE0",
                         showocean=True, oceancolor="#DCE8EC",
                         projection_type="natural earth")
        figm.update_layout(**PLOTLY_LAYOUT, height=380,
                           coloraxis_showscale=False,
                           title=dict(text="Where we are", x=0.02))
        st.plotly_chart(figm, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Population", f"{ind['population']/1e6:,.1f} M"
                  if ind["population"] else "n/a")
        k2.metric("GDP (total)", f"${ind['gdp_total']/1e9:,.0f} B"
                  if ind["gdp_total"] else "n/a")
        k3.metric("CO2 per capita", f"{ind['co2_pc']:.2f} t"
                  if ind["co2_pc"] else "n/a")

    with c2:
        ts = engine.timeseries(indicators, sel["iso3"])
        figt = go.Figure()
        figt.add_scatter(x=ts["eci"]["Year"], y=ts["eci"]["ECI"],
                         mode="lines", name="ECI",
                         line=dict(color=ACCENT, width=3))
        figt.update_layout(**PLOTLY_LAYOUT, height=200,
                           title=dict(text="Economic Complexity Index", x=0.02),
                           showlegend=False)
        st.plotly_chart(figt, use_container_width=True)

        figu = go.Figure()
        figu.add_scatter(x=ts["unemp"]["Year"],
                         y=ts["unemp"]["Unemployment_Rate_ILO"],
                         mode="lines", name="Unemployment",
                         line=dict(color="#5B7C99", width=3))
        figu.update_layout(**PLOTLY_LAYOUT, height=200,
                           title=dict(text="Unemployment rate (%)", x=0.02),
                           showlegend=False)
        st.plotly_chart(figu, use_container_width=True)

        figc = go.Figure()
        figc.add_scatter(x=ts["co2"]["Year"], y=ts["co2"]["CO2_Emissions_Mt"],
                         mode="lines", name="CO2",
                         line=dict(color="#7A6A53", width=3))
        figc.update_layout(**PLOTLY_LAYOUT, height=200,
                           title=dict(text="CO2 emissions (Mt)", x=0.02),
                           showlegend=False)
        st.plotly_chart(figc, use_container_width=True)

    if len(ts["sectors"]):
        st.markdown("##### Employment structure")
        sec = ts["sectors"].rename(columns={
            "Employment_Agriculture_Pct": "Agriculture",
            "Employment_Industry_Pct": "Industry",
            "Employment_Services_Pct": "Services"})
        figs = px.area(sec, x="Year", y=["Agriculture", "Industry", "Services"],
                       color_discrete_sequence=["#9BB068", ACCENT, "#5B7C99"])
        figs.update_layout(**PLOTLY_LAYOUT, height=260,
                           legend=dict(orientation="h", y=1.12, title=None),
                           yaxis_title="% of employment", xaxis_title=None)
        st.plotly_chart(figs, use_container_width=True)

# ── Simulation ──────────────────────────────────────────────────────────────
with tab_sim:
    if run:
        with st.spinner("Optimizing the export portfolio …"):
            try:
                res = engine.run_policy_simulation(
                    state, models, indicators, sel["code"],
                    g_target, co2_g, un_g)
                st.session_state["result"] = res
                st.session_state["result_country"] = sel["name"]
            except Exception as exc:      # pragma: no cover
                st.error(f"Simulation failed: {exc}")

    res = st.session_state.get("result")
    if res is None:
        st.info("Choose a country and your objectives in the sidebar, then "
                "press **Run simulation**.")
    else:
        s = res["summary"]
        st.markdown(f"### Results — {st.session_state['result_country']}")

        if res["capped"]:
            st.markdown(f"""<div class="warnbox"><b>Objective not fully
            attainable.</b> Reaching {s['Growth objective (%/yr)']}% growth under
            this scenario would require an ECI of {s['ECI required for the objective']},
            beyond the country's maximum feasible complexity
            ({s['Maximum feasible ECI']}). The simulation therefore runs at the
            feasibility frontier: the portfolio below is the most ambitious
            available, and delivers an estimated
            <b>{s['Attainable growth under the scenario (%/yr)']}% per year</b>.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="okbox"><b>Objective attainable.</b>
            The portfolio below reaches the complexity required for
            {s['Growth objective (%/yr)']}% annual growth under your scenario.
            </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Attainable growth",
                  f"{s['Attainable growth under the scenario (%/yr)']} %/yr",
                  delta=f"{s['Gain from diversification (pp/yr)']} pp vs no action")
        m2.metric("Products to develop", s["Number of recommended products"])
        m3.metric("Required investment",
                  f"${s['Required investment (USD bn added exports)']} B")
        m4.metric("… as share of GDP",
                  f"{s['Investment as share of GDP (%)']} %")
        m5.metric("ECI target", s["ECI target used"],
                  delta=f"max feasible {s['Maximum feasible ECI']}",
                  delta_color="off")

        st.markdown("#### Recommended products")
        st.caption("Ordered by priority (lowest effort first). ‘Share of "
                   "GDP’ is the added export volume required to gain "
                   "comparative advantage, relative to latest GDP.")
        st.dataframe(
            res["products"].style.format({
                "RCA 2022": "{:.2f}", "Estimated PCI 2032": "{:.2f}",
                "Effort": "{:.2f}", "Added exports (USD M)": "{:,.1f}",
                "Share of GDP (%)": "{:.3f}"}),
            use_container_width=True, height=420)

        cA, cB = st.columns([1, 1])
        with cA:
            top = res["products"].groupby("Group")["Added exports (USD M)"] \
                .sum().sort_values(ascending=False).reset_index()
            figg = px.treemap(res["products"], path=["Group", "Name"],
                              values="Added exports (USD M)",
                              color="Effort",
                              color_continuous_scale=["#9BB068", "#E8CFC4", ACCENT])
            figg.update_layout(**PLOTLY_LAYOUT, height=420,
                               title=dict(text="Where the investment goes", x=0.02))
            st.plotly_chart(figg, use_container_width=True)
        with cB:
            pr = res["products"]
            fige = px.scatter(pr, x="Effort", y="Estimated PCI 2032",
                              size="Added exports (USD M)", color="Group",
                              hover_name="Name", size_max=34)
            fige.update_layout(**PLOTLY_LAYOUT, height=420,
                               title=dict(text="Effort vs complexity of the "
                                               "portfolio", x=0.02),
                               legend=dict(font=dict(size=10)))
            st.plotly_chart(fige, use_container_width=True)

        xlsx = engine.build_excel(res, models)
        st.download_button(
            "⬇  Export products & summary (Excel, with metadata and sources)",
            data=xlsx,
            file_name=f"diversification_{s['ISO3']}_{s['Growth objective (%/yr)']}pct.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

# ── Method & sources ────────────────────────────────────────────────────────
with tab_method:
    st.markdown("""
### How the simulator works

**1. The growth anchor.** A panel regression on ten-year windows
(1999–2009, 2009–2019) relates GDP-per-capita growth to economic
complexity (ECI), initial income, the pace of CO2-emission reduction and
the pace of unemployment reduction, with interactions. Both sustainability
margins are significant: emission reduction has a growth price
(≈0.29 pp per point of pace), unemployment reduction a growth payoff
(≈0.10 pp, rising with complexity).

**2. Your objectives become a complexity target.** The equation is linear
in ECI, so it can be inverted: given your growth objective and your CO2
and unemployment paths, it returns the ECI the country needs. If that
exceeds the country's *feasibility frontier* (the highest average
complexity any attainable export basket can deliver), the tool caps the
target and reports the growth actually attainable.

**3. The target becomes products.** A 0–1 integer program (the ECI
optimization of Stojkoski & Hidalgo, *Research Policy* 2026) selects the
new export products that reach the target at minimal total *effort* — the
increase in comparative advantage needed by 2027 for the model to predict
competitive exports in 2032.

**Read the results with care.** The anchor is correlational; inverting it
under counterfactual scenarios is an extrapolation, not a causal claim.
The tool is a disciplined way to confront growth ambitions with climate
and employment commitments — not a substitute for country expertise.
""")
    st.markdown("#### Data sources")
    st.table(pd.DataFrame(engine.SOURCES, columns=["Source", "Details"]))
    st.caption("Replication package: `extension_simulation/` — "
               "`python run_simulation.py` reproduces the paper; this app "
               "reuses the same engine (`app/engine.py`).")
