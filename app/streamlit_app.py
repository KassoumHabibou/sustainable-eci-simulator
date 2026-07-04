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

  /* ── main navigation (radio styled as full-width tab cards) ─────────── */
  .st-key-main-nav div[role="radiogroup"] {{
      display: flex; flex-direction: row; width: 100%; gap: 10px; }}
  .st-key-main-nav div[role="radiogroup"] > label {{
      flex: 1 1 0; justify-content: center; text-align: center;
      background: {CARD}; border: 1px solid #E5E1D8; border-radius: 12px;
      padding: 12px 0; margin: 0; cursor: pointer;
      box-shadow: 0 1px 3px rgba(41,38,27,.05); }}
  .st-key-main-nav div[role="radiogroup"] > label p {{
      font-size: 1.05rem !important; font-family: {SERIF};
      color: {INK} !important; }}
  .st-key-main-nav div[role="radiogroup"] > label > div:first-child {{
      display: none; }}   /* hide the radio circle */
  .st-key-main-nav div[role="radiogroup"] > label:has(input:checked) {{
      background: {ACCENT}; border-color: {ACCENT}; }}
  .st-key-main-nav div[role="radiogroup"] > label:has(input:checked) p {{
      color: #FFFFFF !important; font-weight: 700; }}

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

  /* tables */
  div[data-testid="stDataFrame"] {{ border:1px solid #E5E1D8;
      border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Georgia, serif", color=INK, size=16),
    margin=dict(l=10, r=10, t=46, b=10),
)


def style_axes(fig):
    """Large, dark, readable axis numbers and titles on every chart."""
    fig.update_xaxes(tickfont=dict(size=14, color=INK),
                     title_font=dict(size=15, color=INK),
                     gridcolor="#E8E3D7", zerolinecolor="#D8D2C4")
    fig.update_yaxes(tickfont=dict(size=14, color=INK),
                     title_font=dict(size=15, color=INK),
                     gridcolor="#E8E3D7", zerolinecolor="#D8D2C4")
    fig.update_layout(title_font=dict(size=17, color=INK),
                      hoverlabel=dict(font_size=14))
    return fig


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

NAV_PROFILE = "🗺️  Country profile"
NAV_RESULTS = "🎯  Simulation results"
NAV_METHOD = "📚  Method & sources"


# ════════════════════════════════════════════════════════════════════════════
# Sidebar — policy levers (runs BEFORE the navigation widget, so a finished
# simulation can jump the user straight to the results section)
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

    lo, hi = engine.RANGES["growth_target"]
    g_target = st.number_input(
        "📈 GDP per-capita growth objective (%/yr)",
        min_value=lo, max_value=hi, value=3.50, step=0.20, format="%.2f",
        help=f"Type any value to two decimals, between {lo} and {hi}. "
             "The +/- buttons move by 0.2.")
    lo, hi = engine.RANGES["co2_growth"]
    co2_g = st.number_input(
        "🌱 CO2 emissions growth (%/yr) — 0 or negative",
        min_value=lo, max_value=hi, value=-2.00, step=0.20, format="%.2f",
        help="Negative = emissions decline. The Paris Agreement requires a "
             "declining path; positive values are not allowed. "
             "Type any value to two decimals (e.g. −1.35).")
    lo, hi = engine.RANGES["unemp_growth"]
    un_g = st.number_input(
        "🤝 Unemployment rate growth (%/yr) — 0 or negative",
        min_value=lo, max_value=hi, value=-2.00, step=0.20, format="%.2f",
        help="Negative = unemployment falls (SDG 8). −2%/yr ≈ one fifth "
             "lower after a decade. Type any value to two decimals.")

    run = st.button("🚀  Run simulation", use_container_width=True)
    if run:
        with st.spinner("Optimizing the export portfolio …"):
            try:
                res = engine.run_policy_simulation(
                    state, models, indicators, sel["code"],
                    g_target, co2_g, un_g)
                st.session_state["result"] = res
                st.session_state["result_country"] = sel["name"]
                st.session_state["sim_error"] = None
                st.session_state["just_finished"] = True
                n_prod = res["summary"]["Number of recommended products"]
                st.toast(f"✅ Simulation complete — {n_prod} products "
                         f"recommended for {sel['name']}.", icon="🎉")
            except Exception as exc:      # pragma: no cover
                st.session_state["sim_error"] = str(exc)
                st.toast("❌ Simulation failed.", icon="⚠️")
        # jump straight to the results section (nav widget not built yet)
        st.session_state["nav"] = NAV_RESULTS

    st.markdown("---")
    st.caption("Engine: ECI optimization (Stojkoski & Hidalgo, Research "
               "Policy 2026) with the CERDI–World Bank sustainability "
               "extension. Data: OEC trade (2022), World Bank WDI.")


# ════════════════════════════════════════════════════════════════════════════
# Header + navigation
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

with st.container(key="main-nav"):
    nav = st.radio("Navigation", [NAV_PROFILE, NAV_RESULTS, NAV_METHOD],
                   key="nav", horizontal=True, label_visibility="collapsed")

# ════════════════════════════════════════════════════════════════════════════
# Section 1 — Country profile
# ════════════════════════════════════════════════════════════════════════════
if nav == NAV_PROFILE:
    c1, c2 = st.columns([1.05, 1])
    with c1:
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
        figt.update_layout(**PLOTLY_LAYOUT, height=215,
                           title=dict(text="Economic Complexity Index", x=0.02),
                           showlegend=False)
        st.plotly_chart(style_axes(figt), use_container_width=True)

        figu = go.Figure()
        figu.add_scatter(x=ts["unemp"]["Year"],
                         y=ts["unemp"]["Unemployment_Rate_ILO"],
                         mode="lines", name="Unemployment",
                         line=dict(color="#5B7C99", width=3))
        figu.update_layout(**PLOTLY_LAYOUT, height=215,
                           title=dict(text="Unemployment rate (%)", x=0.02),
                           showlegend=False)
        st.plotly_chart(style_axes(figu), use_container_width=True)

        figc = go.Figure()
        figc.add_scatter(x=ts["co2"]["Year"], y=ts["co2"]["CO2_Emissions_Mt"],
                         mode="lines", name="CO2",
                         line=dict(color="#7A6A53", width=3))
        figc.update_layout(**PLOTLY_LAYOUT, height=215,
                           title=dict(text="CO2 emissions (Mt)", x=0.02),
                           showlegend=False)
        st.plotly_chart(style_axes(figc), use_container_width=True)

    ts = engine.timeseries(indicators, sel["iso3"])
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
        st.plotly_chart(style_axes(figs), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Simulation results
# ════════════════════════════════════════════════════════════════════════════
elif nav == NAV_RESULTS:
    err = st.session_state.get("sim_error")
    res = st.session_state.get("result")
    if err:
        st.error(f"Simulation failed: {err}")
    if res is None and not err:
        st.info("Choose a country and your objectives in the sidebar, then "
                "press **Run simulation**.")
    elif res is not None:
        s = res["summary"]
        if st.session_state.pop("just_finished", False):
            n_prod = s["Number of recommended products"]
            st.success(f"**Simulation complete.** The optimal diversification "
                       f"portfolio for **{st.session_state['result_country']}**"
                       f" is ready below — {n_prod} products identified. "
                       f"You can export everything to Excel at the bottom "
                       f"of the page.")
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
            figg = px.treemap(res["products"], path=["Group", "Name"],
                              values="Added exports (USD M)",
                              color="Effort",
                              color_continuous_scale=["#9BB068", "#E8CFC4", ACCENT])
            figg.update_traces(textfont=dict(size=15, color="#29261B"),
                               insidetextfont=dict(size=15))
            figg.update_layout(**PLOTLY_LAYOUT, height=420,
                               title=dict(text="Where the investment goes", x=0.02),
                               coloraxis_colorbar=dict(tickfont=dict(size=13)))
            st.plotly_chart(figg, use_container_width=True)
        with cB:
            pr = res["products"]
            fige = px.scatter(pr, x="Effort", y="Estimated PCI 2032",
                              size="Added exports (USD M)", color="Group",
                              hover_name="Name", size_max=34)
            fige.update_layout(**PLOTLY_LAYOUT, height=440,
                               title=dict(text="Effort vs complexity of the "
                                               "portfolio", x=0.02),
                               legend=dict(font=dict(size=12)))
            st.plotly_chart(style_axes(fige), use_container_width=True)

        xlsx = engine.build_excel(res, models)
        st.download_button(
            "⬇  Export products & summary (Excel, with metadata and sources)",
            data=xlsx,
            file_name=f"diversification_{s['ISO3']}_{s['Growth objective (%/yr)']}pct.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Method & sources
# ════════════════════════════════════════════════════════════════════════════
else:
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
