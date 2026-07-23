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
BLUE = "#5B7C99"
GREEN = "#6B8F71"
BROWN = "#7A6A53"
SERIF = "Georgia, 'Times New Roman', serif"

st.markdown(f"""
<style>
  /* ── base: fill the screen, dark ink on warm paper ──────────────────── */
  .stApp {{ background-color: {PAPER}; }}
  .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
  .stMarkdown, .stCaption, div[data-testid="stCaptionContainer"] p {{
      color: {INK} !important; }}
  h1, h2, h3, h4 {{ font-family: {SERIF}; color: {INK} !important; }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem;
      padding-left: 2.2rem; padding-right: 2.2rem; max-width: 100%; }}

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

  /* section headers as bands, easy to spot while scrolling */
  .band {{ background: {ACCENT_SOFT}; border-radius: 12px;
      padding: 10px 20px; margin: 26px 0 14px 0;
      border-left: 6px solid {ACCENT}; }}
  .band h3 {{ margin: 0; font-size: 1.35rem; }}

  /* hero banner */
  .hero {{
      background: linear-gradient(135deg, #F3E8E1 0%, {PAPER} 70%);
      border: 1px solid #E8E2D5; border-radius: 18px;
      padding: 20px 28px; margin-bottom: 12px; }}
  .hero h1 {{ margin: 0 0 4px 0; font-size: 2.0rem; color:{INK}; }}
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
    margin=dict(l=10, r=10, t=48, b=10),
)


def band(title):
    st.markdown(f'<div class="band"><h3>{title}</h3></div>',
                unsafe_allow_html=True)


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


def line_chart(df, x, y, title, color, height=260):
    fig = go.Figure()
    fig.add_scatter(x=df[x], y=df[y], mode="lines",
                    line=dict(color=color, width=3),
                    fill="tozeroy", fillcolor="rgba(0,0,0,0.03)")
    fig.update_layout(**PLOTLY_LAYOUT, height=height,
                      title=dict(text=title, x=0.02), showlegend=False)
    return style_axes(fig)


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


@st.cache_resource(show_spinner="Loading world trade panels …")
def _trade():
    return engine.load_trade()


def world_share_map(shares, title, height=620, highlight=None):
    """Large choropleth of world-trade shares (%) with a neutral,
    eye-friendly palette. `highlight` (ISO3) outlines the country."""
    fig = px.choropleth(
        shares, locations="iso3", color="share_pct", hover_name="name",
        color_continuous_scale="Blues", range_color=(0, None),
        labels={"share_pct": "% of world"})
    fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=0.5,
                      hovertemplate="<b>%{hovertext}</b><br>"
                                    "%{z:.2f} % of world<extra></extra>")
    if highlight is not None and highlight in set(shares["iso3"]):
        hl = shares[shares["iso3"] == highlight]
        fig.add_trace(go.Choropleth(
            locations=hl["iso3"], z=hl["share_pct"], showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color=ACCENT, marker_line_width=3,
            hoverinfo="skip"))
    fig.update_geos(showcountries=True, countrycolor="#CFC8B8",
                    showland=True, landcolor="#F2EFE7",
                    showocean=True, oceancolor="#E9EEF0",
                    showframe=False, projection_type="natural earth",
                    lataxis_range=[-58, 85])
    fig.update_layout(**PLOTLY_LAYOUT, height=height,
                      title=dict(text=title, x=0.02,
                                 font=dict(size=19, color=INK)),
                      hoverlabel=dict(font_size=15),
                      coloraxis_colorbar=dict(
                          title=dict(text="% of world",
                                     font=dict(size=15)),
                          tickfont=dict(size=14),
                          thickness=20, len=0.8, x=1.0))
    return fig


state, models, indicators = _state(), _models(), _indicators()
trade = _trade()
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
               "extension. Data: OEC trade (1998-2022), World Bank WDI.")


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
# Section 1 — Country profile: one scrollable page
#   Overview → Trends → Trade structure (exports, then imports)
# ════════════════════════════════════════════════════════════════════════════
if nav == NAV_PROFILE:

    # ── Overview ──────────────────────────────────────────────────────────
    band(f"📌 Overview — {sel['name']}")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        map_df = pd.DataFrame({"iso3": [sel["iso3"]], "v": [1.0],
                               "name": [sel["name"]]})
        figm = px.choropleth(map_df, locations="iso3", color="v",
                             hover_name="name",
                             color_continuous_scale=[[0, ACCENT],
                                                     [1, ACCENT]])
        figm.update_traces(marker_line_color="white",
                           marker_line_width=0.6, showscale=False,
                           hovertemplate="<b>%{hovertext}</b><extra></extra>")
        figm.update_geos(fitbounds="locations", visible=True,
                         showcountries=True, countrycolor="#D8D2C4",
                         showland=True, landcolor="#EFEBE0",
                         showocean=True, oceancolor="#DCE8EC",
                         showframe=False, projection_type="natural earth")
        figm.update_layout(**PLOTLY_LAYOUT, height=460,
                           coloraxis_showscale=False,
                           title=dict(text="Where we are", x=0.02))
        st.plotly_chart(figm, use_container_width=True)
    with c2:
        st.markdown("##### Key indicators (latest)")
        k1, k2 = st.columns(2)
        k1.metric("Population", f"{ind['population']/1e6:,.1f} M"
                  if ind["population"] else "n/a")
        k2.metric("GDP (total)", f"${ind['gdp_total']/1e9:,.0f} B"
                  if ind["gdp_total"] else "n/a")
        k3, k4 = st.columns(2)
        k3.metric("GDP per capita", f"${ind['gdp_pc']:,.0f}"
                  if ind["gdp_pc"] else "n/a")
        k4.metric("Economic Complexity", f"{ind['eci']:.2f}"
                  if ind["eci"] is not None else "n/a")
        k5, k6 = st.columns(2)
        k5.metric("Unemployment", f"{ind['unemp']:.1f} %"
                  if ind["unemp"] is not None else "n/a")
        k6.metric("CO2 per capita", f"{ind['co2_pc']:.2f} t"
                  if ind["co2_pc"] else "n/a")
        st.caption("Sources: World Bank WDI (latest available year), OEC. "
                   "Scroll down for trends and the full trade structure.")

    # ── Trends ────────────────────────────────────────────────────────────
    band("📈 Trends")
    ts = engine.timeseries(indicators, sel["iso3"])
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(line_chart(ts["eci"], "Year", "ECI",
                                   "Economic Complexity Index", ACCENT),
                        use_container_width=True)
    with g2:
        st.plotly_chart(line_chart(ts["gdp_pc"], "Year", "GDP_pc",
                                   "GDP per capita (constant USD)", GREEN),
                        use_container_width=True)
    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(line_chart(ts["unemp"], "Year",
                                   "Unemployment_Rate_ILO",
                                   "Unemployment rate (%)", BLUE),
                        use_container_width=True)
    with g4:
        st.plotly_chart(line_chart(ts["co2"], "Year", "CO2_Emissions_Mt",
                                   "CO2 emissions (Mt)", BROWN),
                        use_container_width=True)
    if len(ts["sectors"]):
        sec = ts["sectors"].rename(columns={
            "Employment_Agriculture_Pct": "Agriculture",
            "Employment_Industry_Pct": "Industry",
            "Employment_Services_Pct": "Services"})
        figs = px.area(sec, x="Year",
                       y=["Agriculture", "Industry", "Services"],
                       color_discrete_sequence=["#9BB068", ACCENT, BLUE])
        figs.update_layout(**PLOTLY_LAYOUT, height=280,
                           title=dict(text="Employment structure "
                                           "(% of employment)", x=0.02),
                           legend=dict(orientation="h", y=1.15, title=None),
                           yaxis_title=None, xaxis_title=None)
        st.plotly_chart(style_axes(figs), use_container_width=True)

    # ── Trade structure ───────────────────────────────────────────────────
    for flow, icon, verb in (("exports", "📤", "exported"),
                             ("imports", "📥", "imported")):
        band(f"{icon} {flow.capitalize()} of {sel['name']}")
        h1, h2, h3 = st.columns([1, 1, 2])
        with h1:
            y = st.selectbox("Year", trade["years"],
                             index=len(trade["years"]) - 1,
                             key=f"year_{flow}")
        tbl, total = engine.country_trade_table(
            trade, indicators["cls"], flow, sel["code"], y)
        h2.metric(f"Total {flow} ({y})", f"${total/1e9:,.1f} B")
        h3.metric("Products " + verb, f"{len(tbl):,}")

        cL, cR = st.columns([1, 1.6])
        with cL:
            st.markdown(f"**Products {verb} in {y}** — ordered by value")
            st.dataframe(
                tbl.style.format({
                    "Value (USD M)": "{:,.1f}",
                    f"Share of country {flow} (%)": "{:.2f}"}),
                use_container_width=True, height=620)
        with cR:
            if len(tbl):
                opts = [f"{r.Code} — {r.Product}"
                        for r in tbl.head(400).itertuples()]
                pick = st.selectbox(
                    "Product to map", opts, key=f"prod_{flow}",
                    help="Default: the country's top product. The map shows "
                         "each country's share of world trade in it.")
                hs = pick.split(" — ")[0]
                shares, world = engine.product_world_shares(
                    trade, flow, hs, y)
                own = shares.loc[shares["iso3"] == sel["iso3"], "share_pct"]
                own_share = float(own.iloc[0]) if len(own) else 0.0
                st.plotly_chart(world_share_map(
                    shares, f"World {flow} shares — {pick[:46]} ({y})",
                    height=560, highlight=sel["iso3"]),
                    use_container_width=True)
                st.caption(f"World {flow} of this product in {y}: "
                           f"**${world/1e9:,.2f} B** — {sel['name']}'s "
                           f"share: **{own_share:.2f}%** (outlined in "
                           f"terracotta on the map).")

# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Simulation results: one scrollable page
#   Verdict → Product list → Portfolio charts → World markets
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
                       f" is ready — {n_prod} products identified. Scroll "
                       f"down for the list, the charts, and the world "
                       f"market of every product.")

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

        # ── Product list ──────────────────────────────────────────────────
        band("📋 Recommended products")
        st.caption("Ordered by priority (lowest effort first). ‘Share of "
                   "GDP’ is the added export volume required to gain "
                   "comparative advantage, relative to latest GDP.")
        st.dataframe(
            res["products"].style.format({
                "RCA 2022": "{:.2f}", "Estimated PCI 2032": "{:.2f}",
                "Effort": "{:.2f}", "Added exports (USD M)": "{:,.1f}",
                "Share of GDP (%)": "{:.3f}"}),
            use_container_width=True, height=480)
        xlsx = engine.build_excel(res, models)
        st.download_button(
            "⬇  Export products & summary (Excel, with metadata and sources)",
            data=xlsx,
            file_name=(f"diversification_{s['ISO3']}_"
                       f"{s['Growth objective (%/yr)']}pct.xlsx"),
            mime=("application/vnd.openxmlformats-officedocument"
                  ".spreadsheetml.sheet"),
            use_container_width=True)

        # ── Portfolio charts ──────────────────────────────────────────────
        band("🧭 Portfolio charts")
        cA, cB = st.columns(2)
        with cA:
            figg = px.treemap(res["products"], path=["Group", "Name"],
                              values="Added exports (USD M)",
                              color="Effort",
                              color_continuous_scale=["#9BB068",
                                                      "#E8CFC4", ACCENT])
            figg.update_traces(textfont=dict(size=15, color="#29261B"),
                               insidetextfont=dict(size=15))
            figg.update_layout(**PLOTLY_LAYOUT, height=540,
                               title=dict(text="Where the investment goes",
                                          x=0.02),
                               coloraxis_colorbar=dict(
                                   tickfont=dict(size=13)))
            st.plotly_chart(figg, use_container_width=True)
        with cB:
            pr = res["products"]
            fige = px.scatter(pr, x="Effort", y="Estimated PCI 2032",
                              size="Added exports (USD M)", color="Group",
                              hover_name="Name", size_max=34)
            fige.update_layout(**PLOTLY_LAYOUT, height=540,
                               title=dict(text="Effort vs complexity of "
                                               "the portfolio", x=0.02),
                               legend=dict(font=dict(size=12)))
            st.plotly_chart(style_axes(fige), use_container_width=True)
        st.caption("Left: added export volume by sector and product (color "
                   "= effort). Right: each product positioned by effort and "
                   "estimated 2032 complexity (bubble size = required "
                   "volume).")

        # ── World markets ─────────────────────────────────────────────────
        band("🌐 World market for each recommended product")
        st.caption("Pick any recommended product: who the **competitors** "
                   "are (share of world exports) and where the **demand** "
                   "is concentrated (share of world imports).")
        pr = res["products"]
        popts = [f"{r.Product} — {r.Name}" for r in pr.itertuples()]
        cP, cY = st.columns([3, 1])
        with cP:
            ppick = st.selectbox("Recommended product", popts,
                                 key="market_prod")
        with cY:
            py = st.selectbox("Year", trade["years"],
                              index=len(trade["years"]) - 1,
                              key="market_year")
        hs = ppick.split(" — ")[0]

        exp_sh, exp_w = engine.product_world_shares(trade, "exports", hs, py)
        imp_sh, imp_w = engine.product_world_shares(trade, "imports", hs, py)
        own_e = exp_sh.loc[exp_sh["iso3"] == s["ISO3"], "share_pct"]
        own_e = float(own_e.iloc[0]) if len(own_e) else 0.0
        own_i = imp_sh.loc[imp_sh["iso3"] == s["ISO3"], "share_pct"]
        own_i = float(own_i.iloc[0]) if len(own_i) else 0.0

        w1, w2, w3 = st.columns(3)
        w1.metric("World trade in this product",
                  f"${exp_w/1e9:,.2f} B ({py})")
        w2.metric(f"{st.session_state['result_country']}'s export share",
                  f"{own_e:.2f} %")
        w3.metric(f"{st.session_state['result_country']}'s import share",
                  f"{own_i:.2f} %")

        st.plotly_chart(world_share_map(
            exp_sh, f"Competitors — world EXPORT shares ({py})",
            height=600, highlight=s["ISO3"]),
            use_container_width=True)
        top5 = exp_sh.head(5)
        st.caption("Top exporters: " + ", ".join(
            f"{r.name} ({r.share_pct:.1f}%)" for r in top5.itertuples()))

        st.plotly_chart(world_share_map(
            imp_sh, f"Demand — world IMPORT shares ({py})",
            height=600, highlight=s["ISO3"]),
            use_container_width=True)
        top5 = imp_sh.head(5)
        st.caption("Top importers: " + ", ".join(
            f"{r.name} ({r.share_pct:.1f}%)" for r in top5.itertuples()))

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
