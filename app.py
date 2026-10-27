import datetime
import urllib.parse
import altair as alt
import numpy as np
import pandas as pd
import pytz
import requests
from scipy.integrate import trapezoid
import scipy.stats as stats
import streamlit as st
import yfinance as yf

# ==============================================================================
# 0. Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AI Investment Hub | Actuarial Risk & Live Alert Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

HKT_TZ = pytz.timezone("Asia/Hong_Kong")

# ==============================================================================
# 1. Market Data Fetcher & Distribution Fitting
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_live_market_data(symbol: str):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    if hist.empty:
        return None

    # Calculate logarithmic daily returns
    hist["Log_Return"] = np.log(hist["Close"] / hist["Close"].shift(1))
    returns = hist["Log_Return"].dropna().values

    # Rolling 20-period Volume-Weighted Average Price (VWAP)
    recent_20 = hist.tail(20)
    vol_sum = recent_20["Volume"].sum()
    vwap_20 = (
        (recent_20["Close"] * recent_20["Volume"]).sum() / vol_sum
        if vol_sum > 0
        else float(recent_20["Close"].mean())
    )
    current_price = float(hist["Close"].iloc[-1])
    vwap_breakout = current_price >= vwap_20

    # Fundamental proxies
    info = stock.info
    peg = info.get("pegRatio", None)
    if peg is None or peg <= 0:
        fwd_pe = info.get("forwardPE", 25.0)
        peg = round(fwd_pe / 20.0, 2)

    inst_ownership = float(info.get("heldPercentInstitutions", 0.75))
    gross_margin = float(info.get("grossMargins", 0.0))
    roe = float(info.get("returnOnEquity", 0.0))
    pricing_power = bool((gross_margin >= 0.45) or (roe >= 0.18))

    fcf = info.get("freeCashflow", 0)
    net_income = info.get("netIncomeToCommon", 1)
    if net_income and net_income > 0 and fcf:
        fcf_conversion = float(min(max(fcf / net_income, 0.4), 1.0))
    else:
        fcf_conversion = 0.75

    # Actuarial Student's t parameter estimation (degrees of freedom nu, location mu, scale sigma)
    nu_fit, mu_fit, sigma_fit = stats.t.fit(returns)
    nu_fit = max(2.1, min(float(nu_fit), 30.0))

    return {
        "current_price": current_price,
        "returns": returns,
        "vwap_20": float(vwap_20),
        "vwap_breakout": vwap_breakout,
        "forward_peg": float(peg),
        "inst_ownership": inst_ownership,
        "fcf_conversion": fcf_conversion,
        "pricing_power": pricing_power,
        "days_to_earnings": 14,
        "nrr": 125 if pricing_power else 108,
        "mu": float(mu_fit),
        "sigma": float(sigma_fit),
        "nu": float(nu_fit),
    }

# ==============================================================================
# 2. Golden Triad Scoring (GTS: 0-100) & Risk Management Verification
# ==============================================================================
def calculate_gts(p1_data, p2_data, p3_data):
    # Pillar 1: DNA (35 pts, passing score: 21)
    p1_pricing = 12 if p1_data.get("pricing_power") else 0
    p1_nrr = 13 if p1_data.get("nrr", 0) >= 120 else (7 if p1_data.get("nrr", 0) >= 110 else 0)
    p1_fcf = 10 if p1_data.get("fcf_conversion", 0) >= 0.75 else 4
    p1_score = p1_pricing + p1_nrr + p1_fcf

    # Pillar 2: Expectation Gap (35 pts, passing score: 21)
    peg = p2_data.get("forward_peg", 2.0)
    p2_peg = 20 if peg <= 1.0 else (15 if peg <= 1.4 else (5 if peg <= 1.8 else 0))
    inst_own = p2_data.get("inst_ownership", 1.0)
    p2_crowd = 15 if inst_own < 0.75 else (10 if inst_own < 0.85 else 0)
    p2_score = p2_peg + p2_crowd

    # Pillar 3: Catalyst (30 pts, passing score: 18)
    dte = p3_data.get("days_to_earnings", 999)
    p3_pead = 15 if 0 <= dte <= 15 else 0
    p3_vwap = 15 if p3_data.get("vwap_breakout") else 0
    p3_score = p3_pead + p3_vwap

    gts_total = p1_score + p2_score + p3_score
    p1_pass = p1_score >= 21
    p2_pass = p2_score >= 21
    p3_pass = p3_score >= 18

    if gts_total >= 80 and p1_pass and p2_pass and p3_pass:
        rating = "Strong Buy (Model B Core Alpha Focus)"
        allocation = "20% ~ 30% Heavy Allocation (Kelly Bound)"
    elif gts_total >= 65:
        rating = "Buy (Watchlist / Rotation Candidate)"
        allocation = "8% ~ 10% Tactical Allocation"
    elif gts_total >= 50:
        rating = "Hold / Monitor"
        allocation = "Maintain Existing Weight"
    else:
        rating = "Avoid / Prune"
        allocation = "0% (Prohibited Allocation)"

    return {
        "p1_score": p1_score,
        "p1_pass": p1_pass,
        "p2_score": p2_score,
        "p2_pass": p2_pass,
        "p3_score": p3_score,
        "p3_pass": p3_pass,
        "gts_total": gts_total,
        "rating": rating,
        "allocation": allocation,
    }

def verify_cro_compliance(portfolio_drawdown: float, cvar_95: float):
    now_hkt = datetime.datetime.now(HKT_TZ)
    market_open = now_hkt.replace(hour=21, minute=30, second=0, microsecond=0)
    market_close = now_hkt.replace(hour=23, minute=59, second=59, microsecond=0)
    in_window = market_open <= now_hkt <= market_close
    t30_window = now_hkt >= now_hkt.replace(hour=22, minute=0, second=0, microsecond=0) if in_window else False

    veto = False
    reasons = []

    if portfolio_drawdown < -0.08:
        veto = True
        reasons.append("🚨 Cumulative portfolio drawdown breached -8.0% limit. CRO absolute veto invoked.")
    if cvar_95 < -0.045:
        reasons.append("⚠️ Daily CVaR (95%) breached -4.50% boundary. Downside tail risk elevated.")
    if not in_window:
        reasons.append(f"⏳ Outside trading window (HKT 21:30–24:00 only, Current: {now_hkt.strftime('%H:%M:%S')}).")
    elif not t30_window:
        reasons.append("⏳ Pre-market noise dampening active. Orders execute after t+30m delay (HKT 22:00).")

    return {
        "cro_veto": veto,
        "in_window": in_window,
        "t30_active": t30_window,
        "hkt_time": now_hkt.strftime("%Y-%m-%d %H:%M:%S HKT"),
        "reasons": reasons,
    }

# ==============================================================================
# 3. WhatsApp Notification Dispatcher (CallMeBot API)
# ==============================================================================
def send_whatsapp_alert(phone: str, apikey: str, message: str) -> bool:
    if not phone or not apikey:
        return False
    encoded_text = urllib.parse.quote(message)
    endpoint = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={encoded_text}&apikey={apikey}"
    try:
        response = requests.get(endpoint, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

# ==============================================================================
# 4. Sidebar Controls (Clean Layout Without Toggles)
# ==============================================================================
with st.sidebar:
    st.title("🏛️ Quant Terminal")
    st.caption("Live Yahoo Finance & Actuarial Model")
    st.markdown("---")

    st.subheader("Ticker & Risk Parameters")
    target_ticker = st.text_input("Equity Ticker", value="AVGO").upper()
    sim_portfolio_dd = st.slider("Simulated Portfolio Drawdown", -0.15, 0.05, -0.025, 0.005, format="%.3f")

    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.subheader("WhatsApp Notification Config")
    wa_phone = st.text_input("Phone (e.g., 852XXXXXXXX)", value="")
    wa_apikey = st.text_input("CallMeBot API Key", type="password", value="")

# ==============================================================================
# 5. Core Execution & Computations
# ==============================================================================
with st.spinner(f"Retrieving market data and fundamentals for {target_ticker}..."):
    market_data = fetch_live_market_data(target_ticker)

if market_data is None:
    st.error(f"Failed to fetch market data for {target_ticker}. Please confirm the ticker symbol.")
    st.stop()

# Compute GTS
gts_output = calculate_gts(
    p1_data={
        "pricing_power": market_data["pricing_power"],
        "nrr": market_data["nrr"],
        "fcf_conversion": market_data["fcf_conversion"],
    },
    p2_data={
        "forward_peg": market_data["forward_peg"],
        "inst_ownership": market_data["inst_ownership"],
    },
    p3_data={
        "days_to_earnings": market_data["days_to_earnings"],
        "vwap_breakout": market_data["vwap_breakout"],
    },
)

# Actuarial 95% CVaR computation via scipy.integrate.trapezoid (NumPy 2.x safe)
alpha_risk = 0.05
fitted_dist = stats.t(
    df=market_data["nu"],
    loc=market_data["mu"],
    scale=market_data["sigma"],
)
var_95 = float(fitted_dist.ppf(alpha_risk))
x_grid = np.linspace(-0.25, var_95, 1000)
cvar_95 = float(trapezoid(x_grid * fitted_dist.pdf(x_grid), x_grid) / alpha_risk)

# Risk officer verification
cro_status = verify_cro_compliance(sim_portfolio_dd, cvar_95)

# ==============================================================================
# 6. Main Dashboard Render
# ==============================================================================
st.title(f"📊 {target_ticker} Live Quant & Actuarial Monitor (${market_data['current_price']:.2f})")
st.caption(f"Last updated: {cro_status['hkt_time']} | 20-Day Rolling VWAP: ${market_data['vwap_20']:.2f}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("GTS Score", f"{gts_output['gts_total']} / 100", delta=gts_output["rating"])
k2.metric(
    "Fitted DoF (ν)",
    f"{market_data['nu']:.2f}",
    delta="Severe Fat Tail" if market_data["nu"] < 5.0 else "Tempered Tail",
    delta_color="inverse",
)
k3.metric("95% Expected Shortfall (CVaR)", f"{cvar_95*100:.2f}%", delta="Downside Tail Exposure", delta_color="inverse")

if cro_status["cro_veto"]:
    k4.metric("Risk Status", "VETOED", delta="-8% Max Drawdown", delta_color="inverse")
elif not cro_status["in_window"]:
    k4.metric("Trading Window", "Off-Market", delta="Active 21:30–24:00 HKT", delta_color="off")
else:
    k4.metric("Risk Status", "APPROVED", delta="Window Validated")

st.markdown("---")

# Section 1: GTS Breakdown Grid
st.subheader("🏛️ Golden Triad Scoring Audit")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**Pillar 1: Structural DNA (`{gts_output['p1_score']}/35` pts)**")
    st.progress(gts_output["p1_score"] / 35.0)
    st.caption(f"Pricing Power: `{'Yes' if market_data['pricing_power'] else 'No'}` | FCF Ratio: `{market_data['fcf_conversion']*100:.0f}%`")

with c2:
    st.markdown(f"**Pillar 2: Expectation Gap (`{gts_output['p2_score']}/35` pts)**")
    st.progress(gts_output["p2_score"] / 35.0)
    st.caption(f"Forward PEG: `{market_data['forward_peg']}x` | Inst. Ownership: `{market_data['inst_ownership']*100:.1f}%`")

with c3:
    st.markdown(f"**Pillar 3: Catalysts (`{gts_output['p3_score']}/30` pts)**")
    st.progress(gts_output["p3_score"] / 30.0)
    st.caption(f"VWAP Status: `{'Breakout' if market_data['vwap_breakout'] else 'Below VWAP'}` | Earnings Window: `{market_data['days_to_earnings']}d`")

st.markdown("---")

# Section 2: Fitted Student's t Distribution vs Gaussian Baseline
st.subheader("📈 Actuarial Empirical Tail-Risk Distribution")
eval_range = np.linspace(-0.08, 0.08, 400)
pdf_student = stats.t.pdf(
    eval_range,
    df=market_data["nu"],
    loc=market_data["mu"],
    scale=market_data["sigma"],
)
pdf_gaussian = stats.norm.pdf(
    eval_range,
    loc=market_data["mu"],
    scale=market_data["sigma"],
)

plot_df = pd.DataFrame({
    "Return": np.concatenate([eval_range, eval_range]),
    "Density": np.concatenate([pdf_student, pdf_gaussian]),
    "Distribution": (
        ["Fitted Student's t (Tail Risk)" for _ in range(len(eval_range))]
        + ["Gaussian Baseline (Underestimates Risk)" for _ in range(len(eval_range))]
    ),
})

density_chart = (
    alt.Chart(plot_df)
    .mark_line()
    .encode(
        x=alt.X("Return:Q", title="Daily Return Range", axis=alt.Axis(format="%")),
        y=alt.Y("Density:Q", title="Probability Density"),
        color=alt.Color(
            "Distribution:N",
            scale=alt.Scale(
                domain=["Fitted Student's t (Tail Risk)", "Gaussian Baseline (Underestimates Risk)"],
                range=["#E02424", "#3F83F8"],
            ),
        ),
    )
    .properties(height=320)
)
st.altair_chart(density_chart, use_container_width=True)

st.markdown("---")

# Section 3: Committee Action SOP & Dispatcher
st.subheader("🤖 Advisory Directives & WhatsApp Dispatcher")
col_sop, col_notify = st.columns([1.2, 1])

stop_price = market_data["current_price"] * 0.955
take_profit_price = market_data["current_price"] * 1.10

with col_sop:
    if cro_status["cro_veto"]:
        st.error(f"### 🛑 CRO VETO ACTIVE\n{cro_status['reasons'][0]}")
    else:
        sop_box = st.success if gts_output["gts_total"] >= 80 else (st.warning if gts_output["gts_total"] >= 65 else st.info)
        sop_box(
            f"### 🎯 Evaluation: {gts_output['rating']}\n"
            f"- **Market Price**: `${market_data['current_price']:.2f}` (20-Day VWAP: `${market_data['vwap_20']:.2f}`)\n"
            f"- **Allocation Strategy**: {gts_output['allocation']}\n"
            f"- **Trailing Stop Floor**: `${stop_price:.2f}` (-4.5% Dynamic Stop-Loss)\n"
            f"- **First Target Threshold**: `${take_profit_price:.2f}` (+10.0% Stepped Lock-in)"
        )

    for notice in cro_status["reasons"]:
        st.warning(notice)

with col_notify:
    st.markdown("**📲 Push Alert to WhatsApp**")
    st.write("Dispatches current valuation metrics, stop levels, and risk parameters directly to your device via CallMeBot.")

    dispatch_text = (
        f"📈 [AI Investment Hub Alert]\n\n"
        f"Ticker: {target_ticker}\n"
        f"Price: ${market_data['current_price']:.2f}\n"
        f"GTS Score: {gts_output['gts_total']}/100 ({gts_output['rating']})\n"
        f"Stop-Loss: ${stop_price:.2f}\n"
        f"95% CVaR: {cvar_95*100:.2f}%\n"
        f"Time: {cro_status['hkt_time']}"
    )

    if st.button("📤 Send WhatsApp Summary", use_container_width=True):
        if not wa_phone or not wa_apikey:
            st.error("Please configure your Phone Number and CallMeBot API Key in the sidebar.")
        else:
            with st.spinner("Dispatching WhatsApp alert..."):
                success = send_whatsapp_alert(wa_phone, wa_apikey, dispatch_text)
                if success:
                    st.success("Alert successfully sent to WhatsApp.")
                else:
                    st.error("Failed to send message. Verify phone format (e.g., 852XXXXXXXX) and API key validity.")
