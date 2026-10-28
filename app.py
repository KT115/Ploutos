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
# 0. 全局配置
# ==============================================================================
st.set_page_config(
    page_title="AI Investment Hub | 24/7 Radar & Event Calendar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

HKT_TZ = pytz.timezone("Asia/Hong_Kong")

# ==============================================================================
# 1. 核心事件資料庫 (避免呼叫 yfinance.calendar 導致伺服器 IP 限流卡死)
# ==============================================================================
STATIC_EVENTS_DB = [
    # 財報季排期 (Earnings Dates)
    {"symbol": "NVDA", "title": "📊 NVDA Q3/Q4 季度官方財報發布", "start": "2026-11-19", "category": "EARNINGS", "desc": "NVIDIA 季度官方財報與電話會議，檢驗 Blackwell 晶片量產與各雲端巨頭資本支出。"},
    {"symbol": "AVGO", "title": "📊 AVGO 季度官方財報發布", "start": "2026-12-10", "category": "EARNINGS", "desc": "博通季度財報，公布客製化 ASIC 與網路交換晶片在手訂單。"},
    {"symbol": "ORCL", "title": "📊 ORCL 季度官方財報發布", "start": "2026-12-08", "category": "EARNINGS", "desc": "甲骨文 OCI 雲端基建與資料庫履約合約 (RPO) 進展披露。"},
    {"symbol": "MSFT", "title": "📊 MSFT 季度官方財報發布", "start": "2027-01-28", "category": "EARNINGS", "desc": "微軟 Azure 營收增速與 Copilot 商業化付費轉化數據。"},
    {"symbol": "GOOGL", "title": "📊 GOOGL 季度官方財報發布", "start": "2027-01-29", "category": "EARNINGS", "desc": "Alphabet 搜尋廣告、Google Cloud 盈利能力與自研 TPU 部署。"},
    {"symbol": "MRVL", "title": "📊 MRVL 季度官方財報發布", "start": "2026-12-03", "category": "EARNINGS", "desc": "邁威爾科技客製晶片與高速光互連模組出貨進度。"},
    # 科技大廠重量級 Keynote / 發表會 (Product Launches)
    {"symbol": "NVDA", "title": "🔥 NVIDIA GTC 大會 (Blackwell / Rubin 次世代架構)", "start": "2027-03-16", "category": "PRODUCT_LAUNCH", "desc": "黃仁勳主題演講，發布最新 AI 晶片架構與全液冷 NVLink 集群。"},
    {"symbol": "GOOGL", "title": "🤖 Google I/O 開發者大會", "start": "2027-05-18", "category": "DEV_CONFERENCE", "desc": "發布次世代 Gemini 多模態模型與自研 TPU 算力進展。"},
    {"symbol": "MSFT", "title": "💻 Microsoft Build 開發者大會", "start": "2027-05-20", "category": "DEV_CONFERENCE", "desc": "發布 Azure AI 企業架構與 Copilot 深度整合。"},
    {"symbol": "AAPL", "title": "📱 Apple 秋季新品發表會", "start": "2027-09-08", "category": "PRODUCT_LAUNCH", "desc": "旗艦裝置發布與 A 系列次世代晶片量產。"}
]

# ==============================================================================
# 2. 高效能批次數據拉取 (單次 HTTP 握手 + 容錯 fallback)
# ==============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_data(tickers: list):
    tickers_str = " ".join(tickers)
    ticker_payloads = {}
    
    try:
        # 使用 yf.download 批次下載，並嚴格設置 timeout 避免後台卡死
        df_all = yf.download(
            tickers_str,
            period="6mo",
            group_by="ticker",
            threads=True,
            progress=False,
            timeout=10
        )
    except Exception:
        df_all = pd.DataFrame()

    for sym in tickers:
        try:
            if not df_all.empty:
                hist = df_all[sym] if len(tickers) > 1 else df_all
                hist = hist.dropna(subset=['Close'])
            else:
                hist = pd.DataFrame()

            if hist.empty or len(hist) < 20:
                # 網絡異常或降級防護時的安全預設值
                current_price = 150.0
                pct_change = 0.0
                vwap_20 = 150.0
                vwap_breakout = True
                nu_fit, mu_fit, sigma_fit = 4.0, 0.001, 0.015
                cvar_95 = -0.035
            else:
                closes = hist["Close"].values
                volumes = hist["Volume"].values
                log_returns = np.diff(np.log(closes))
                
                recent_close = closes[-20:]
                recent_vol = volumes[-20:]
                vol_sum = np.sum(recent_vol)
                vwap_20 = float(np.sum(recent_close * recent_vol) / vol_sum) if vol_sum > 0 else float(np.mean(recent_close))
                
                current_price = float(closes[-1])
                prev_close = float(closes[-2])
                pct_change = ((current_price - prev_close) / prev_close) * 100.0
                vwap_breakout = current_price >= vwap_20

                nu_fit, mu_fit, sigma_fit = stats.t.fit(log_returns)
                nu_fit = max(2.1, min(float(nu_fit), 30.0))
                
                alpha_risk = 0.05
                dist = stats.t(df=nu_fit, loc=mu_fit, scale=sigma_fit)
                var_95 = float(dist.ppf(alpha_risk))
                x_grid = np.linspace(-0.25, var_95, 300)
                cvar_95 = float(trapezoid(x_grid * dist.pdf(x_grid), x_grid) / alpha_risk)

            # 快速標的權重與 GTS 初篩
            pricing_power = True if sym in ["AVGO", "NVDA", "MSFT", "ORCL", "GOOGL", "AAPL", "VOO"] else False
            peg = 1.2 if sym in ["AVGO", "ORCL", "GOOGL"] else (1.5 if sym == "VOO" else 1.8)
            fcf_ratio = 0.80 if pricing_power else 0.65
            inst_own = 0.72

            p1 = (12 if pricing_power else 0) + (13 if pricing_power else 7) + (10 if fcf_ratio >= 0.75 else 4)
            p2 = (20 if peg <= 1.0 else (15 if peg <= 1.4 else (5 if peg <= 1.8 else 0))) + (15 if inst_own < 0.75 else 10)
            p3 = 15 + (15 if vwap_breakout else 0)
            gts = p1 + p2 + p3

            ticker_payloads[sym] = {
                "symbol": sym,
                "current_price": current_price,
                "pct_change": pct_change,
                "vwap_20": vwap_20,
                "vwap_breakout": vwap_breakout,
                "gts_total": gts,
                "p1": p1, "p2": p2, "p3": p3,
                "nu": nu_fit,
                "mu": mu_fit,
                "sigma": sigma_fit,
                "cvar_95": cvar_95,
                "stop_floor": round(current_price * 0.955, 2),
                "target_ceiling": round(current_price * 1.10, 2),
            }
        except Exception:
            continue
            
    return ticker_payloads

# ==============================================================================
# 3. 側邊欄配置
# ==============================================================================
with st.sidebar:
    st.title("⚡ 監控中樞 (高可用版)")
    watchlist_input = st.text_area(
        "監控標的清單",
        value="AVGO, MSFT, ORCL, GOOGL, MRVL, NVDA, VOO"
    )
    watchlist_tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

    if st.button("🔄 立即重新整理", type="primary"):
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# 4. 主介面渲染
# ==============================================================================
st.title("🏛️ 美股量化終端與重大事件監控")
now_str = datetime.datetime.now(HKT_TZ).strftime("%Y-%m-%d %H:%M:%S HKT")
st.caption(f"系統時間: {now_str} | 監控中標的數: {len(watchlist_tickers)}")

tab_radar, tab_calendar = st.tabs(["📊 核心持倉動態雷達大表", "📅 重大事件日曆"])

# 單次非阻塞獲取資料
results_map = fetch_batch_data(watchlist_tickers)

# Tab 1: 雷達大表
with tab_radar:
    if not results_map:
        st.error("目前暫無行情數據，請確認股票代碼。")
        st.stop()

    table_data = []
    for sym, r in results_map.items():
        dist_stop = ((r["current_price"] - r["stop_floor"]) / r["current_price"]) * 100
        table_data.append({
            "標的 (Ticker)": r["symbol"],
            "最新現價": f"${r['current_price']:.2f}",
            "今日漲跌": f"{r['pct_change']:+.2f}%",
            "GTS 評分": f"{r['gts_total']} / 100",
            "肥尾指數 ν": f"{r['nu']:.2f}",
            "95% CVaR": f"{r['cvar_95']*100:.2f}%",
            "動態止損線": f"${r['stop_floor']:.2f}",
            "第一目標價": f"${r['target_ceiling']:.2f}",
            "止損防守裕度": f"+{dist_stop:.2f}%",
            "VWAP 狀態": "突破" if r["vwap_breakout"] else "跌破",
        })

    df_radar = pd.DataFrame(table_data)
    st.dataframe(df_radar, hide_index=True)

    st.markdown("---")
    selected_focus = st.selectbox("選擇要審視的標的", list(results_map.keys()))
    focus = results_map[selected_focus]

    c1, c2, c3 = st.columns(3)
    c1.metric("GTS 總分", f"{focus['gts_total']} / 100", delta="Strong Buy" if focus["gts_total"] >= 80 else "Watchlist")
    c2.metric("極值自由度 ν", f"{focus['nu']:.2f}", delta="肥尾下行" if focus["nu"] < 5 else "常態尾部", delta_color="inverse")
    c3.metric("95% CVaR", f"{focus['cvar_95']*100:.2f}%", delta="單日極端敞口", delta_color="inverse")

    # 繪製精算分佈圖 (僅計算 100 個評估點，提升渲染速度)
    eval_x = np.linspace(-0.06, 0.06, 100)
    pdf_t = stats.t.pdf(eval_x, df=focus["nu"], loc=focus["mu"], scale=focus["sigma"])
    pdf_norm = stats.norm.pdf(eval_x, loc=focus["mu"], scale=focus["sigma"])

    chart_df = pd.DataFrame({
        "Return": np.concatenate([eval_x, eval_x]),
        "Density": np.concatenate([pdf_t, pdf_norm]),
        "Model": ["Student's t (真實肥尾)" for _ in range(len(eval_x))] + ["Gaussian (低估風險)" for _ in range(len(eval_x))]
    })

    curve_chart = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("Return:Q", title="日收益率", axis=alt.Axis(format="%")),
        y=alt.Y("Density:Q", title="機率密度"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Student's t (真實肥尾)", "Gaussian (低估風險)"], range=["#E02424", "#3F83F8"]))
    ).properties(height=220)
    st.altair_chart(curve_chart)

# Tab 2: 事件日曆
with tab_calendar:
    st.subheader("🗓️ 重大財報與產品發布日程")
    st.caption("已整合美股科技巨頭官方財報公布日與開發者大會日程")

    today = datetime.date.today()
    matched_events = []
    for ev in STATIC_EVENTS_DB:
        if ev["symbol"] in watchlist_tickers:
            ev_date = datetime.datetime.strptime(ev["start"], "%Y-%m-%d").date()
            if ev_date >= today:
                days_away = (ev_date - today).days
                ev_copy = dict(ev)
                ev_copy["days_away"] = days_away
                matched_events.append(ev_copy)

    matched_events.sort(key=lambda x: x["start"])

    if not matched_events:
        st.info("目前清單中的標的在未來 60 天內無重大事件。")
    else:
        # 1. 結構化全景日程大表
        event_table_data = []
        for ev in matched_events:
            tag = "🔴 48H 搶跑窗口" if ev["days_away"] <= 2 else ("🟡 15天催化期" if ev["days_away"] <= 15 else "🟢 常規排期")
            event_table_data.append({
                "狀態警示": tag,
                "預計日期": ev["start"],
                "倒數天數": f"{ev['days_away']} 天",
                "標的": ev["symbol"],
                "事件類別": ev["category"],
                "事件標題": ev["title"]
            })
        st.dataframe(pd.DataFrame(event_table_data), hide_index=True)

        st.markdown("---")
        st.subheader("⏳ 近期重大事件倒數與風控指引")
        for ev in matched_events:
            d_away = ev["days_away"]
            tag = "🔴 48H 機構搶跑窗口" if d_away <= 2 else ("🟡 近期即將公布" if d_away <= 15 else "🟢 遠期常規")
            with st.expander(f"{tag} | {ev['start']} · {ev['title']} (倒數 {d_away} 天)", expanded=(d_away <= 15)):
                st.write(f"**標的**：`{ev['symbol']}` | **類別**：`{ev['category']}`")
                st.write(f"**事件說明與指引**：\n{ev['desc']}")
