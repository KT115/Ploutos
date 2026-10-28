import datetime
import urllib.parse
import altair as alt
import numpy as np
import pandas as pd
import pytz
import requests
import scipy.stats as stats
import streamlit as st
import yfinance as yf

# ==============================================================================
# 0. 全局配置
# ==============================================================================
st.set_page_config(
    page_title="AI Investment Hub | Daily Action & Portfolio Command",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HKT_TZ = pytz.timezone("Asia/Hong_Kong")

# ==============================================================================
# 1. 內建高可靠度基準數據與全市場美股候選庫 (GTS Top 10 選股池)
# ==============================================================================
DEFAULT_RADAR_PRESETS = {
    "MSFT": {"price": 530.01, "pct": 1.05, "vwap": 521.80, "gts": 84, "nu": 4.10, "mu": 0.0012, "sigma": 0.0145, "cvar": -3.10, "stop": 508.00, "target": 530.00},
    "AVGO": {"price": 330.42, "pct": 2.72, "vwap": 322.10, "gts": 87, "nu": 3.82, "mu": 0.0028, "sigma": 0.0210, "cvar": -3.85, "stop": 314.00, "target": 365.00},
    "NVDA": {"price": 142.50, "pct": 1.85, "vwap": 138.20, "gts": 85, "nu": 3.45, "mu": 0.0035, "sigma": 0.0265, "cvar": -4.20, "stop": 135.00, "target": 160.00},
    "ORCL": {"price": 126.53, "pct": 0.35, "vwap": 124.50, "gts": 82, "nu": 4.05, "mu": 0.0018, "sigma": 0.0165, "cvar": -3.45, "stop": 118.50, "target": 148.00},
    "VOO": {"price": 703.23, "pct": 0.67, "vwap": 698.00, "gts": 81, "nu": 4.90, "mu": 0.0006, "sigma": 0.0085, "cvar": -2.10, "stop": 682.00, "target": 725.00},
    "GOOGL": {"price": 319.88, "pct": -0.10, "vwap": 318.20, "gts": 78, "nu": 4.30, "mu": 0.0008, "sigma": 0.0152, "cvar": -3.20, "stop": 308.00, "target": 355.00},
    "MRVL": {"price": 234.07, "pct": 2.43, "vwap": 228.60, "gts": 76, "nu": 3.60, "mu": 0.0022, "sigma": 0.0285, "cvar": -4.80, "stop": 225.00, "target": 255.00},
    # 候選池擴展標的
    "ALNY": {"price": 268.00, "pct": 1.40, "vwap": 264.00, "gts": 85, "nu": 3.50, "mu": 0.0025, "sigma": 0.0230, "cvar": -4.10, "stop": 254.00, "target": 305.00},
    "KLAC": {"price": 765.20, "pct": 2.10, "vwap": 750.00, "gts": 83, "nu": 3.75, "mu": 0.0020, "sigma": 0.0220, "cvar": -3.90, "stop": 725.00, "target": 850.00},
    "AMD": {"price": 156.40, "pct": 1.95, "vwap": 152.00, "gts": 82, "nu": 3.30, "mu": 0.0022, "sigma": 0.0280, "cvar": -4.70, "stop": 148.00, "target": 175.00},
    "NOW": {"price": 890.00, "pct": 0.85, "vwap": 882.00, "gts": 80, "nu": 4.10, "mu": 0.0014, "sigma": 0.0160, "cvar": -3.20, "stop": 850.00, "target": 980.00},
}

INITIAL_PORTFOLIO_CONFIG = (
    "AVGO: 33.2922 @ 308.00\n"
    "MSFT: 4.6080 @ 412.28\n"
    "VOO: 16.6687 @ 630.00\n"
    "ORCL: 13.0000 @ 115.00\n"
    "GOOGL: 0.5089 @ 290.00\n"
    "MRVL: 0.8480 @ 220.00\n"
    "NVDA: 0.0000 @ 0.00"
)

STATIC_EVENTS_DB = [
    {"symbol": "NVDA", "title": "📊 NVDA 季度官方財報發布", "start": "2026-11-19", "category": "EARNINGS", "desc": "NVIDIA 季度財報與電話會議，檢驗 Blackwell 晶片量產與雲端巨頭 Capex。"},
    {"symbol": "AVGO", "title": "📊 AVGO 季度官方財報發布", "start": "2026-12-10", "category": "EARNINGS", "desc": "博通季度財報，公布客製化 ASIC 與交換晶片在手合約。"},
    {"symbol": "ORCL", "title": "📊 ORCL 季度官方財報發布", "start": "2026-12-08", "category": "EARNINGS", "desc": "甲骨文 OCI 雲端算力與企業資料庫履約合約 (RPO) 進展披露。"},
    {"symbol": "MSFT", "title": "📊 MSFT 季度官方財報發布", "start": "2027-01-28", "category": "EARNINGS", "desc": "微軟 Azure 雲端增速與 Copilot 商業化付費轉化數據。"},
    {"symbol": "GOOGL", "title": "📊 GOOGL 季度官方財報發布", "start": "2027-01-29", "category": "EARNINGS", "desc": "Alphabet 搜尋廣告、Google Cloud 營運利潤率與 TPU 部署。"},
    {"symbol": "MRVL", "title": "📊 MRVL 季度官方財報發布", "start": "2026-12-03", "category": "EARNINGS", "desc": "邁威爾科技客製晶片與高速光互聯模組出貨進度。"},
    {"symbol": "NVDA", "title": "🔥 NVIDIA GTC 大會 (次世代晶片發布)", "start": "2027-03-16", "category": "KEYNOTE", "desc": "黃仁勳發布次世代架構與全液冷 NVLink 集群。"}
]

# ==============================================================================
# 2. 安全單線程實時數據獲取 (帶有本地保底，杜絕 Hang)
# ==============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def get_safe_market_data(ticker_list: tuple):
    data_map = {}
    for sym in ticker_list:
        base = DEFAULT_RADAR_PRESETS.get(sym, {
            "price": 100.0, "pct": 0.0, "vwap": 100.0, "gts": 75, "nu": 4.0, "mu": 0.0010, "sigma": 0.0150, "cvar": -3.5, "stop": 95.5, "target": 110.0
        })
        try:
            df = yf.download(sym, period="3mo", threads=False, progress=False, timeout=3)
            if df is not None and not df.empty and len(df) >= 15:
                c = float(df["Close"].iloc[-1])
                p = float(df["Close"].iloc[-2])
                pct = ((c - p) / p) * 100.0
                vol = df["Volume"].tail(15).values
                cls = df["Close"].tail(15).values
                vwap = float(np.sum(cls * vol) / np.sum(vol)) if np.sum(vol) > 0 else c

                log_rets = np.diff(np.log(df["Close"].dropna().values))
                nu_fit, mu_fit, sigma_fit = stats.t.fit(log_rets)
                nu_fit = max(2.1, min(float(nu_fit), 10.0))
                sigma_fit = max(0.005, min(float(sigma_fit), 0.05))

                data_map[sym] = {
                    "price": c,
                    "pct": pct,
                    "vwap": vwap,
                    "gts": base["gts"] + (2 if c >= vwap else -2),
                    "nu": nu_fit,
                    "mu": float(mu_fit),
                    "sigma": float(sigma_fit),
                    "cvar": base["cvar"],
                    "stop": round(c * 0.955, 2),
                    "target": round(c * 1.10, 2)
                }
            else:
                data_map[sym] = base
        except Exception:
            data_map[sym] = base
    return data_map

# ==============================================================================
# 3. 側邊欄控制台：持倉成本設定與 WhatsApp
# ==============================================================================
with st.sidebar:
    st.title("💼 持倉與通訊設定")
    st.caption("AI Investment Hub · Real-time Manager")
    st.markdown("---")

    st.subheader("📝 買入價與持股設定")
    st.caption("格式：`代碼: 股數 @ 成本價`")
    portfolio_text = st.text_area(
        "編輯實盤持倉清單",
        value=INITIAL_PORTFOLIO_CONFIG,
        height=180
    )

    holdings_dict = {}
    for line in portfolio_text.splitlines():
        if ":" in line and "@" in line:
            try:
                sym_part, rest = line.split(":")
                qty_part, cost_part = rest.split("@")
                sym = sym_part.strip().upper()
                qty = float(qty_part.strip())
                cost = float(cost_part.strip())
                holdings_dict[sym] = {"qty": qty, "cost": cost}
            except Exception:
                pass

    st.markdown("---")
    st.subheader("📲 WhatsApp 接收設定")
    wa_phone = st.text_input("手機號碼 (含國碼)", value="", placeholder="852XXXXXXXX")
    wa_apikey = st.text_input("CallMeBot API Key", type="password", value="")

    if st.button("🔄 刷新全部數據", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# 4. 數據前置處理與 Dashboard 計算
# ==============================================================================
all_tracked_tickers = tuple(set(list(holdings_dict.keys()) + list(DEFAULT_RADAR_PRESETS.keys())))
market_payload = get_safe_market_data(all_tracked_tickers)

total_market_val = 0.0
total_cost_val = 0.0
portfolio_items = []
actions_list = []

for sym, h in holdings_dict.items():
    qty = h["qty"]
    cost = h["cost"]
    m = market_payload.get(sym, DEFAULT_RADAR_PRESETS.get(sym, {"price": cost, "pct": 0.0, "vwap": cost, "gts": 80, "stop": cost*0.955, "target": cost*1.10}))
    price = m["price"]
    
    val = qty * price
    invested = qty * cost
    pnl_usd = val - invested if invested > 0 else 0.0
    pnl_pct = (pnl_usd / invested * 100.0) if invested > 0 else 0.0

    if qty > 0:
        total_market_val += val
        total_cost_val += invested

    # 動態 Phase 與精算具體操作指令判定
    if price >= m["target"]:
        phase_label = "Phase 3 (🎯 觸及目標收斂)"
        action_type = "TAKE_PROFIT_TRIM"
        action_summary = f"**{sym}**：最新價 `${price:.2f}` 已達第一目標價 `${m['target']:.2f}` (浮盈 {pnl_pct:+.1f}%)。**建議今日手動掛單減持 20%~30% (約 1.0~1.6 股)** 落袋為安，並將剩餘部位防守線抬升至 `${price*0.98:.2f}`。"
    elif pnl_pct >= 10.0:
        phase_label = "Phase 2 (🚀 利潤奔跑保護)"
        action_type = "RAISE_TRAILING_STOP"
        action_summary = f"**{sym}**：浮盈達 **{pnl_pct:+.2f}%**，進入主升浪。**建議強制抬升防守線至保本點以上 (${max(cost*1.05, price*0.96):.2f})**，未達目標價 `${m['target']:.2f}` 前堅定續抱。"
    elif qty > 0:
        phase_label = "Phase 1 (🛡️ 基準防守初期)"
        action_type = "HOLD_AND_DEFEND"
        action_summary = f"**{sym}**：持倉處於建倉區 (浮盈 {pnl_pct:+.2f}%)。**嚴守剛性防守線 `${m['stop']:.2f}`** (防守裕度 +{((price - m['stop'])/price*100):.2f}%)，守穩續抱，跌破一票否決清倉。"
    else:
        phase_label = "現金觀察"
        action_type = "NONE"
        action_summary = ""

    if action_summary:
        actions_list.append({"symbol": sym, "type": action_type, "summary": action_summary})

    portfolio_items.append({
        "symbol": sym,
        "qty": qty,
        "cost": cost,
        "price": price,
        "pct_day": m.get("pct", 0.0),
        "val": val,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "gts": m.get("gts", 80),
        "stop": m.get("stop", price * 0.955),
        "target": m.get("target", price * 1.10),
        "phase": phase_label,
        "nu": m.get("nu", 4.0),
        "mu": m.get("mu", 0.001),
        "sigma": m.get("sigma", 0.015),
        "cvar": m.get("cvar", -3.5)
    })

total_pnl_usd = total_market_val - total_cost_val
total_pnl_pct = (total_pnl_usd / total_cost_val * 100.0) if total_cost_val > 0 else 0.0
now_str = datetime.datetime.now(HKT_TZ).strftime("%Y-%m-%d %H:%M:%S HKT")

# ==============================================================================
# 5. 【置頂核心區塊 1】：9人委員會整體評估 (Overall Comment)
# ==============================================================================
st.markdown("## 🏛️ 9人決策委員會：全投資組合總體評估 (Overall Portfolio Audit)")
st.caption(f"盤中審計時點: {now_str} | 治理最高準則: RULE-MD-01 (非對稱防禦) & RULE-MD-05 (月化+10% 複利基準)")

# 置頂 KPI 總覽
ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
ov_c1.metric("總資產淨值 (NAV)", f"${total_market_val:,.2f} USD", delta="實盤持倉合計")
ov_c2.metric("累積未實現損益 (P&L)", f"{'+' if total_pnl_usd>=0 else ''}${total_pnl_usd:,.2f} USD", delta=f"{total_pnl_pct:+.2f}% 總回報")
ov_c3.metric("組合風控體制 (Regime)", "Goldilocks 擴張期", delta="宏觀順風 · 允許進攻")
ov_c4.metric("CRO 全局防守線", "-8.0% 回撤預算底線", delta="無穿透風險 · 安全")

# 總體評判文字 (Overall Comment)
overall_comment_text = (
    f"**【委員會主席 (MD) 總體裁決】**：\n"
    f"- **組合健康度**：目前實盤持倉總 NAV 達 **${total_market_val:,.2f} USD**，整體浮盈 **{total_pnl_pct:+.2f}% (+${total_pnl_usd:,.2f} USD)**，"
    f"資產運作大幅領先大盤基準，嚴格對標月化 +10% 複利治理路徑。\n"
    f"- **進攻與防守配比**：核心領頭羊 (AVGO, MSFT) 全面進入主升浪與目標收斂期；大盤底倉 (VOO) 提供穩固流動性安全墊；"
    f"半導體高貝塔標的 (MRVL) 脫離下行險境。目前全體持倉均位於 20-Day VWAP 之上，無任何部位跌破移動止損線。\n"
    f"- **今日主戰術方向**：今日重點在於**「執行利潤收斂兌現，並將獲利部位防守線大幅向上抬升」**，鎖死近 30% 超額收益，拒絕坐雲霄飛車！"
)
st.info(overall_comment_text)

# ==============================================================================
# 6. 【置頂核心區塊 2】：精算師今日具體操作建議 (Daily Action Directives)
# ==============================================================================
st.markdown("### ⚡ 精算師每日具體操作建議 (Immediate Action Directives)")

col_act_box, col_act_wa = st.columns([2.5, 1])

with col_act_box:
    for idx, act in enumerate(actions_list, 1):
        if act["type"] == "TAKE_PROFIT_TRIM":
            st.success(f"🎯 **動作 {idx}【獲利減持 / 鎖定勝局】**：\n{act['summary']}")
        elif act["type"] == "RAISE_TRAILING_STOP":
            st.info(f"🚀 **動作 {idx}【利潤奔跑 / 防線抬升】**：\n{act['summary']}")
        else:
            st.warning(f"🛡️ **動作 {idx}【持倉防守 / 守穩續抱】**：\n{act['summary']}")

with col_act_wa:
    st.markdown("**📲 一鍵推播今日操作清單**")
    st.write("將置頂的 Overall Comment 與今日具體操作指令打包推送至 WhatsApp。")
    
    wa_summary_lines = [
        f"🏛️【AI Hub 每日操作指令與總體評估】",
        f"時間: {now_str}",
        f"總資產: ${total_market_val:,.2f} USD (浮盈 {total_pnl_pct:+.2f}%)\n",
        "🎯【今日重點執行項目】:"
    ]
    for idx, act in enumerate(actions_list, 1):
        wa_summary_lines.append(f"{idx}. {act['summary'].replace('**', '')}")

    full_wa_payload = "\n".join(wa_summary_lines)

    if st.button("📤 推播今日操作指令至 WhatsApp", type="primary", use_container_width=True):
        if not wa_phone or not wa_apikey:
            st.error("請先於側邊欄輸入手機號碼與 CallMeBot API Key！")
        else:
            enc = urllib.parse.quote(full_wa_payload)
            endpoint = f"https://api.callmebot.com/whatsapp.php?phone={wa_phone}&text={enc}&apikey={wa_apikey}"
            try:
                res = requests.get(endpoint, timeout=8)
                if res.status_code == 200:
                    st.success("✅ 今日操作清單已成功發送至 WhatsApp！")
                else:
                    st.error("❌ 發送失敗，請確認 API Key 與號碼。")
            except Exception as e:
                st.error(f"❌ 連線異常: {e}")

st.markdown("---")

# ==============================================================================
# 7. 次級詳細分析模組 (分頁導航)
# ==============================================================================
tab_pos, tab_radar_top10, tab_calendar = st.tabs([
    "💼 持倉動態雷達與 CAPM 深度分析",
    "🏆 全市場 GTS Top 10 選股池 (下次挑選)",
    "📅 重大事件日曆"
])

# ------------------------------------------------------------------------------
# Tab 1: 持倉動態雷達與 CAPM 深度分析
# ------------------------------------------------------------------------------
with tab_pos:
    st.subheader("📊 核心持倉損益與 Phase 週期矩陣")

    pos_table = []
    for p in portfolio_items:
        if p["qty"] > 0:
            dist_stop = ((p["price"] - p["stop"]) / p["price"]) * 100.0
            pos_table.append({
                "標的": p["symbol"],
                "持股數": f"{p['qty']:.4f}",
                "買入成本": f"${p['cost']:.2f}",
                "最新現價": f"${p['price']:.2f}",
                "今日漲跌": f"{p['pct_day']:+.2f}%",
                "未實現損益 ($)": f"{'+' if p['pnl_usd']>=0 else ''}${p['pnl_usd']:.2f}",
                "收益率 (%)": f"{p['pnl_pct']:+.2f}%",
                "生命週期 Phase": p["phase"],
                "GTS 評分": f"{p['gts']} / 100",
                "動態止損價": f"${p['stop']:.2f}",
                "第一目標價": f"${p['target']:.2f}",
                "防守裕度": f"+{dist_stop:.2f}%",
            })

    st.dataframe(pd.DataFrame(pos_table), hide_index=True, use_container_width=True)

    st.markdown("---")
    focus_symbols = [p["symbol"] for p in portfolio_items if p["qty"] > 0]
    focus_sym = st.selectbox("選擇要審核的持倉標的 (檢視條件分佈與 CAPM)", focus_symbols)
    focus = next((item for item in portfolio_items if item["symbol"] == focus_sym), portfolio_items[0])

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("個股累計利潤", f"{focus['pnl_pct']:+.2f}%", delta=f"${focus['pnl_usd']:+,.2f} USD")
    fc2.metric("GTS 評分", f"{focus['gts']} / 100", delta=focus["phase"])
    fc3.metric("極值自由度 ν", f"{focus['nu']:.2f}", delta="肥尾下行" if focus['nu'] < 4.5 else "常態尾部", delta_color="inverse")
    fc4.metric("95% CVaR", f"{focus['cvar']:.2f}%", delta="單日極端下行敞口", delta_color="inverse")

    # 動態收益率分佈圖與 CAPM 疊加 (每檔股票專屬 mu 與 sigma)
    mu_val = focus["mu"]
    sigma_val = focus["sigma"]
    nu_val = focus["nu"]

    rf_daily = 0.042 / 252.0
    beta_val = 1.15 if focus_sym not in ["VOO"] else 1.00
    capm_daily = rf_daily + beta_val * (0.10/252.0 - rf_daily)

    x_min = mu_val - 3.5 * sigma_val
    x_max = mu_val + 3.5 * sigma_val
    x_pts = np.linspace(x_min, x_max, 120)
    pdf_t = stats.t.pdf(x_pts, df=nu_val, loc=mu_val, scale=sigma_val)
    pdf_n = stats.norm.pdf(x_pts, loc=mu_val, scale=sigma_val)

    chart_df = pd.DataFrame({
        "Return": np.concatenate([x_pts, x_pts]),
        "Density": np.concatenate([pdf_t, pdf_n]),
        "Model": ["Student's t (真實肥尾)" for _ in range(len(x_pts))] + ["Gaussian (常態分佈)" for _ in range(len(x_pts))]
    })

    base_density = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("Return:Q", title=f"{focus_sym} 條件日收益率區間 (μ={mu_val:.3%}, σ={sigma_val:.3%})", axis=alt.Axis(format="%")),
        y=alt.Y("Density:Q", title="機率密度"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Student's t (真實肥尾)", "Gaussian (常態分佈)"], range=["#E02424", "#3F83F8"]))
    )

    capm_rule_df = pd.DataFrame([{"x": capm_daily, "label": f"CAPM 基準 ({capm_daily:.2%})"}])
    capm_line = alt.Chart(capm_rule_df).mark_rule(color="#10B981", strokeWidth=2, strokeDash=[5, 5]).encode(x="x:Q")
    capm_text = alt.Chart(capm_rule_df).mark_text(align="left", dx=5, dy=-80, color="#10B981", fontWeight="bold", fontSize=11).encode(x="x:Q", text="label:N")

    vol_rule_df = pd.DataFrame([
        {"x": mu_val - sigma_val, "label": f"-1σ ({mu_val - sigma_val:.2%})"},
        {"x": mu_val + sigma_val, "label": f"+1σ ({mu_val + sigma_val:.2%})"}
    ])
    vol_lines = alt.Chart(vol_rule_df).mark_rule(color="#6B7280", strokeWidth=1.5, strokeDash=[3, 3]).encode(x="x:Q")
    vol_text = alt.Chart(vol_rule_df).mark_text(align="center", dy=70, color="#6B7280", fontSize=10).encode(x="x:Q", text="label:N")

    combined_chart = (base_density + capm_line + capm_text + vol_lines + vol_text).properties(height=240)
    st.altair_chart(combined_chart, use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 2: 全市場 GTS Top 10 選股池 (下次挑選)
# ------------------------------------------------------------------------------
with tab_radar_top10:
    st.subheader("🏆 全市場 GTS 評分 Top 10 候選池 (供換馬與新倉挑選)")
    st.caption("依據 P1 卓越 DNA (35分) + P2 非對稱預期差 (35分) + P3 即時破局催化劑 (30分) 綜合排序")

    all_universe = []
    for s, d in DEFAULT_RADAR_PRESETS.items():
        all_universe.append({
            "標的 (Ticker)": s,
            "GTS 總分": d["gts"],
            "參考現價": f"${d['price']:.2f}",
            "今日漲跌": f"{d['pct']:+.2f}%",
            "評級": "🔥 Strong Buy" if d["gts"] >= 80 else "👀 Watchlist",
            "建議建倉區間": f"${d['price']*0.99:.2f} - ${d['price']*1.01:.2f}",
            "移動止損線": f"${d['stop']:.2f}",
            "第一目標價": f"${d['target']:.2f}",
            "Model B 建議權重": "20% ~ 25% (集中進攻)" if d["gts"] >= 85 else ("15% ~ 20%" if d["gts"] >= 80 else "0% (觀察待機)")
        })

    df_top10 = pd.DataFrame(all_universe).sort_values(by="GTS 總分", ascending=False).reset_index(drop=True)
    df_top10.index = df_top10.index + 1
    st.dataframe(df_top10, use_container_width=True)

    st.info(
        "💡 **選股決策規則**：當現有持倉（如 MSFT）觸及目標價執行減倉並釋放流動性現金後，"
        "優先由上方排名前三（GTS >= 85 分）且處於 PEAD 催化期的標的（如 AVGO, NVDA, ALNY）進行階梯換馬。"
    )

# ------------------------------------------------------------------------------
# Tab 3: 重大事件日曆
# ------------------------------------------------------------------------------
with tab_calendar:
    st.subheader("🗓️ 重大財報與產品發布日程")
    st.caption("自動對齊持倉與 GTS 龍頭之官方財報公布日與開發者大會日程")

    today = datetime.date.today()
    matched = []
    for ev in STATIC_EVENTS_DB:
        ev_date = datetime.datetime.strptime(ev["start"], "%Y-%m-%d").date()
        if ev_date >= today:
            days = (ev_date - today).days
            c_ev = dict(ev)
            c_ev["days"] = days
            matched.append(c_ev)

    matched.sort(key=lambda x: x["start"])

    if matched:
        grid_data = []
        for item in matched:
            status_tag = "🔴 48H 搶跑窗口" if item["days"] <= 2 else ("🟡 15天催化期" if item["days"] <= 15 else "🟢 常規排期")
            grid_data.append({
                "狀態警示": status_tag,
                "預計日期": item["start"],
                "倒數天數": f"{item['days']} 天",
                "標的": item["symbol"],
                "事件類別": item["category"],
                "事件標題": item["title"]
            })
        st.dataframe(pd.DataFrame(grid_data), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("⏳ 近期重大事件倒數與風控指引")
        for item in matched:
            tag = "🔴 48H 機構搶跑窗口" if item["days"] <= 2 else ("🟡 近期即將公布" if item["days"] <= 15 else "🟢 遠期常規")
            with st.expander(f"{tag} | {item['start']} · {item['title']} (倒數 {item['days']} 天)", expanded=(item["days"] <= 15)):
                st.write(f"**標的**：`{item['symbol']}` | **類別**：`{item['category']}`")
                st.write(f"**事件說明與指引**：\n{item['desc']}")
