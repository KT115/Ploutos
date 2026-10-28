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
    page_title="AI Investment Hub | Actuarial Portfolio & Radar",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HKT_TZ = pytz.timezone("Asia/Hong_Kong")

# ==============================================================================
# 1. 內建基準數據與全市場美股候選庫 (GTS Top 10 挑選池)
# ==============================================================================
DEFAULT_RADAR_PRESETS = {
    "MSFT": {"price": 530.01, "pct": 1.05, "vwap": 521.80, "gts": 84, "nu": 4.10, "mu": 0.0012, "sigma": 0.0145, "cvar": -3.10, "stop": 508.00, "target": 550.00},
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

# 預設真實部位資料 (可由用戶在側邊欄自由修改)
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
    st.title("💼 持倉與系統設定")
    st.caption("AI Investment Hub · Real-time Manager")
    st.markdown("---")

    st.subheader("📝 買入價與持股設定 (Portfolio Config)")
    st.caption("格式：`代碼: 股數 @ 成本價`")
    portfolio_text = st.text_area(
        "編輯持倉清單",
        value=INITIAL_PORTFOLIO_CONFIG,
        height=180
    )

    # 解析持倉輸入
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
    st.subheader("📲 WhatsApp 警報設定")
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

# 計算總投組數據 (Portfolio Total Overview)
total_market_val = 0.0
total_cost_val = 0.0
portfolio_items = []

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

    # 動態 Phase 生命週期判定
    # Phase 1: 試探/成本期 (|PnL| < 10% 且未達目標)
    # Phase 2: 利潤奔跑期 (PnL >= +10% 且未達目標價)
    # Phase 3: 到價收斂/鎖利期 (價格 >= 第一目標價)
    if price >= m["target"]:
        phase_label = "Phase 3 (🎯 觸及目標收斂)"
    elif pnl_pct >= 10.0:
        phase_label = "Phase 2 (🚀 利潤奔跑保護)"
    else:
        phase_label = "Phase 1 (🛡️ 基準防守初期)"

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

# ==============================================================================
# 5. 主介面渲染
# ==============================================================================
st.title("🏛️ 美股量化決策終端與投資組合中樞")
now_str = datetime.datetime.now(HKT_TZ).strftime("%Y-%m-%d %H:%M:%S HKT")
st.caption(f"系統時間: {now_str} | 決策框架: 9人委員會 · Model B & 精算極值體系")

# ==============================================================================
# 6. 置頂：Portfolio Overview Dashboard (整體持倉看板)
# ==============================================================================
with st.container():
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("持倉總市值 (NAV)", f"${total_market_val:,.2f} USD", delta="實盤持倉合計")
    k2.metric("總未實現利潤 (P&L $)", f"{'+' if total_pnl_usd>=0 else ''}${total_pnl_usd:,.2f} USD")
    k3.metric("整體報酬率 (ROI %)", f"{total_pnl_pct:+.2f}%", delta="對標月化 +10% 複利")
    k4.metric("活躍持倉數", f"{len([x for x in portfolio_items if x['qty'] > 0])} 檔核心", delta="Model B 集中進攻")

st.markdown("---")

# ==============================================================================
# 7. 主分頁架構
# ==============================================================================
tab_pos, tab_radar_top10, tab_calendar = st.tabs([
    "💼 持倉動態雷達與 P&L",
    "🏆 全市場 GTS Top 10 選股池",
    "📅 重大事件日曆"
])

# ------------------------------------------------------------------------------
# Tab 1: 持倉動態雷達大表 (含買入價、P&L、Phase 與 Immediate Action)
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

    st.dataframe(pd.DataFrame(pos_table), hide_index=True)

    # --------------------------------------------------------------------------
    # 個股深度檢視與動態分佈圖
    # --------------------------------------------------------------------------
    st.markdown("---")
    focus_symbols = [p["symbol"] for p in portfolio_items if p["qty"] > 0]
    if not focus_symbols:
        focus_symbols = list(holdings_dict.keys())

    focus_sym = st.selectbox("選擇要深度審核的持倉標的", focus_symbols)
    focus = next((item for item in portfolio_items if item["symbol"] == focus_sym), portfolio_items[0])

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("個股累計利潤", f"{focus['pnl_pct']:+.2f}%", delta=f"${focus['pnl_usd']:+,.2f} USD")
    fc2.metric("GTS 評分", f"{focus['gts']} / 100", delta=focus["phase"])
    fc3.metric("極值自由度 ν", f"{focus['nu']:.2f}", delta="肥尾風險" if focus['nu'] < 4.5 else "常態分佈", delta_color="inverse")
    fc4.metric("95% CVaR", f"{focus['cvar']:.2f}%", delta="左尾下行敞口", delta_color="inverse")

    # 動態收益率分佈圖 (每隻股票均有其專屬的 mu 與 sigma)
    mu_val = focus["mu"]
    sigma_val = focus["sigma"]
    nu_val = focus["nu"]

    x_min = mu_val - 3.5 * sigma_val
    x_max = mu_val + 3.5 * sigma_val
    x_pts = np.linspace(x_min, x_max, 100)
    pdf_t = stats.t.pdf(x_pts, df=nu_val, loc=mu_val, scale=sigma_val)
    pdf_n = stats.norm.pdf(x_pts, loc=mu_val, scale=sigma_val)

    chart_df = pd.DataFrame({
        "Return": np.concatenate([x_pts, x_pts]),
        "Density": np.concatenate([pdf_t, pdf_n]),
        "Model": ["Student's t (真實肥尾)" for _ in range(len(x_pts))] + ["Gaussian (常態分佈)" for _ in range(len(x_pts))]
    })

    curve_chart = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("Return:Q", title=f"{focus_sym} 條件日收益率區間 (μ={mu_val:.3%}, σ={sigma_val:.3%})", axis=alt.Axis(format="%")),
        y=alt.Y("Density:Q", title="機率密度"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Student's t (真實肥尾)", "Gaussian (常態分佈)"], range=["#E02424", "#3F83F8"]))
    ).properties(height=230)
    st.altair_chart(curve_chart)

    # --------------------------------------------------------------------------
    # 精算師 Immediate Action Session (依據數據與精算思維裁決)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"⚡ 精算師即時行動裁決 (Immediate Action Session: {focus_sym})")
    
    col_act, col_send = st.columns([1.3, 1])

    # 精算師邏輯判定：
    # 1. 觸及第一目標價 (Phase 3)：建議鎖利減持 20%~30% 部位，防守線抬升至保本價上方
    # 2. 獲利 > 10% (Phase 2)：啟動階梯止盈防禦，將防線拉升至成本價以上
    # 3. 初始建倉 (Phase 1)：嚴守 -4.5% 動態止損
    p_price = focus["price"]
    p_cost = focus["cost"]
    p_target = focus["target"]
    p_pnl = focus["pnl_pct"]

    if p_price >= p_target:
        action_title = "🎯 【觸及目標價 · 鎖利減持與保本抬升指令】"
        action_detail = (
            f"1. **落袋為安建議**：最新價 `${p_price:.2f}` 已精準收斂至預設第一目標價 `${p_target:.2f}`。\n"
            f"2. **精算鎖利減倉**：依據期望效用最大化原則，建議手動掛單減持 **20% ~ 30%** 股數，將近 **{p_pnl:+.1f}%** 超額收益兌現為流動性現金。\n"
            f"3. **移動防守線強制抬升**：剩餘部位防守線由 `${focus['stop']:.2f}` 強制調升至 **${p_price * 0.98:.2f}**，下一個擴展目標價上看 **${p_price * 1.10:.2f}**。"
        )
        box_type = st.success
    elif p_pnl >= 10.0:
        action_title = "🚀 【利潤奔跑中 · 階梯抬升防守線指令】"
        action_detail = (
            f"1. **利潤安全墊充足**：當前浮盈 **{p_pnl:+.2f}%**，已進入 Phase 2 主升浪。\n"
            f"2. **鎖死獲利**：建議將原止損線 `${focus['stop']:.2f}` 抬升至 **保本點以上 (${max(p_cost * 1.05, p_price * 0.96):.2f})**，確保本部位絕不轉盈為虧。\n"
            f"3. **續抱指令**：未觸及第一目標價 `${p_target:.2f}` 前堅定續抱，享受複利擴張。"
        )
        box_type = st.info
    else:
        action_title = "🛡️ 【基準防禦期 · 嚴守下行止損線】"
        action_detail = (
            f"1. **部位狀態**：當前處於 Phase 1 建倉成本區 (浮盈 {p_pnl:+.2f}%)。\n"
            f"2. **CRO 剛性防守線**：嚴格鎖定於 **${focus['stop']:.2f}** (當前裕度 +{((p_price - focus['stop'])/p_price*100):.2f}%)。\n"
            f"3. **執行紀律**：未跌破防線前維持部位，若盤中跌破防線，CRO 將行使一票否決權執行階梯撤退。"
        )
        box_type = st.warning

    with col_act:
        box_type(f"### {action_title}\n{action_detail}")

    with col_send:
        st.markdown("**📲 推播此行動指令至 WhatsApp**")
        st.write("將當前持倉盈虧、Phase 狀態與精算師操作指令即時同步到你的手機。")

        action_msg = (
            f"🏛️【AI Hub 精算師行動指令】\n\n"
            f"• 標的：{focus_sym}\n"
            f"• 成本：${focus['cost']:.2f} | 現價：${focus['price']:.2f}\n"
            f"• 浮動盈虧：{focus['pnl_pct']:+.2f}% (${focus['pnl_usd']:+,.2f})\n"
            f"• 生命週期：{focus['phase']}\n"
            f"• 第一目標價：${focus['target']:.2f}\n"
            f"• 動態防守線：${focus['stop']:.2f}\n"
            f"• 裁決行動：{action_title}\n"
            f"• 時間：{now_str}"
        )

        if st.button(f"📤 發送 {focus_sym} 精算行動報告", type="primary", use_container_width=True):
            if not wa_phone or not wa_apikey:
                st.error("請先於側邊欄輸入手機號碼與 CallMeBot API Key！")
            else:
                encoded_msg = urllib.parse.quote(action_msg)
                url = f"https://api.callmebot.com/whatsapp.php?phone={wa_phone}&text={encoded_msg}&apikey={wa_apikey}"
                try:
                    res = requests.get(url, timeout=8)
                    if res.status_code == 200:
                        st.success("✅ 行動報告已成功發送至 WhatsApp！")
                    else:
                        st.error("❌ 發送失敗，請確認 API Key 與號碼。")
                except Exception as e:
                    st.error(f"❌ 連線異常: {e}")

# ------------------------------------------------------------------------------
# Tab 2: 全市場 GTS Top 10 選股池 (供下次換馬與建倉)
# ------------------------------------------------------------------------------
with tab_radar_top10:
    st.subheader("🏆 全市場 GTS 評分 Top 10 標的名單 (供下次挑選)")
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

    # 按 GTS 總分降冪排列
    df_top10 = pd.DataFrame(all_universe).sort_values(by="GTS 總分", ascending=False).reset_index(drop=True)
    df_top10.index = df_top10.index + 1  # 排名從 1 開始

    st.dataframe(df_top10, use_container_width=True)

    st.info(
        "💡 **挑選準則 (RULE-MD-07 / RULE-MD-10)**：\n"
        "- **GTS >= 85 分 (如 AVGO, NVDA, ALNY)**：符合 Model B 核心進攻標的，允許重倉 20%~25% 並鎖定機構 PEAD 催化浪潮。\n"
        "- **80 <= GTS < 85 分**：優質換馬備選，適合在核心持倉達標平倉後作為次階輪動資金池。"
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
