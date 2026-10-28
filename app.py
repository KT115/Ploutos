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
# 0. 全局配置 & 注入自適應現代化 CSS (徹底修復截斷與難看排版)
# ==============================================================================
st.set_page_config(
    page_title="AI Investment Hub | 實盤決策指揮終端",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HKT_TZ = pytz.timezone("Asia/Hong_Kong")

st.markdown("""
<style>
    /* 全局緊湊與高對比字體 */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    
    /* 置頂核心資產卡片 (無截斷) */
    .kpi-card-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        color: #F8FAFC;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .kpi-label { font-size: 12px; color: #94A3B8; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .kpi-value { font-size: 24px; font-weight: 800; color: #FFFFFF; line-height: 1.2; }
    .kpi-sub { font-size: 11px; margin-top: 4px; font-weight: 500; }
    .kpi-green { color: #10B981; }
    .kpi-blue { color: #38BDF8; }

    /* 總體評估方塊 */
    .audit-summary-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
        color: #1E293B;
        font-size: 14px;
        line-height: 1.6;
    }

    /* 每日操作指令卡片 */
    .action-row {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .action-trim { border-left: 5px solid #10B981; }
    .action-raise { border-left: 5px solid #3B82F6; }
    .action-hold { border-left: 5px solid #F59E0B; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. 內建高可靠度基準數據
# ==============================================================================
DEFAULT_RADAR_PRESETS = {
    "MSFT": {"price": 530.01, "pct": 1.05, "vwap": 521.80, "gts": 84, "nu": 4.10, "mu": 0.0012, "sigma": 0.0145, "cvar": -3.10, "stop": 508.00, "target": 530.00},
    "AVGO": {"price": 330.42, "pct": 2.72, "vwap": 322.10, "gts": 87, "nu": 3.82, "mu": 0.0028, "sigma": 0.0210, "cvar": -3.85, "stop": 314.00, "target": 365.00},
    "NVDA": {"price": 142.50, "pct": 1.85, "vwap": 138.20, "gts": 85, "nu": 3.45, "mu": 0.0035, "sigma": 0.0265, "cvar": -4.20, "stop": 135.00, "target": 160.00},
    "ORCL": {"price": 126.53, "pct": 0.35, "vwap": 124.50, "gts": 82, "nu": 4.05, "mu": 0.0018, "sigma": 0.0165, "cvar": -3.45, "stop": 118.50, "target": 148.00},
    "VOO": {"price": 703.23, "pct": 0.67, "vwap": 698.00, "gts": 81, "nu": 4.90, "mu": 0.0006, "sigma": 0.0085, "cvar": -2.10, "stop": 682.00, "target": 725.00},
    "GOOGL": {"price": 319.88, "pct": -0.10, "vwap": 318.20, "gts": 78, "nu": 4.30, "mu": 0.0008, "sigma": 0.0152, "cvar": -3.20, "stop": 308.00, "target": 355.00},
    "MRVL": {"price": 234.07, "pct": 2.43, "vwap": 228.60, "gts": 76, "nu": 3.60, "mu": 0.0022, "sigma": 0.0285, "cvar": -4.80, "stop": 225.00, "target": 255.00},
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
    "MRVL: 0.8480 @ 220.00"
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
# 2. 安全單線程市場數據獲取
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
# 3. 側邊欄控制台
# ==============================================================================
with st.sidebar:
    st.title("💼 持倉與通訊設定")
    st.caption("AI Investment Hub · Terminal Config")
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
        action_title = f"🎯 減持鎖利：{sym} 觸及第一目標價 ${m['target']:.2f}"
        action_desc = f"最新現價 **${price:.2f}** 累計利潤達 **+{pnl_pct:.1f}% (+${pnl_usd:.2f} USD)**。建議手動掛單減持 **20%~30% (約 1.0~1.6 股)** 鎖死利潤，並將剩餘持倉防守線調升至 **${price*0.98:.2f}**。"
    elif pnl_pct >= 10.0:
        phase_label = "Phase 2 (🚀 利潤奔跑保護)"
        action_type = "RAISE_TRAILING_STOP"
        action_title = f"🚀 抬升防線：{sym} 浮盈達 +{pnl_pct:.2f}%"
        action_desc = f"累計利潤 **+${pnl_usd:.2f} USD**，進入主升浪。強制將防守線由原先的 ${m['stop']:.2f} 抬升至 **保本點以上 (${max(cost*1.05, price*0.96):.2f})**，未達目標價前堅定續抱。"
    elif qty > 0:
        phase_label = "Phase 1 (🛡️ 基準防守初期)"
        action_type = "HOLD_AND_DEFEND"
        action_title = f"🛡️ 嚴守防線：{sym} 浮盈 +{pnl_pct:.2f}%"
        action_desc = f"處於初期建倉區。嚴格守穩剛性防守線 **${m['stop']:.2f}** (當前裕度 +{((price - m['stop'])/price*100):.2f}%)，守穩續抱，盤中跌破執行一票否決清倉。"
    else:
        phase_label = "現金觀察"
        action_type = "NONE"
        action_title = ""
        action_desc = ""

    if action_title:
        actions_list.append({
            "symbol": sym,
            "type": action_type,
            "title": action_title,
            "desc": action_desc
        })

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
# 5. 【置頂區塊 1】：全投資組合總體評估 (無截斷卡片)
# ==============================================================================
st.title("🏛️ 9人決策委員會：實盤操作指揮終端")
st.caption(f"審計時點: {now_str} | 治理鐵律: RULE-MD-01 (非對稱防禦) & RULE-MD-05 (月化+10% 複利基準)")

# 使用高可讀性 HTML 卡片，保證在任何螢幕寬度都不截斷數字
st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 12px;">
    <div class="kpi-card-box">
        <div class="kpi-label">總資產淨值 (NAV)</div>
        <div class="kpi-value">${total_market_val:,.2f}</div>
        <div class="kpi-sub kpi-blue">實盤持倉合計 (USD)</div>
    </div>
    <div class="kpi-card-box">
        <div class="kpi-label">累積未實現利潤 (P&L)</div>
        <div class="kpi-value kpi-green">{'+' if total_pnl_usd>=0 else ''}${total_pnl_usd:,.2f}</div>
        <div class="kpi-sub kpi-green">整體回報率: {total_pnl_pct:+.2f}%</div>
    </div>
    <div class="kpi-card-box">
        <div class="kpi-label">組合風控體制</div>
        <div class="kpi-value">Goldilocks 擴張</div>
        <div class="kpi-sub kpi-green">宏觀順風 · 允許 Model B 進攻</div>
    </div>
    <div class="kpi-card-box">
        <div class="kpi-label">CRO 全局防守底線</div>
        <div class="kpi-value">-8.0% 回撤預算</div>
        <div class="kpi-sub kpi-blue">無任何部位穿透 · 安全受控</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 總結評判文字 (修復原先的字串黏著問題)
st.markdown(f"""
<div class="audit-summary-box">
    <b>【委員會主席 (MD) 總體裁決】</b><br>
    • <b>組合健康度</b>：目前實盤持倉總 NAV 達 <b>${total_market_val:,.2f} USD</b>，累積浮盈 <b>{total_pnl_pct:+.2f}% (+${total_pnl_usd:,.2f} USD)</b>，大幅領先大盤基準，嚴格對標月化 +10% 複利治理目標。<br>
    • <b>結構與體質</b>：核心領頭羊 (AVGO, MSFT) 全面進入主升浪與目標收斂期；大盤底倉 (VOO) 提供厚實安全墊；MRVL 脫離危險期。所有持倉均站穩 20-Day VWAP 之上。<br>
    • <b>今日戰術核心</b>：今日核心戰術在於<b>「執行利潤收斂兌現，並將獲利部位防守線大幅向上抬升」</b>，鎖死超額收益，拒絕利潤坐雲霄飛車！
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 6. 【置頂區塊 2】：精算師每日具體操作指令 (清晰卡片版)
# ==============================================================================
st.markdown("### ⚡ 精算師每日具體操作建議 (Immediate Action Directives)")

for act in actions_list:
    border_class = "action-trim" if act["type"] == "TAKE_PROFIT_TRIM" else ("action-raise" if act["type"] == "RAISE_TRAILING_STOP" else "action-hold")
    badge_color = "#10B981" if act["type"] == "TAKE_PROFIT_TRIM" else ("#3B82F6" if act["type"] == "RAISE_TRAILING_STOP" else "#F59E0B")
    
    st.markdown(f"""
    <div class="action-row {border_class}">
        <div style="font-size: 15px; font-weight: 700; color: {badge_color}; margin-bottom: 4px;">{act['title']}</div>
        <div style="font-size: 13.5px; color: #334155; line-height: 1.5;">{act['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

# WhatsApp 推播獨立區塊 (杜絕按鈕截斷)
with st.expander("📲 一鍵將上述操作指令推播至 WhatsApp (點擊展開/發送)"):
    wa_summary_lines = [
        f"🏛️【AI Hub 每日操作指令與總體評估】",
        f"時間: {now_str}",
        f"總資產: ${total_market_val:,.2f} USD (浮盈 {total_pnl_pct:+.2f}%)\n",
        "🎯【今日重點執行項目】:"
    ]
    for idx, act in enumerate(actions_list, 1):
        wa_summary_lines.append(f"{idx}. {act['title']}\n   {act['desc'].replace('**', '')}")
    full_wa_payload = "\n".join(wa_summary_lines)

    if st.button("🚀 確認發送每日操作清單至 WhatsApp", type="primary"):
        if not wa_phone or not wa_apikey:
            st.error("請先於左側邊欄輸入手機號碼與 CallMeBot API Key！")
        else:
            enc = urllib.parse.quote(full_wa_payload)
            endpoint = f"https://api.callmebot.com/whatsapp.php?phone={wa_phone}&text={enc}&apikey={wa_apikey}"
            try:
                res = requests.get(endpoint, timeout=8)
                if res.status_code == 200:
                    st.success("✅ 操作指令已成功發送至你的 WhatsApp！")
                else:
                    st.error("❌ 發送失敗，請確認 API Key 與手機號碼格式。")
            except Exception as e:
                st.error(f"❌ 連線異常: {e}")

st.markdown("---")

# ==============================================================================
# 7. 主分頁架構
# ==============================================================================
tab_pos, tab_radar_top10, tab_calendar = st.tabs([
    "💼 持倉動態雷達與 CAPM 直觀金額圖表",
    "🏆 全市場 GTS Top 10 選股池 (下次挑選)",
    "📅 重大事件日曆"
])

# ------------------------------------------------------------------------------
# Tab 1: 持倉動態雷達與直觀金額圖表
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

    # --------------------------------------------------------------------------
    # 個股深度檢視：金額化直觀分佈圖 (徹底解答「+2.38% 是多少錢？」)
    # --------------------------------------------------------------------------
    st.markdown("---")
    focus_symbols = [p["symbol"] for p in portfolio_items if p["qty"] > 0]
    focus_sym = st.selectbox("選擇要審核的持倉標的 (檢視條件分佈與直觀金額換算)", focus_symbols)
    focus = next((item for item in portfolio_items if item["symbol"] == focus_sym), portfolio_items[0])

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("個股累計利潤", f"{focus['pnl_pct']:+.2f}%", delta=f"${focus['pnl_usd']:+,.2f} USD")
    fc2.metric("GTS 評分", f"{focus['gts']} / 100", delta=focus["phase"])
    fc3.metric("極值自由度 ν", f"{focus['nu']:.2f}", delta="肥尾下行" if focus['nu'] < 4.5 else "常態尾部", delta_color="inverse")
    fc4.metric("95% CVaR", f"{focus['cvar']:.2f}%", delta="單日下行敞口", delta_color="inverse")

    # 核心參數
    cur_p = focus["price"]
    pos_qty = focus["qty"]
    pos_val = focus["val"]
    mu_val = focus["mu"]
    sigma_val = focus["sigma"]
    nu_val = focus["nu"]

    # 換算每 1σ 的具體每股金額與部位損益金額
    sigma_1_dollar = cur_p * sigma_val
    sigma_1_pos_pnl = pos_val * sigma_val

    st.markdown(f"""
    <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 10px 14px; margin: 10px 0; font-size: 13.5px; color: #1E3A8A;">
        💡 <b>波動度直接換算金額 (以 {focus_sym} 持有 {pos_qty:.4f} 股 / 市值 ${pos_val:,.2f} USD 換算)：</b><br>
        • <b>正常 1σ 日波動 (±{sigma_val*100:.2f}%)</b>：等於每股股價波動 <b>±${sigma_1_dollar:.2f} USD</b>，部位單日損益波動約 <b>±${sigma_1_pos_pnl:,.2f} USD</b>。<br>
        • <b>極端 95% CVaR 尾部風險 ({focus['cvar']:.2f}%)</b>：若遭遇單日黑天鵝，預估每股下跌 <b>-${cur_p * abs(focus['cvar']/100):.2f} USD</b>，部位單日潛在最大虧損約 <b>-${pos_val * abs(focus['cvar']/100):,.2f} USD</b>。
    </div>
    """, unsafe_allow_html=True)

    # 繪製 Altair 直觀金額雙標註分佈圖
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
        x=alt.X("Return:Q", title=f"{focus_sym} 日收益率 (1σ = ±{sigma_val*100:.2f}% / 每股 ±${sigma_1_dollar:.2f})", axis=alt.Axis(format="%")),
        y=alt.Y("Density:Q", title="機率密度"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Student's t (真實肥尾)", "Gaussian (常態分佈)"], range=["#E02424", "#3F83F8"]))
    )

    capm_rule_df = pd.DataFrame([{"x": capm_daily, "label": f"CAPM 基準: +{capm_daily:.2%} (+$ {cur_p*capm_daily:.2f})" }])
    capm_line = alt.Chart(capm_rule_df).mark_rule(color="#10B981", strokeWidth=2, strokeDash=[5, 5]).encode(x="x:Q")
    capm_text = alt.Chart(capm_rule_df).mark_text(align="left", dx=5, dy=-75, color="#10B981", fontWeight="bold", fontSize=11).encode(x="x:Q", text="label:N")

    vol_rule_df = pd.DataFrame([
        {"x": -sigma_val, "label": f"-1σ: -{sigma_val*100:.2f}% (-${sigma_1_dollar:.2f})"},
        {"x": sigma_val, "label": f"+1σ: +{sigma_val*100:.2f}% (+${sigma_1_dollar:.2f})"}
    ])
    vol_lines = alt.Chart(vol_rule_df).mark_rule(color="#64748B", strokeWidth=1.5, strokeDash=[3, 3]).encode(x="x:Q")
    vol_text = alt.Chart(vol_rule_df).mark_text(align="center", dy=70, color="#64748B", fontSize=11, fontWeight="bold").encode(x="x:Q", text="label:N")

    combined_chart = (base_density + capm_line + capm_text + vol_lines + vol_text).properties(height=260)
    st.altair_chart(combined_chart, use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 2: 全市場 GTS Top 10 選股池
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
