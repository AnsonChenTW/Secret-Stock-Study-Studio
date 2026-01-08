import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
import time
import random

# ===========================
# 1. 頁面設定 (必須在所有 st 指令之前)
# ===========================
st.set_page_config(
    page_title="ProTrader Mobile", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# ===========================
# 2. 系統診斷 (檢查套件版本)
# ===========================
try:
    # 顯示目前安裝的 Google AI 版本，確認是否更新成功
    version_info = genai.__version__
    if version_info < "0.7.0":
        st.error(f"⚠️ 系統偵測到舊版套件 ({version_info})。AI 功能將無法使用。請在 requirements.txt 指定 google-generativeai>=0.7.0 並重啟 App。")
    else:
        # 如果版本正確，顯示一個小小的成功提示 (可隨時移除)
        st.toast(f"✅ 系統環境正常 (GenAI v{version_info})", icon="🤖")
except Exception as e:
    st.warning(f"無法檢測套件版本: {e}")

st.title("📱 ProTrader 操盤室")
st.caption("AI 驅動・台美股智慧分析")

# ===========================
# 3. 常用台股代碼對照表
# ===========================
TW_STOCK_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2303": "聯電", "2308": "台達電",
    "2881": "富邦金", "2882": "國泰金", "2891": "中信金", "2886": "兆豐金", "2884": "玉山金",
    "2603": "長榮", "2609": "陽明", "2615": "萬海", "2618": "長榮航", "2610": "華航",
    "3008": "大立光", "3034": "聯詠", "3037": "欣興", "3045": "台灣大", "2412": "中華電",
    "2912": "統一超", "1216": "統一", "2002": "中鋼", "1101": "台泥", "1102": "亞泥",
    "3231": "緯創", "2382": "廣達", "2376": "技嘉", "2356": "英業達", "6669": "緯穎",
    "2324": "仁寶", "2357": "華碩", "2301": "光寶科", "2344": "華邦電", "2409": "友達",
    "3481": "群創", "2395": "研華", "5871": "中租-KY", "9910": "豐泰", "9921": "巨大"
}

# ===========================
# 4. AI 模型設定 (自動切換修復 404)
# ===========================
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
    llm_available = True
except:
    llm_available = False

def get_gemini_response(prompt):
    """
    自動嘗試不同模型，解決 404 問題
    """
    if not llm_available: return "⚠️ 請先設定 Google API Key"
    
    # 優先順序：Flash (快) -> Pro 1.5 (強) -> Pro (舊版穩定)
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue 
            
    return "⚠️ AI 暫時無法連線 (可能是 API Key 額度或地區限制)"

# ===========================
# 5. 核心數據函數
# ===========================

def get_ticker_info(input_str):
    input_str = input_str.strip().upper()
    if input_str.isdigit():
        real_ticker = f"{input_str}.TW"
        zh_name = TW_STOCK_NAMES.get(input_str, "")
        display_name = f"{input_str} {zh_name}".strip()
        return real_ticker, display_name, "TW"
    return input_str, input_str, "US"

def fetch_data_robust(ticker):
    max_retries = 3
    for i in range(max_retries):
        try:
            time.sleep(random.uniform(0.5, 1.0))
            df = yf.download(ticker, period="1y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if not df.empty and 'Close' in df.columns:
                return df
        except:
            continue
    return None

def fetch_news(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.news
    except: return []

def calculate_technical_score(df):
    if len(df) < 60: return 50, "資料不足"
    score = 50
    last = df.iloc[-1]
    
    # 1. 均線趨勢
    if last['MA20'] > last['MA60'] and last['Close'] > last['MA20']: score += 25
    elif last['Close'] < last['MA60']: score -= 25
    # 2. 短線支撐
    if last['Close'] > last['MA20']: score += 10
    # 3. 量能
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if vol_ma5 > 0 and (last['Volume'] / vol_ma5) > 1.5: score += 15
        
    final_score = min(100, max(0, score))
    
    if final_score >= 75: trend = "🔥 強力多頭"
    elif final_score >= 60: trend = "📈 偏多震盪"
    elif final_score <= 40: trend = "📉 偏空修正"
    else: trend = "⚖️ 盤整觀望"
    
    return final_score, trend

def analyze_ai_summary(news_list, ticker, trend_tag):
    if not news_list: return "無近期新聞。"
    headlines = [f"- {n.get('title')}" for n in news_list[:5]]
    txt = "\n".join(headlines)
    prompt = f"""
    你是手機看盤 App 的 AI 助手。標的：{ticker} (技術面：{trend_tag})
    請根據新聞標題給出「手機易讀」結論 (100字內)：
    1. 【一句話結論】：(利多/利空) + 原因。
    2. 【操作建議】：(拉回買/觀望/停損)。
    新聞：{txt}
    """
    return get_gemini_response(prompt)

def render_indicator_card(title, value, status, explanation):
    """
    使用 HTML/CSS 渲染卡片，確保手機上文字自動換行且易讀
    """
    # 根據狀態決定顏色
    if "✅" in status or "👌" in status or "🧱" in status:
        border_color = "#4CAF50" # Green
    elif "❌" in status or "⚠️" in status or "🔨" in status or "⚡" in status:
        border_color = "#FF5252" # Red
    else:
        border_color = "#FFC107" # Yellow/Orange

    st.markdown(f"""
    <div style="
        background-color: #262730; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 12px; 
        border-left: 5px solid {border_color};
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span style="font-size: 1.1em; font-weight: bold; color: #fafafa;">{title}</span>
            <span style="font-size: 1.1em; font-weight: bold; color: {border_color};">{value}</span>
        </div>
        <div style="font-weight: bold; color: {border_color}; margin-bottom: 5px;">{status}</div>
        <div style="font-size: 0.9em; color: #dddddd; line-height: 1.5;">💡 {explanation}</div>
    </div>
    """, unsafe_allow_html=True)

def generate_educational_report(df, vol_profile):
    """
    生成「帶入數值」的白話文教學
    """
    if len(df) < 60: return
    last = df.iloc[-1]
    price = last['Close']
    ma60 = last['MA60']
    
    # 1. 季線教學
    if price > ma60:
        status_ma = "✅ 站上季線 (多頭)"
        desc_ma = f"目前股價 {price:.1f} 高於季線 {ma60:.1f}。季線是長期的生命線，股價穩在上面，代表長期趨勢健康，主力還在顧。"
    else:
        status_ma = "❌ 跌破季線 (空頭)"
        desc_ma = f"目前股價 {price:.1f} 低於季線 {ma60:.1f}。季線變成上方的「蓋頭反壓」，代表過去一季買的人都賠錢，容易有賣壓。"
    render_indicator_card("季線 (生命線)", f"{ma60:.1f}", status_ma, desc_ma)

    # 2. 乖離率教學
    bias = ((price - last['MA20']) / last['MA20']) * 100
    if bias > 15:
        status_bias = "⚠️ 過熱 (正乖離大)"
        desc_bias = f"目前乖離率 {bias:.1f}%，超過 +15%。股價衝太快了，像橡皮筋拉太緊，隨時可能回檔，千萬別追高。"
    elif bias < -15:
        status_bias = "⚡ 超跌 (負乖離大)"
        desc_bias = f"目前乖離率 {bias:.1f}%，低於 -15%。股價跌太深了，像皮球壓到底，短線有機會反彈。"
    else:
        status_bias = "👌 正常範圍"
        desc_bias = f"目前乖離率 {bias:.1f}%，位於安全區間。股價走勢穩健，沒有失控暴漲或暴跌。"
    render_indicator_card("月線乖離率", f"{bias:.1f}%", status_bias, desc_bias)

    # 3. 籌碼教學
    vp_price = vol_profile.idxmax().mid if vol_profile is not None else 0
    if price > vp_price:
        status_vp = "🧱 下檔有支撐"
        desc_vp = f"股價({price:.1f}) 在大量成交區({vp_price:.0f}) 之上。這個價位是地板，跌回來會有人想買，形成防守。"
    else:
        status_vp = "🔨 上檔有壓力"
        desc_vp = f"股價({price:.1f}) 在大量成交區({vp_price:.0f}) 之下。這個價位是天花板，漲上去會遇到解套賣壓，難以突破。"
    render_indicator_card("籌碼大量區", f"{vp_price:.1f}", status_vp, desc_vp)

# ===========================
# 6. UI 主畫面
# ===========================

input_container = st.container()
with input_container:
    raw_input = st.text_input("輸入代號 (支援多檔，如: 2330, NVDA)", value="").strip()
    start_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

if start_btn and raw_input:
    tickers = [t.strip() for t in raw_input.replace("，", ",").split(",") if t.strip()]
    results_for_ranking = []
    
    with st.status("🔍 AI 正在掃描市場數據...", expanded=True) as status:
        
        for idx, t_str in enumerate(tickers):
            real_ticker, display_name, market_loc = get_ticker_info(t_str)
            status.write(f"正在分析 ({idx+1}/{len(tickers)}): **{display_name}** ...")
            
            df = fetch_data_robust(real_ticker)
            
            if df is not None:
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                try:
                    df_recent = df.tail(120).copy()
                    bins = pd.cut(df_recent['Close'], bins=30)
                    vol_profile = df_recent.groupby(bins, observed=False)['Volume'].sum()
                except: vol_profile = None
                
                score, trend_tag = calculate_technical_score(df)
                last_price = df['Close'].iloc[-1]
                change = last_price - df['Close'].iloc[-2]
                change_pct = (change / df['Close'].iloc[-2]) * 100
                
                news = fetch_news(real_ticker)
                ai_comment = analyze_ai_summary(news, display_name, trend_tag)
                
                # === 卡片顯示區 ===
                st.markdown("---")
                
                # A. 標題與價格
                c1, c2 = st.columns([1.8, 1])
                with c1:
                    st.markdown(f"### **{display_name}**")
                    st.caption(f"{market_loc} Market")
                with c2:
                    color = "#FF5252" if change > 0 else "#4CAF50" # 台股紅漲綠跌
                    st.markdown(f"<h3 style='color:{color}; text-align:right;'>{last_price:.2f}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color}; text-align:right; margin-top:-15px;'>{change:+.2f} ({change_pct:+.1f}%)</p>", unsafe_allow_html=True)

                # B. 結論
                st.info(f"**{trend_tag} (評分: {score})**\n\n🤖 **AI 觀點**：\n{ai_comment}")

                # C. K線圖
                st.markdown("##### 📊 K線結構")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=1), name='MA60'))
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # D. 新手教學診斷 (卡片式)
                st.markdown("##### 🩺 關鍵指標診斷書")
                generate_educational_report(df, vol_profile)

                results_for_ranking.append({"代號": display_name, "評分": score, "趨勢": trend_tag})
            else:
                st.error(f"❌ 無法讀取 {display_name}")
        
        status.update(label="✅ 分析完成！", state="complete", expanded=False)

    if results_for_ranking:
        st.markdown("---")
        st.subheader("🏆 綜合排行")
        # 手機上用 table 呈現簡單排行，避免複雜
        df_rank = pd.DataFrame(results_for_ranking).sort_values("評分", ascending=False).reset_index(drop=True)
        st.table(df_rank[["代號", "評分", "趨勢"]])

st.write("\n\n")
