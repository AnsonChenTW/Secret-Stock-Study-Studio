import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai
import requests
import time
import random

# ===========================
# 1. 手機版面設定
# ===========================

st.set_page_config(
    page_title="ProTrader Mobile", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

st.title("📱 ProTrader 操盤室")
st.caption("AI 驅動・台美股智慧分析")

# ===========================
# 2. 常用台股代碼對照表 (解決 yfinance 只有英文名的問題)
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
# 3. 模型自動修復機制
# ===========================

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
    llm_available = True
except:
    llm_available = False

def get_gemini_response(prompt):
    """自動嘗試不同模型，解決 404 問題"""
    if not llm_available: return "⚠️ 請先設定 API Key"
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI 分析暫時無法使用 ({str(e)[:20]}...)"

# ===========================
# 4. 核心函數
# ===========================

def get_ticker_info(input_str):
    """
    智慧判斷台美股，並附加中文名稱
    回傳: (真實代號, 顯示名稱, 市場)
    """
    input_str = input_str.strip().upper()
    
    # 判斷是否為台股 (純數字)
    if input_str.isdigit():
        real_ticker = f"{input_str}.TW"
        # 嘗試從字典找中文名，找不到就用代號
        zh_name = TW_STOCK_NAMES.get(input_str, "")
        display_name = f"{input_str} {zh_name}".strip()
        return real_ticker, display_name, "TW"
    
    # 否則認定為美股
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

def generate_indicator_report(df, vol_profile):
    """
    新手教學版：狀態 + 定義教學
    """
    if len(df) < 60: return []
    last = df.iloc[-1]
    
    # 1. 季線邏輯
    if last['Close'] > last['MA60']:
        ma60_status = "✅ 站上季線｜這條是生命線，股價在上面代表長期趨勢健康，主力還在顧。"
    else:
        ma60_status = "❌ 跌破季線｜生命線失守，代表長期趨勢轉弱，上方有套牢壓力，不宜接刀。"

    # 2. 乖離率邏輯
    bias = ((last['Close'] - last['MA20']) / last['MA20']) * 100
    if bias > 15:
        bias_status = "⚠️ 過熱 (正乖離大)｜漲太兇了，像橡皮筋拉太緊，隨時可能回檔休息，不要追高。"
    elif bias < -15:
        bias_status = "⚡ 超跌 (負乖離大)｜跌太深了，像皮球壓到底，有機會出現反彈。"
    else:
        bias_status = "👌 正常範圍｜股價走勢穩健，沒有暴漲暴跌的風險。"

    # 3. 籌碼大量區邏輯
    vp_price = vol_profile.idxmax().mid if vol_profile is not None else 0
    if last['Close'] > vp_price:
        vp_status = f"🧱 有支撐｜股價在大量區({vp_price:.0f})之上。代表大部分人都賺錢，拉回這裡會有人想買。"
    else:
        vp_status = f"🔨 有壓力｜股價在大量區({vp_price:.0f})之下。代表大部分人都賠錢，漲到這裡會有人想賣。"
    
    return [
        {"指標": "季線 (生命線)", "數值": f"{last['MA60']:.1f}", "診斷與教學": ma60_status},
        {"指標": "月線乖離率", "數值": f"{bias:.1f}%", "診斷與教學": bias_status},
        {"指標": "籌碼大量區", "數值": f"{vp_price:.1f}", "診斷與教學": vp_status},
    ]

# ===========================
# 5. 手機版 UI 邏輯
# ===========================

input_container = st.container()
with input_container:
    raw_input = st.text_input("輸入代號 (自動辨識台美股，支援多檔)", 
                              placeholder="例: 2330, NVDA, 2317", 
                              value="").strip()
    start_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

if start_btn and raw_input:
    tickers = [t.strip() for t in raw_input.replace("，", ",").split(",") if t.strip()]
    results_for_ranking = []
    
    with st.status("🔍 AI 正在掃描市場數據...", expanded=True) as status:
        
        for idx, t_str in enumerate(tickers):
            # 取得代號與中文名
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
                
                # A. 標題區 (顯示中文名)
                c1, c2 = st.columns([1.8, 1])
                with c1:
                    st.markdown(f"### **{display_name}**") # 這裡會顯示 "2330 台積電"
                    st.caption(f"{market_loc} Market")
                with c2:
                    color = "red" if change > 0 else "green"
                    st.markdown(f"<h3 style='color:{color}; text-align:right;'>{last_price:.2f}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color}; text-align:right; margin-top:-15px;'>{change:+.2f} ({change_pct:+.1f}%)</p>", unsafe_allow_html=True)

                # B. 結論區
                st.info(f"**{trend_tag} (評分: {score})**\n\n🤖 **AI 觀點**：\n{ai_comment}")

                # C. 細節區 (移除 Expander，直接顯示)
                st.markdown("##### 📊 K線圖與指標診斷")
                
                # 1. K線圖
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=1), name='MA60'))
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. 指標表格 (新手教學版)
                report = generate_indicator_report(df, vol_profile)
                
                # 使用 dataframe 並設定 column config 讓文字可以換行顯示
                st.dataframe(
                    pd.DataFrame(report),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "指標": st.column_config.TextColumn("指標", width="small"),
                        "數值": st.column_config.TextColumn("數值", width="small"),
                        "診斷與教學": st.column_config.TextColumn("診斷與教學", width="large"),
                    }
                )

                results_for_ranking.append({"代號": display_name, "評分": score, "趨勢": trend_tag})
            else:
                st.error(f"❌ 無法讀取 {display_name}")
        
        status.update(label="✅ 分析完成！", state="complete", expanded=False)

    if results_for_ranking:
        st.markdown("---")
        st.subheader("🏆 綜合排行")
        df_rank = pd.DataFrame(results_for_ranking).sort_values("評分", ascending=False).reset_index(drop=True)
        st.table(df_rank[["代號", "評分", "趨勢"]])

st.write("\n\n")
