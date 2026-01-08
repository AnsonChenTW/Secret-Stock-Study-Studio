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
    layout="centered", # 手機版建議用 centered 比較聚焦
    initial_sidebar_state="collapsed" # 預設收起側邊欄，讓主畫面更大
)

st.title("📱 ProTrader 操盤室")
st.caption("AI 驅動・台美股智慧分析")

# ===========================
# 2. 模型自動修復機制
# ===========================

# 讀取 Key
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
    if not llm_available: return "⚠️ 請先設定 API Key"
    
    # 優先嘗試 Flash (快且新)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        # 如果失敗 (404)，改用 Pro (舊版穩定)
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI 分析暫時無法使用 ({str(e)[:20]}...)"

# ===========================
# 3. 核心函數
# ===========================

def get_ticker_symbol(input_str):
    """智慧判斷台美股"""
    input_str = input_str.strip().upper()
    # 如果全是數字 (如 2330)，認定為台股
    if input_str.isdigit():
        return f"{input_str}.TW", "TW"
    # 否則認定為美股 (如 AAPL, TSLA)
    return input_str, "US"

def fetch_data_robust(ticker):
    max_retries = 3
    for i in range(max_retries):
        try:
            time.sleep(random.uniform(0.5, 1.5)) # 隨機休息防封鎖
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
    except:
        return []

def calculate_technical_score(df):
    if len(df) < 60: return 50, "資料不足"
    score = 50
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. 均線趨勢
    if last['MA20'] > last['MA60'] and last['Close'] > last['MA20']:
        score += 25 # 多頭排列
    elif last['Close'] < last['MA60']:
        score -= 25 # 空頭
        
    # 2. 短線支撐
    if last['Close'] > last['MA20']: score += 10
    
    # 3. 量能爆發
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if vol_ma5 > 0 and (last['Volume'] / vol_ma5) > 1.5:
        score += 15
        
    final_score = min(100, max(0, score))
    
    # 簡易趨勢標籤
    if final_score >= 75: trend = "🔥 強力多頭"
    elif final_score >= 60: trend = "📈 偏多震盪"
    elif final_score <= 40: trend = "📉 偏空修正"
    else: trend = "⚖️ 盤整觀望"
    
    return final_score, trend

def analyze_ai_summary(news_list, ticker, trend_tag):
    if not news_list: return "無近期新聞可供分析。"
    
    headlines = [f"- {n.get('title')}" for n in news_list[:5]]
    txt = "\n".join(headlines)
    
    prompt = f"""
    你是一位手機看盤 App 的 AI 助手。
    分析標的：{ticker} (目前技術面狀態：{trend_tag})
    
    請根據新聞標題，給出「手機易讀」的結論 (總字數 100 字內)：
    1. 【一句話結論】：(利多/利空/中性) + 核心原因。
    2. 【操作建議】：(簡短建議，如拉回買進、觀望、停損)。
    
    新聞：
    {txt}
    """
    return get_gemini_response(prompt)

def generate_indicator_report(df, vol_profile):
    if len(df) < 60: return []
    last = df.iloc[-1]
    
    # 產生簡短的手機版指標報告
    bias = ((last['Close'] - last['MA20']) / last['MA20']) * 100
    
    vp_price = vol_profile.idxmax().mid if vol_profile is not None else 0
    vp_status = "支撐" if last['Close'] > vp_price else "壓力"
    
    return [
        {"指標": "季線 (生命線)", "數值": f"{last['MA60']:.1f}", "狀態": "✅ 在之上" if last['Close'] > last['MA60'] else "❌ 跌破"},
        {"指標": "月線乖離", "數值": f"{bias:.1f}%", "狀態": "⚠️ 過熱" if bias > 15 else ("⚡ 超跌" if bias < -15 else "👌 正常")},
        {"指標": "籌碼大量區", "數值": f"{vp_price:.1f}", "狀態": f"{vp_status}"},
    ]

# ===========================
# 4. 手機版 UI 邏輯
# ===========================

# 將輸入框移到最上方，方便手機操作
input_container = st.container()
with input_container:
    # 支援逗號分隔多檔
    raw_input = st.text_input("輸入代號 (自動辨識台美股，支援多檔)", 
                              placeholder="例: 2330, NVDA, 2317", 
                              value="").strip()
    
    start_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

if start_btn and raw_input:
    tickers = [t.strip() for t in raw_input.replace("，", ",").split(",") if t.strip()]
    
    results_for_ranking = []
    
    # 使用 st.status 取代進度條，手機上看更乾淨
    with st.status("🔍 AI 正在掃描市場數據...", expanded=True) as status:
        
        for idx, t_str in enumerate(tickers):
            # 1. 轉換代號 (智慧判斷)
            real_ticker, market_loc = get_ticker_symbol(t_str)
            status.write(f"正在分析 ({idx+1}/{len(tickers)}): **{real_ticker}** ...")
            
            # 2. 抓取數據
            df = fetch_data_robust(real_ticker)
            
            if df is not None:
                # 計算指標
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                
                # 計算籌碼大量區
                try:
                    df_recent = df.tail(120).copy()
                    bins = pd.cut(df_recent['Close'], bins=30)
                    vol_profile = df_recent.groupby(bins, observed=False)['Volume'].sum()
                except: vol_profile = None
                
                # 評分與趨勢
                score, trend_tag = calculate_technical_score(df)
                last_price = df['Close'].iloc[-1]
                change = last_price - df['Close'].iloc[-2]
                change_pct = (change / df['Close'].iloc[-2]) * 100
                
                # 抓新聞與 AI 分析
                news = fetch_news(real_ticker)
                ai_comment = analyze_ai_summary(news, real_ticker, trend_tag)
                
                # === 手機版卡片顯示 (Card View) ===
                st.markdown("---") # 分隔線
                
                # A. 標題區 (大字體)
                col_head1, col_head2 = st.columns([1.5, 1])
                with col_head1:
                    st.markdown(f"### **{t_str.upper()}**")
                    st.caption(f"{market_loc} Market")
                with col_head2:
                    color = "red" if change > 0 else "green" # 台股紅漲綠跌邏輯(可自調)
                    st.markdown(f"<h3 style='color:{color}; text-align:right;'>{last_price:.2f}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color}; text-align:right; margin-top:-15px;'>{change:+.2f} ({change_pct:+.1f}%)</p>", unsafe_allow_html=True)

                # B. 結論區 (最優先顯示)
                st.info(f"**{trend_tag} (評分: {score})**\n\n🤖 **AI 觀點**：\n{ai_comment}")

                # C. 細節區 (Expander 收納)
                with st.expander("📊 點擊查看 K線圖與詳細指標"):
                    # 1. K線圖
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=1), name='MA60'))
                    fig.update_layout(
                        height=300, # 手機版圖表高度縮小
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_rangeslider_visible=False, 
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. 指標表格
                    st.markdown("##### 關鍵指標診斷")
                    report = generate_indicator_report(df, vol_profile)
                    st.table(pd.DataFrame(report)) # 手機上用 table 比 dataframe 更好讀

                # 收集資料做排行
                results_for_ranking.append({
                    "代號": t_str.upper(),
                    "評分": score,
                    "現價": last_price,
                    "趨勢": trend_tag
                })
                
            else:
                st.error(f"❌ 無法讀取 {t_str}")
        
        status.update(label="✅ 所有分析完成！", state="complete", expanded=False)

    # === 最終排行榜 (手機版優化) ===
    if results_for_ranking:
        st.markdown("---")
        st.subheader("🏆 投資潛力排行")
        
        # 排序
        df_rank = pd.DataFrame(results_for_ranking).sort_values("評分", ascending=False).reset_index(drop=True)
        
        # 使用簡單表格顯示，避免手機橫向捲動
        st.table(df_rank[["代號", "評分", "趨勢"]])
        
        top = df_rank.iloc[0]
        st.success(f"💡 **首選建議：{top['代號']}**\n\n目前技術面最強 ({top['趨勢']})，建議優先關注。")

# 頁尾墊高，避免手機操作被底部遮擋
st.write("\n\n")
st.caption("ProTrader Mobile v3.0 | Designed for iPhone")
