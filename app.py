import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import openai
import requests
import time
import random

# ===========================
# 1. 基礎設定與 API 初始化
# ===========================

st.set_page_config(page_title="ProTrader 專業操盤室", layout="wide", initial_sidebar_state="expanded")
st.title("🖥️ ProTrader 專業操盤室 (Robust Ver.)")
st.markdown("---")

# 讀取 OpenAI Key
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=openai_api_key)
    llm_available = True
except Exception:
    # 這裡不顯示錯誤，只標記無法使用，避免干擾主畫面
    llm_available = False

if "watch_list" not in st.session_state:
    st.session_state.watch_list = []

# ===========================
# 2. 抗封鎖核心函數 (Plan B)
# ===========================

def get_random_agent():
    """隨機產生 User-Agent 以偽裝成不同裝置"""
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ]
    return random.choice(agents)

def fetch_data_robust(ticker):
    """
    強韌型數據抓取：包含重試機制
    使用 yf.download 替代 history，對抗封鎖能力較強
    """
    max_retries = 3
    for i in range(max_retries):
        try:
            # 隨機延遲，模擬人類行為
            time.sleep(random.uniform(0.5, 1.5))
            
            # 使用 yf.download (通常比 Ticker.history 穩定)
            # progress=False 關閉進度條以避免 Streamlit 報錯
            df = yf.download(ticker, period="1y", progress=False, multi_level_index=False)
            
            if not df.empty:
                return df
        except Exception as e:
            if i == max_retries - 1: # 最後一次嘗試也失敗
                print(f"Failed to fetch {ticker}: {e}")
                return None
            continue # 失敗則重試
    return None

def fetch_news_robust(ticker):
    """獨立抓取新聞，失敗不影響股價顯示"""
    try:
        t = yf.Ticker(ticker)
        return t.news
    except:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_status(market_type):
    """大盤紅綠燈"""
    ticker = "SPY" if market_type == "美股 (US)" else "0050.TW"
    name = "標普500" if market_type == "美股 (US)" else "台灣50"
    
    df = fetch_data_robust(ticker)
    
    if df is None or len(df) < 60:
        return name, "數據連線中斷", "grey"
        
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    last = df.iloc[-1]
    
    if last['Close'] > last['MA60'] and last['MA20'] > last['MA60']:
        return name, "多頭格局 (安全)", "green"
    elif last['Close'] < last['MA60']:
        return name, "空頭修正 (危險)", "red"
    else:
        return name, "震盪整理 (觀望)", "yellow"

@st.cache_data(ttl=900, show_spinner=False)
def process_stock_data(ticker, market):
    """處理個股數據與指標"""
    ticker = ticker.upper().strip()
    # 台股後綴處理
    if market == "台股 (TW)" and not ticker.endswith(".TW") and ticker.isdigit():
        ticker = f"{ticker}.TW"
        
    # 1. 抓取股價 (優先)
    df = fetch_data_robust(ticker)
    if df is None or df.empty:
        return None, None, None, ticker

    # 2. 計算指標
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # 3. 計算大量區 (Volume Profile)
    try:
        df_recent = df.tail(120).copy()
        if not df_recent.empty:
            bins = pd.cut(df_recent['Close'], bins=30)
            vol_profile = df_recent.groupby(bins, observed=False)['Volume'].sum()
        else:
            vol_profile = None
    except:
        vol_profile = None
        
    # 4. 抓取新聞 (獨立抓取，失敗回傳空陣列)
    news = fetch_news_robust(ticker)
    
    return df, news, vol_profile, ticker

def calculate_score(df):
    """計算操盤分數"""
    if len(df) < 60: return 50
    score = 50
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 趨勢
    if last['MA20'] > last['MA60'] and last['Close'] > last['MA20']: score += 25
    elif last['Close'] < last['MA60']: score -= 25
    
    # 短線支撐
    if last['Close'] > last['MA20']: score += 10
    
    # 量能
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if last['Volume'] > vol_ma5 * 1.5 and last['Close'] > prev['Close']:
        score += 15
        
    return min(100, max(0, score))

def analyze_ai(news_list):
    """OpenAI 新聞分析"""
    if not news_list or not llm_available:
        return "⚠️ 無法執行 AI 分析 (無新聞資料或 API Key)"
        
    headlines = [f"- {n.get('title')}" for n in news_list[:5]]
    txt = "\n".join(headlines)
    
    prompt = f"""
    你是一位專業操盤手。請根據以下新聞標題，給出「三句話」總結：
    1. 市場情緒 (偏多/偏空)
    2. 核心原因
    3. 操作建議
    
    新聞：
    {txt}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI 分析錯誤: {e}"

# ===========================
# 3. UI 介面
# ===========================

# 側邊欄
st.sidebar.header("🔍 股票搜尋")
m_type = st.sidebar.radio("市場", ["美股 (US)", "台股 (TW)"])
t_input = st.sidebar.text_input("輸入代號", "2330").strip()
btn = st.sidebar.button("開始分析", type="primary")

# 排行榜
if st.session_state.watch_list:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 觀察名單")
    rank_df = pd.DataFrame(st.session_state.watch_list).sort_values("Score", ascending=False)
    st.sidebar.dataframe(
        rank_df[['Ticker', 'Score', 'Price']], 
        hide_index=True, 
        column_config={"Score": st.column_config.ProgressColumn("分數", max_value=100)}
    )

# 主畫面 - 大盤
name, status, color = get_market_status(m_type)
if color == "green": st.success(f"**{name}**：{status}")
elif color == "red": st.error(f"**{name}**：{status}")
else: st.warning(f"**{name}**：{status}")

# 主畫面 - 個股
if btn and t_input:
    with st.spinner("🔄 數據連線中 (正在對抗封鎖機制)..."):
        df, news, vol, final_t = process_stock_data(t_input, m_type)
        
    if df is not None:
        last = df.iloc[-1]
        score = calculate_score(df)
        
        # 顯示頭部資訊
        c1, c2 = st.columns([2, 1])
        with c1:
            st.header(f"{final_t}")
            st.metric("股價", f"{last['Close']:.2f}", f"{(last['Close']-df.iloc[-2]['Close']):.2f}")
        with c2:
            st.write("操盤評分")
            st.progress(score)
            st.caption(f"{score} 分")
            
        # 畫圖
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=2), name='MA60'))
        
        # 大量區線
        if vol is not None:
            mp = vol.idxmax().mid
            fig.add_hline(y=mp, line_dash="dot", line_color="red", annotation_text="大量支撐區")
            
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # 下方分析
        t1, t2 = st.tabs(["數據解讀", "AI 新聞分析"])
        with t1:
            col_a, col_b = st.columns(2)
            bias = ((last['Close'] - last['MA20']) / last['MA20']) * 100
            col_a.info(f"季線成本: {last['MA60']:.2f} (判斷多空分界)")
            col_b.warning(f"月線乖離: {bias:.2f}% (過大留意拉回)")
            
        with t2:
            if news:
                if llm_available:
                    st.success(analyze_ai(news))
                for n in news[:3]:
                    st.markdown(f"- [{n.get('title')}]({n.get('link')})")
            else:
                st.info("⚠️ 本次查詢未抓取到新聞 (可能被 Yahoo 暫時阻擋)，但股價數據正常。")

        # 更新清單
        new_data = {'Ticker': final_t, 'Score': score, 'Price': float(last['Close'])}
        st.session_state.watch_list = [x for x in st.session_state.watch_list if x['Ticker'] != final_t]
        st.session_state.watch_list.append(new_data)
        
    else:
        st.error(f"❌ 無法獲取 {t_input} 數據。Yahoo 伺服器忙碌中，請稍等 1 分鐘後再試。")
