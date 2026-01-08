import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
import requests
import time
import random

# ===========================
# 1. 基礎設定
# ===========================

st.set_page_config(page_title="ProTrader 專業操盤室", layout="wide", initial_sidebar_state="expanded")
st.title("🖥️ ProTrader 專業操盤室 (Gemini版)")
st.markdown("---")

# 讀取 Google Gemini Key
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
    # 使用免費且強大的 Gemini 1.5 Flash 模型
    model = genai.GenerativeModel('gemini-1.5-flash')
    llm_available = True
except Exception:
    llm_available = False

if "watch_list" not in st.session_state:
    st.session_state.watch_list = []

# ===========================
# 2. 核心函數
# ===========================

def get_random_agent():
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ]
    return random.choice(agents)

def fetch_data_robust(ticker):
    """強韌型數據抓取"""
    max_retries = 3
    for i in range(max_retries):
        try:
            time.sleep(random.uniform(0.1, 0.5))
            df = yf.download(ticker, period="1y", progress=False)
            
            # 修復 MultiIndex 問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]

            if not df.empty and 'Close' in df.columns:
                return df
        except Exception:
            continue
    return None

def fetch_news_robust(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.news
    except:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_status(market_type):
    ticker = "SPY" if market_type == "美股 (US)" else "0050.TW"
    name = "標普500" if market_type == "美股 (US)" else "台灣50"
    df = fetch_data_robust(ticker)
    
    if df is None or len(df) < 60:
        return name, "數據連線中", "grey"
        
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
    ticker = ticker.upper().strip()
    if market == "台股 (TW)" and not ticker.endswith(".TW") and ticker.isdigit():
        ticker = f"{ticker}.TW"
        
    df = fetch_data_robust(ticker)
    
    if df is None or df.empty or 'Close' not in df.columns:
        return None, None, None, ticker

    try:
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
    except:
        return None, None, None, ticker
    
    try:
        df_recent = df.tail(120).copy()
        if not df_recent.empty:
            bins = pd.cut(df_recent['Close'], bins=30)
            vol_profile = df_recent.groupby(bins, observed=False)['Volume'].sum()
        else:
            vol_profile = None
    except:
        vol_profile = None
        
    news = fetch_news_robust(ticker)
    return df, news, vol_profile, ticker

def calculate_score(df):
    if len(df) < 60: return 50
    score = 50
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if pd.isna(last['MA20']) or pd.isna(last['MA60']): return 50

        # 趨勢
        if last['MA20'] > last['MA60'] and last['Close'] > last['MA20']: score += 25
        elif last['Close'] < last['MA60']: score -= 25
        # 支撐
        if last['Close'] > last['MA20']: score += 10
        # 量能
        vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
        if last['Volume'] > vol_ma5 * 1.5 and last['Close'] > prev['Close']: score += 15
    except:
        pass
    return min(100, max(0, score))

def analyze_ai(news_list):
    """改用 Google Gemini 進行分析"""
    if not news_list or not llm_available:
        return "⚠️ 無法執行 AI 分析 (無新聞或 API Key)"
        
    headlines = [f"- {n.get('title')}" for n in news_list[:5]]
    txt = "\n".join(headlines)
    
    prompt = f"""
    你是一位專業操盤手。請根據以下新聞標題，給出「三句話」總結：
    1. 市場情緒 (偏多/偏空)
    2. 核心原因
    3. 操作建議
    
    新聞標題：
    {txt}
    """
    try:
        # 使用 Gemini 生成內容
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 分析錯誤: {e}"

# ===========================
# 3. UI 操作區
# ===========================

# --- 側邊欄輸入 ---
st.sidebar.header("🔍 股票搜尋")
m_type = st.sidebar.radio("市場", ["美股 (US)", "台股 (TW)"])
t_input = st.sidebar.text_input("輸入代號 (一次一支)", "2330").strip()
btn = st.sidebar.button("開始分析", type="primary")

# --- 主畫面：大盤狀態 ---
name, status, color = get_market_status(m_type)
if color == "green": st.success(f"**{name}**：{status}")
elif color == "red": st.error(f"**{name}**：{status}")
else: st.warning(f"**{name}**：{status}")

# --- 主畫面：個股分析邏輯 ---
if btn and t_input:
    if "," in t_input:
        st.toast("⚠️ 檢測到多個代號，系統將僅分析第一個。", icon="ℹ️")
        t_input = t_input.split(",")[0].strip()

    with st.spinner(f"正在分析 {t_input} (Gemini AI 解讀中)..."):
        df, news, vol, final_t = process_stock_data(t_input, m_type)
        
    if df is not None:
        last = df.iloc[-1]
        score = calculate_score(df)
        
        # 1. 顯示數據
        c1, c2 = st.columns([2, 1])
        with c1:
            st.header(f"{final_t}")
            st.metric("股價", f"{last['Close']:.2f}", f"{(last['Close']-df.iloc[-2]['Close']):.2f}")
        with c2:
            st.write("操盤評分")
            st.progress(score)
            st.caption(f"{score} 分")
            
        # 2. 顯示圖表
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=2), name='MA60'))
        if vol is not None:
            try:
                mp = vol.idxmax().mid
                fig.add_hline(y=mp, line_dash="dot", line_color="red", annotation_text="大量區")
            except: pass
        fig.update_layout(height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. AI 分析 (Gemini)
        if news:
            if llm_available:
                st.info(analyze_ai(news))
            else:
                st.write("📰 最新消息：")
                for n in news[:3]: st.markdown(f"- [{n.get('title')}]({n.get('link')})")

        # 4. 更新排行榜
        new_data = {'Ticker': final_t, 'Score': score, 'Price': float(last['Close'])}
        st.session_state.watch_list = [x for x in st.session_state.watch_list if x['Ticker'] != final_t]
        st.session_state.watch_list.append(new_data)
        
    else:
        st.error(f"❌ 找不到 {t_input} 數據，請確認代號正確。")

# ===========================
# 4. 側邊欄排行榜
# ===========================

if st.session_state.watch_list:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 觀察名單 (已分析)")
    
    rank_df = pd.DataFrame(st.session_state.watch_list).sort_values("Score", ascending=False)
    
    st.sidebar.dataframe(
        rank_df[['Ticker', 'Score', 'Price']], 
        hide_index=True, 
        column_config={
            "Score": st.column_config.ProgressColumn("分數", max_value=100, format="%d"),
            "Price": st.column_config.NumberColumn("現價", format="%.2f")
        },
        use_container_width=True
    )
    
    if st.sidebar.button("清除清單"):
        st.session_state.watch_list = []
        st.rerun()
else:
    st.sidebar.markdown("---")
    st.sidebar.info("尚未分析任何個股。請輸入代號並按「開始分析」。")
