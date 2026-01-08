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
st.title("🖥️ ProTrader 專業操盤室 (批次慢速版)")
st.markdown("---")

# 讀取 Google Gemini Key
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
    # 使用 Flash 模型速度較快且省額度
    model = genai.GenerativeModel('gemini-1.5-flash')
    llm_available = True
except Exception:
    llm_available = False

if "watch_list" not in st.session_state:
    st.session_state.watch_list = []

# ===========================
# 2. 核心函數 (抗封鎖強化版)
# ===========================

def fetch_data_robust(ticker):
    """強韌型數據抓取"""
    max_retries = 3
    for i in range(max_retries):
        try:
            # 每次抓取前隨機休息，模擬人類行為
            time.sleep(random.uniform(1.0, 3.0))
            
            df = yf.download(ticker, period="1y", progress=False)
            
            # 修復 MultiIndex 問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]

            if not df.empty and 'Close' in df.columns:
                return df
        except Exception as e:
            print(f"Retrying {ticker}... ({e})")
            continue
    return None

def fetch_news_robust(ticker):
    try:
        # 抓新聞前也稍微休息一下
        time.sleep(random.uniform(0.5, 1.0))
        t = yf.Ticker(ticker)
        return t.news
    except:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_status(market_type):
    ticker = "SPY" if market_type == "美股 (US)" else "0050.TW"
    name = "標普500" if market_type == "美股 (US)" else "台灣50"
    
    # 大盤是獨立請求，不要影響到下面的個股 loop，所以這裡不需加太長延遲
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

def process_stock_data(ticker, market):
    """處理單一個股數據"""
    ticker = ticker.upper().strip()
    # 台股後綴處理
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

def analyze_ai(news_list, ticker):
    """Gemini 分析 (加入 Ticker 參數讓回答更精確)"""
    if not news_list or not llm_available:
        return "⚠️ 無法執行 AI 分析 (無新聞或 API Key)"
        
    headlines = [f"- {n.get('title')}" for n in news_list[:5]]
    txt = "\n".join(headlines)
    
    prompt = f"""
    你是一位專業操盤手。請根據以下 {ticker} 的新聞標題，給出「三句話」總結：
    1. 市場情緒 (偏多/偏空)
    2. 核心原因
    3. 操作建議
    
    新聞標題：
    {txt}
    """
    try:
        # 呼叫 AI 前也稍微休息，避免觸發 Gemini 的 RPM 限制
        time.sleep(1)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 分析錯誤: {e}"

def generate_indicator_report(df, vol_profile):
    if len(df) < 60: return []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 指標邏輯 (簡化版)
    ma20_status = "✅ 股價在月線之上" if last['Close'] > last['MA20'] else "🔻 股價跌破月線"
    ma60_status = "✅ 股價在季線之上" if last['Close'] > last['MA60'] else "🔻 股價跌破季線"
    
    bias = ((last['Close'] - last['MA20']) / last['MA20']) * 100
    if bias > 15: bias_status = "⚠️ 正乖離過大"
    elif bias < -15: bias_status = "⚡ 負乖離過大"
    else: bias_status = "👌 乖離率正常"
    
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    vol_ratio = last['Volume'] / vol_ma5 if vol_ma5 > 0 else 0
    if vol_ratio > 1.5: vol_status = "🔥 爆量"
    elif vol_ratio < 0.6: vol_status = "💤 量縮"
    else: vol_status = "⚖️ 溫和"
    
    vp_status = "無資料"
    if vol_profile is not None:
        max_price = vol_profile.idxmax().mid
        if last['Close'] > max_price: vp_status = f"🧱 支撐 ({max_price:.1f})"
        else: vp_status = f"🔨 壓力 ({max_price:.1f})"

    return [
        {"指標": "MA20 (月線)", "數值": f"{last['MA20']:.2f}", "狀態": ma20_status},
        {"指標": "MA60 (季線)", "數值": f"{last['MA60']:.2f}", "狀態": ma60_status},
        {"指標": "乖離率", "數值": f"{bias:.2f}%", "狀態": bias_status},
        {"指標": "量能", "數值": f"{int(last['Volume']):,}", "狀態": vol_status},
        {"指標": "籌碼大量區", "數值": f"{max_price:.2f}" if vol_profile is not None else "-", "狀態": vp_status},
    ]

# ===========================
# 3. UI 操作區
# ===========================

# --- 側邊欄輸入 ---
st.sidebar.header("🔍 股票搜尋")
m_type = st.sidebar.radio("市場", ["美股 (US)", "台股 (TW)"])

# 修改輸入框提示，支援多檔
t_input_str = st.sidebar.text_area("輸入代號 (支援多檔，用逗號分隔)\n例如: 2330, 2317, 2454", value="2330").strip()
btn = st.sidebar.button("開始批次分析", type="primary")

# --- 主畫面：大盤狀態 ---
name, status, color = get_market_status(m_type)
if color == "green": st.success(f"**{name}**：{status}")
elif color == "red": st.error(f"**{name}**：{status}")
else: st.warning(f"**{name}**：{status}")

# --- 主畫面：個股分析邏輯 ---
if btn and t_input_str:
    # 1. 解析輸入代號
    # 把逗號、換行都換成逗號，然後切割
    raw_tickers = t_input_str.replace("\n", ",").split(",")
    target_tickers = [t.strip() for t in raw_tickers if t.strip()]
    
    # 建立一個容器來存放這批次的結果
    batch_results = []
    
    # 建立進度條
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(target_tickers)
    
    # 2. 開始迴圈處理
    for idx, ticker in enumerate(target_tickers):
        # 更新進度
        progress_bar.progress((idx) / total_tickers)
        status_text.markdown(f"### ⏳ 正在分析：**{ticker}** ({idx+1}/{total_tickers})... 請稍候")
        
        # === 關鍵：強制休息 ===
        # 第一支不用休太久，後面的每支隨機休 3~8 秒
        if idx > 0:
            sleep_time = random.uniform(3, 8)
            time.sleep(sleep_time) 
        
        # 執行分析
        df, news, vol, final_t = process_stock_data(ticker, m_type)
        
        if df is not None:
            last = df.iloc[-1]
            score = calculate_score(df)
            
            # === 顯示單一個股結果 (使用 expander 收納，避免畫面太長) ===
            # 預設展開第一支，後面的收起來
            with st.expander(f"📊 {final_t} - 評分: {score}", expanded=(idx==0)):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.metric("股價", f"{last['Close']:.2f}", f"{(last['Close']-df.iloc[-2]['Close']):.2f}")
                with c2:
                    st.progress(score)
                    st.caption(f"操盤評分: {score}")

                # 圖表
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=2), name='MA60'))
                if vol is not None:
                    try:
                        mp = vol.idxmax().mid
                        fig.add_hline(y=mp, line_dash="dot", line_color="red")
                    except: pass
                fig.update_layout(height=350, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # 指標表
                report = generate_indicator_report(df, vol)
                st.dataframe(pd.DataFrame(report), hide_index=True, use_container_width=True)

                # AI 分析
                if news and llm_available:
                    st.info(analyze_ai(news, final_t))

            # 收集結果供最後排行使用
            batch_results.append({
                "代號": final_t,
                "股價": float(f"{last['Close']:.2f}"),
                "評分": score,
                "趨勢": "多頭" if score >= 70 else "空頭" if score <= 30 else "盤整"
            })
            
            # 更新 Session History (選用，避免重複)
            if not any(d['Ticker'] == final_t for d in st.session_state.watch_list):
                 st.session_state.watch_list.append({'Ticker': final_t, 'Score': score, 'Price': float(last['Close'])})
        else:
            st.error(f"❌ {ticker} 分析失敗 (可能代號錯誤或無數據)")

    # 3. 迴圈結束，顯示最終排行
    progress_bar.progress(1.0)
    status_text.success("✅ 全部分析完成！")
    
    if batch_results:
        st.markdown("---")
        st.header("🏆 本次投資建議排名")
        st.markdown("根據操盤評分系統，針對您輸入的個股進行強弱排序：")
        
        # 轉成 DataFrame 並排序
        df_rank = pd.DataFrame(batch_results).sort_values(by="評分", ascending=False).reset_index(drop=True)
        
        # 調整顯示格式
        st.dataframe(
            df_rank,
            use_container_width=True,
            column_config={
                "評分": st.column_config.ProgressColumn(
                    "操盤評分 (越高越好)",
                    format="%d",
                    min_value=0,
                    max_value=100,
                ),
                "股價": st.column_config.NumberColumn(
                    "現價",
                    format="$ %.2f"
                )
            }
        )
        
        # 找出最強的一檔給予建議
        top_stock = df_rank.iloc[0]
        st.info(f"💡 **最佳首選**：**{top_stock['代號']}** (評分 {top_stock['評分']})。在您輸入的這批名單中，它的技術面結構最強。")

# ===========================
# 4. 側邊欄排行榜 (歷史紀錄)
# ===========================
if st.session_state.watch_list:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 歷史查詢紀錄")
    hist_df = pd.DataFrame(st.session_state.watch_list).sort_values("Score", ascending=False)
    st.sidebar.dataframe(hist_df[['Ticker', 'Score']], hide_index=True, use_container_width=True)
    if st.sidebar.button("清除歷史"):
        st.session_state.watch_list = []
        st.rerun()
