import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import openai
import requests

# ===========================
# 1. 基礎設定與 API 初始化
# ===========================

st.set_page_config(page_title="ProTrader 專業操盤室", layout="wide", initial_sidebar_state="expanded")
st.title("🖥️ ProTrader 專業操盤室")
st.markdown("---")

# 讀取 OpenAI Key (從 Streamlit Secrets)
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=openai_api_key)
    llm_available = True
except FileNotFoundError:
    st.warning("⚠️ 未檢測到 OpenAI API Key。請在 Streamlit Secrets 中設定 `OPENAI_API_KEY` 以啟用 AI 新聞解讀功能。目前僅提供基礎數據。")
    llm_available = False
except Exception as e:
    st.error(f"OpenAI 設定錯誤: {e}")
    llm_available = False

# 初始化觀察名單 (Session State)
if "watch_list" not in st.session_state:
    st.session_state.watch_list = []

# ===========================
# 2. 核心函數 (含防封鎖與快取機制)
# ===========================

def get_session():
    """建立偽裝成瀏覽器的 Session"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    return session

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_status_indicator(market_type):
    """
    大盤紅綠燈：分析 SPY (美股) 或 0050.TW (台股)
    快取設定：1小時 (ttl=3600)，因為大盤趨勢日內變化不大
    """
    ticker = "SPY" if market_type == "美股 (US)" else "0050.TW"
    market_name = "標普500 (SPY)" if market_type == "美股 (US)" else "台灣50 (0050)"
    
    try:
        session = get_session()
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period="6mo")
        
        if df.empty: return None, "無法獲取數據", "grey"
        
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        latest = df.iloc[-1]
        
        # 簡易操盤邏輯
        if latest['Close'] > latest['MA60'] and latest['MA20'] > latest['MA60']:
            status = "多頭格局 (安全)"
            color = "green"
        elif latest['Close'] < latest['MA60']:
            status = "空頭修正 (危險)"
            color = "red"
        else:
            status = "震盪整理 (觀望)"
            color = "yellow"
            
        return market_name, status, color
    except Exception as e:
        return market_name, f"連線忙碌 ({str(e)[:15]}...)", "grey"

@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(ticker, market):
    """
    獲取個股數據、指標與新聞
    快取設定：15分鐘 (ttl=900)，避免短時間重複請求導致被鎖
    """
    # 格式化代號
    ticker = ticker.upper().strip()
    if market == "台股 (TW)" and not ticker.endswith(".TW") and not ticker.isdigit():
         pass 
    elif market == "台股 (TW)" and not ticker.endswith(".TW"):
        ticker = f"{ticker}.TW"
    
    try:
        session = get_session()
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period="1y")
        
        if df.empty:
            return None, None, None, None, ticker
        
        # 1. 提取新聞 (需在快取函數內提取並轉為純列表，避免 pickle 問題)
        news_list = stock.news if hasattr(stock, 'news') else []

        # 2. 計算均線
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 3. 計算 Volume Profile (近似大量區)
        df_recent = df.tail(120).copy() # 取近半年
        # 處理可能的錯誤 (如數據不足)
        if len(df_recent) > 1:
            price_bins = pd.cut(df_recent['Close'], bins=30)
            vol_profile = df_recent.groupby(price_bins, observed=False)['Volume'].sum()
        else:
            vol_profile = pd.Series()
        
        return df, news_list, vol_profile, stock.info, ticker

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None, None, None, None, ticker

def calculate_score(df):
    """計算操盤評分 (0-100)"""
    score = 50
    if len(df) < 60: return 50

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 趨勢多頭 (+25)
    if latest['MA20'] > latest['MA60'] and latest['Close'] > latest['MA20']:
        score += 25
    # 站上月線 (+10)
    elif latest['Close'] > latest['MA20']:
        score += 10
    # 跌破季線 (-25)
    elif latest['Close'] < latest['MA60']:
        score -= 25
        
    # 量能異動 (+10)
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if latest['Volume'] > vol_ma5 * 1.5 and latest['Close'] > prev['Close']:
        score += 10
        
    return min(100, max(0, score))

def analyze_news_with_ai(news_list):
    """使用 OpenAI 分析新聞"""
    if not news_list or not llm_available:
        return "無法進行 AI 分析 (無新聞或無 API Key)。"
    
    headlines = [f"- {n.get('title', 'No Title')}" for n in news_list[:5]]
    headlines_text = "\n".join(headlines)
    
    prompt = f"""
    你是一位專業股市操盤手。請閱讀以下新聞標題：
    {headlines_text}
    
    請用簡潔三句話總結：
    1. 市場情緒 (偏多/偏空/中性)
    2. 關鍵因素
    3. 短期操作建議
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "你是一個專業、客觀的操盤專家。"},
                      {"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 分析失敗: {e}"

# ===========================
# 3. UI 介面佈局
# ===========================

# --- 側邊欄 ---
st.sidebar.header("🔍 標的搜尋")
market_type = st.sidebar.radio("市場類型", ["美股 (US)", "台股 (TW)"])
ticker_input = st.sidebar.text_input("輸入代號 (如 AAPL, 2330)", value="").strip()
search_button = st.sidebar.button("開始分析", type="primary")

# 排行榜
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 自選股戰力排行")
if st.session_state.watch_list:
    ranking_df = pd.DataFrame(st.session_state.watch_list)
    ranking_df = ranking_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    st.sidebar.dataframe(
        ranking_df[['Ticker', 'Score', 'Price', 'Trend']],
        column_config={
            "Score": st.column_config.ProgressColumn("評分", format="%d", min_value=0, max_value=100),
            "Price": st.column_config.NumberColumn("現價", format="$%.2f")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.sidebar.info("尚未加入觀察名單")

# --- 主畫面 ---

# 1. 大盤紅綠燈
st.subheader("🌍 大盤環境掃描")
market_name, market_status, status_color = get_market_status_indicator(market_type)
status_container = st.container()
if status_color == "green":
    status_container.success(f"**{market_name}**：**{market_status}**。順勢操作，積極尋找強勢股。")
elif status_color == "red":
    status_container.error(f"**{market_name}**：**{market_status}**。建議提高現金部位，保守操作。")
else:
    status_container.warning(f"**{market_name}**：**{market_status}**。多空不明，耐心等待。")

st.markdown("---")

# 2. 個股分析
if search_button and ticker_input:
    with st.spinner(f"正在分析 {ticker_input.upper()} (含 AI 解讀)..."):
        # 呼叫主函數
        df, news_list, vol_profile, info, final_ticker = get_stock_data(ticker_input, market_type)
    
    # === 錯誤處理區 ===
    if df is None:
        st.error(f"⚠️ 無法獲取 {ticker_input} 的數據。")
        st.warning("可能原因：1. 代號錯誤 2. Yahoo Finance 暫時限制連線 (請稍候再試)")
    
    # === 成功顯示區 ===
    elif len(df) > 60:
        # A. 基本資訊
        c1, c2 = st.columns([2, 1])
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = latest['Close'] - prev['Close']
        pct = (change / prev['Close']) * 100
        
        with c1:
            st.header(f"{final_ticker}")
            st.metric("股價", f"{latest['Close']:.2f}", f"{change:.2f} ({pct:.2f}%)")
        
        score = calculate_score(df)
        trend_str = "多頭" if score >= 70 else ("空頭" if score <= 30 else "盤整")
        
        with c2:
            st.write("操盤評分:")
            st.progress(score)
            st.caption(f"{score} 分 ({trend_str})")
            
        # 更新排行榜
        new_data = {'Ticker': final_ticker, 'Score': score, 'Price': float(latest['Close']), 'Trend': trend_str}
        st.session_state.watch_list = [d for d in st.session_state.watch_list if d['Ticker'] != final_ticker]
        st.session_state.watch_list.append(new_data)
        
        # B. 技術圖表
        st.subheader("📊 結構分析圖")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20 (月線)'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=2), name='MA60 (季線)'))
        
        if not vol_profile.empty:
            max_vol_price = vol_profile.idxmax().mid
            fig.add_hline(y=max_vol_price, line_dash="dot", line_color="red", annotation_text="大量支撐/壓力區")
            
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # C. 深度分析 (Tab)
        tab1, tab2 = st.tabs(["💡 技術籌碼面", "🤖 AI 新聞面"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 關鍵價位")
                st.write(f"**季線 (成本):** {latest['MA60']:.2f}")
                st.write(f"**月線 (防守):** {latest['MA20']:.2f}")
                if not vol_profile.empty:
                    st.write(f"**大量區:** {max_vol_price:.2f}")
            with col_b:
                bias = ((latest['Close'] - latest['MA20']) / latest['MA20']) * 100
                st.markdown("#### 風險指標")
                st.write(f"**月線乖離:** {bias:.2f}%")
                if bias > 15: st.warning("乖離過大，留意回檔")
                
        with tab2:
            if llm_available and news_list:
                with st.spinner("AI 正在閱讀新聞..."):
                    summary = analyze_news_with_ai(news_list)
                st.success(summary)
                st.markdown("---")
                st.markdown("**原始新聞來源：**")
                for n in news_list[:3]:
                    t = datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
                    st.markdown(f"- [{n.get('title')}]({n.get('link')}) ({t})")
            elif not news_list:
                st.info("暫無相關新聞")
            else:
                st.warning("請設定 OpenAI Key 以啟用此功能")
                
    else:
        st.error("數據長度不足，無法計算技術指標。")
else:
    st.info("請輸入代號並點擊分析")
