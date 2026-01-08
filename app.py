import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import openai

# --- 頁面設定 ---
st.set_page_config(page_title="ProTrader 專業操盤室", layout="wide", initial_sidebar_state="expanded")
st.title("🖥️ ProTrader 專業操盤室")
st.markdown("---")

# --- 讀取 Secrets (OpenAI Key) ---
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

# --- Session State 初始化 (用於排行榜) ---
if "watch_list" not in st.session_state:
    st.session_state.watch_list = []

# ===========================
# 函數定義區
# ===========================

def get_market_status_indicator(market_type):
    """
    大盤紅綠燈功能：分析 SPY (美股) 或 0050.TW (台股) 的趨勢
    """
    ticker = "SPY" if market_type == "美股 (US)" else "0050.TW"
    market_name = "標普500 (SPY)" if market_type == "美股 (US)" else "台灣50 (0050)"
    
    try:
        df = yf.Ticker(ticker).history(period="6mo")
        if df.empty: return None, "無法獲取數據", "grey"
        
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        latest = df.iloc[-1]
        
        # 簡易操盤邏輯判斷
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
        return market_name, f"Error: {e}", "grey"

def get_stock_data(ticker, market):
    """
    獲取個股數據並計算技術指標
    """
    # 台股自動後綴處理並轉大寫
    ticker = ticker.upper()
    if market == "台股 (TW)" and not ticker.endswith(".TW") and not ticker.isdigit():
         pass # 如果使用者輸入像是 TSLA 但選台股，就不亂加
    elif market == "台股 (TW)" and not ticker.endswith(".TW"):
        ticker = f"{ticker}.TW"
    
    stock = yf.Ticker(ticker)
    # 抓取 1 年數據以計算長均線
    df = stock.history(period="1y")
    
    if df.empty:
        return None, None, None, ticker
    
    # 1. 均線 (生命線與成本線)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 2. 近似大量區 (Volume Profile 簡易版 - 用過去半年的數據)
    df_recent = df.tail(120).copy() # 取近半年
    # 將價格切分成 30 個區間，計算各區間累積成交量
    price_bins = pd.cut(df_recent['Close'], bins=30)
    vol_profile = df_recent.groupby(price_bins, observed=False)['Volume'].sum()
    
    return df, stock, vol_profile, ticker

def calculate_score(df):
    """
    簡易操盤評分邏輯 (0-100分)
    """
    score = 50
    if len(df) < 60: return 50 # 數據不足

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 趨勢多頭排列 (+25)
    if latest['MA20'] > latest['MA60'] and latest['Close'] > latest['MA20']:
        score += 25
    # 股價在月線之上 (+10)
    elif latest['Close'] > latest['MA20']:
        score += 10
    # 股價跌破季線 (法人成本) (-25)
    elif latest['Close'] < latest['MA60']:
        score -= 25
        
    # 量能異動 (今日比起五日均量放大 1.5 倍) (+10)
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if latest['Volume'] > vol_ma5 * 1.5 and latest['Close'] > prev['Close']:
        score += 10
        
    return min(100, max(0, score))

def analyze_news_with_ai(news_list):
    """
    使用 OpenAI 分析新聞標題
    """
    if not news_list or not llm_available:
        return "無法進行 AI 分析 (無新聞或無 API Key)。"
    
    # 提取前 5 則新聞標題
    headlines = [f"- {n.get('title', 'No Title')}" for n in news_list[:5]]
    headlines_text = "\n".join(headlines)
    
    prompt = f"""
    你是一位經驗豐富的專業股市操盤手。請閱讀以下關於這檔股票的最新新聞標題：
    
    {headlines_text}
    
    請根據這些標題，以操盤手的口吻，用簡潔有力的三句話總結：
    1. 目前市場對該股的消息面情緒是偏多、偏空還是中性？
    2. 最關鍵的利多或利空因素是什麼？
    3. 給出一個短期的操作建議 (例如：留意追高風險、靜待拉回支撐、利空測試底部)。
    
    請直接回答三句話總結，不要有其他廢話。
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # 使用較快速便宜的模型
            messages=[{"role": "system", "content": "你是一個專業、冷靜、客觀的股市操盤專家。"},
                      {"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 分析失敗: {e}"

# ===========================
# 主介面佈局
# ===========================

# --- 側邊欄：輸入與排名 ---
st.sidebar.header("🔍 標的搜尋 & 設定")
market_type = st.sidebar.radio("選擇市場", ["美股 (US)", "台股 (TW)"])
ticker_input = st.sidebar.text_input("輸入代號 (例如 AAPL, TSLA, 2330, 0050)", value="").strip()
search_button = st.sidebar.button("開始分析", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("🏆 自選股戰力排行")
if st.session_state.watch_list:
    # 將 session state 轉為 DataFrame 並排序
    ranking_df = pd.DataFrame(st.session_state.watch_list)
    ranking_df = ranking_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    # 顯示排行榜，調整欄位顯示
    st.sidebar.dataframe(
        ranking_df[['Ticker', 'Score', 'Price', 'Trend']],
        column_config={
            "Score": st.column_config.ProgressColumn("操盤評分", format="%d", min_value=0, max_value=100),
            "Price": st.column_config.NumberColumn("現價", format="$%.2f")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.sidebar.info("尚未加入觀察名單，請先搜尋並分析個股。")


# --- 主畫面內容 ---

# 1. 大盤環境掃描 (最上方顯示)
st.subheader("🌍 大盤環境掃描 (Market Context)")
market_name, market_status, status_color = get_market_status_indicator(market_type)

status_container = st.container()
if status_color == "green":
    status_container.success(f"**{market_name}** 目前狀態：**{market_status}**。順勢操作，積極尋找強勢股。")
elif status_color == "red":
    status_container.error(f"**{market_name}** 目前狀態：**{market_status}**。覆巢之下無完卵，提高現金部位，保守操作。")
else:
    status_container.warning(f"**{market_name}** 目前狀態：**{market_status}**。多空不明，耐心等待方向明確。")

st.markdown("---")

# 2. 個股分析執行
if search_button and ticker_input:
    with st.spinner(f"正在以專業視角分析 {ticker_input.upper()}，請稍候..."):
        df, stock_info, vol_profile, final_ticker = get_stock_data(ticker_input, market_type)
    
    if df is not None and len(df) > 60:
        # --- A. 基本報價區 ---
        col_info1, col_info2 = st.columns([2, 1])
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = latest['Close'] - prev['Close']
        pct_change = (change / prev['Close']) * 100
        
        with col_info1:
            st.header(f"{final_ticker}")
            st.metric("目前股價", f"{latest['Close']:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
            
        # --- B. 評分與趨勢判斷 ---
        score = calculate_score(df)
        trend_str = "多頭強勢" if score >= 70 else ("空頭弱勢" if score <= 30 else "整理格局")
        
        with col_info2:
            st.write("操盤綜合評分:")
            st.progress(score)
            st.caption(f"得分: {score} / 100 ({trend_str})")

        # 更新排行榜數據
        current_stock_data = {
            'Ticker': final_ticker, 
            'Score': score, 
            'Price': float(f"{latest['Close']:.2f}"),
            'Trend': trend_str[0:2] # 取前兩個字
        }
        # 移除舊資料 (如果已存在) 並加入新資料
        st.session_state.watch_list = [d for d in st.session_state.watch_list if d['Ticker'] != final_ticker]
        st.session_state.watch_list.append(current_stock_data)
        

        # --- C. 專業圖表區 (K線 + 成本線 + 大量區示意) ---
        st.subheader("📊 操盤人結構圖 (Structure Chart)")
        
        fig = go.Figure()
        
        # K線圖
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='K線'))
        
        # 關鍵均線
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#FFD700', width=1.5), name='月線 (MA20 波段防守)'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='#32CD32', width=2), name='季線 (MA60 法人成本)'))
        
        # 找出最大量區間 (作為支撐壓力的參考)
        if not vol_profile.empty:
            max_vol_interval = vol_profile.idxmax()
            # 取區間中點作為參考價位
            max_vol_price = max_vol_interval.mid
            
            # 在圖上畫一條水平線標示大量區
            fig.add_hline(y=max_vol_price, line_dash="dot", line_color="rgba(255, 99, 71, 0.8)", annotation_text="近半年最大成交堆積區 (支撐/壓力)", annotation_position="bottom right")

        fig.update_layout(
            xaxis_rangeslider_visible=False, 
            height=550,
            template="plotly_dark", # 使用深色主題看起來更專業
            title_text=f"{final_ticker} - 日線結構分析"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- D. 專業指標解讀與 AI 新聞分析 ---
        st.subheader("🧠 深度分析與解讀")
        tab1, tab2 = st.tabs(["💡 操盤人技術觀點", "🤖 AI 新聞情緒解讀"])
        
        with tab1:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### 關鍵價位結構")
                st.write(f"**• 季線 (法人成本線):** {latest['MA60']:.2f}")
                st.write(f"**• 月線 (波段防守線):** {latest['MA20']:.2f}")
                if not vol_profile.empty:
                    st.write(f"**• 最大套牢/支撐區 (約):** {max_vol_price:.2f}")
                st.info("👉 **解讀：** 股價位於季線之上為多方勢力範圍。大量堆積區是多空交戰最激烈的價位，站上變強力支撐，跌破變沈重壓力。")

            with col_c2:
                st.markdown("#### 籌碼與動能概況")
                # 簡單判斷乖離
                bias = ((latest['Close'] - latest['MA20']) / latest['MA20']) * 100
                st.write(f"**• 月線乖離率:** {bias:.2f}%")
                if bias > 15: st.warning("乖離過大，留意短線獲利回吐賣壓。")
                elif bias < -15: st.warning("負乖離過大，可能出現跌深反彈。")
                
                st.write("*(註：台股真實主力籌碼/融資數據需付費源，此處以價量結構與均線近似推估)*")
                
        with tab2:
            if llm_available:
                try:
                    news = stock_info.news
                    if news:
                        with st.spinner("AI 正在閱讀新聞並撰寫操盤總結..."):
                            ai_summary = analyze_news_with_ai(news)
                        
                        st.markdown("#### 🤖 OpenAI 操盤手摘要")
                        st.success(ai_summary)
                        
                        st.markdown("---")
                        st.markdown("#### 📰 原始新聞標題參考")
                        for n in news[:3]:
                            pub_time = datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
                            st.markdown(f"• [{n.get('title')}]({n.get('link')}) - *{pub_time}*")
                    else:
                        st.warning("找不到近期相關新聞。")
                except Exception as e:
                    st.error(f"獲取新聞時發生錯誤: {e}")
            else:
                st.warning("請先設定 OpenAI API Key 以啟用 AI 智慧解讀功能。")

    elif ticker_input:
        st.error(f"找不到代號 {ticker_input} 的數據，或數據長度不足以計算指標。請確認代號輸入正確。")
else:
    st.info("請在左側側邊欄選擇市場並輸入股票代號，點擊「開始分析」。")
