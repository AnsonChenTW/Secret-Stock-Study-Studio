import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px  # 引入繪圖神器

# ==========================================
# 🔑 設定 API Key
# ==========================================
# 雲端版讀取 Key 的方式
import os
# 如果在 Hugging Face，它會從 Secrets 讀取；如果在本地，請手動填入
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "AIzaSyD8oBaP663IpoU4E5UYlVsw2tCPZ7YUj1g")

try:
    genai.configure(api_key=GENAI_API_KEY)
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False

st.set_page_config(page_title="美股資金戰情室", layout="wide")

# --- 設定：細分產業 ETF 清單 ---
INDUSTRY_MAPPING = {
    "SMH (半導體)": ["NVDA", "TSM", "AVGO", "AMD", "QCOM", "TXN", "MU", "INTC", "AMAT", "LRCX"],
    "IGV (軟體 SaaS)": ["MSFT", "ADBE", "CRM", "ORCL", "PLTR", "NOW", "SNOW", "PANW", "CRWD", "DDOG"],
    "XBI (生物科技)": ["AMGN", "GILD", "VRTX", "REGN", "MRNA", "BNTX", "BIIB", "ILMN"],
    "ITA (航太軍工)": ["RTX", "LMT", "BA", "GD", "NOC", "LHX", "HII", "GE"],
    "KRE (區域銀行)": ["NYCB", "WAL", "KEY", "CFG", "FITB", "HBAN", "RF"],
    "XHB (房屋建築)": ["DHI", "LEN", "PHM", "TOL", "HD", "LOW", "SHW"],
    "TAN (太陽能/綠能)": ["FSLR", "ENPH", "SEDG", "RUN", "JKS", "CSIQ"],
    "XLE (傳統能源)": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "KMI"],
    "XRT (零售消費)": ["AMZN", "WMT", "COST", "TGT", "HD", "LOW", "BBY"],
    "IHI (醫療設備)": ["TMO", "ABT", "MDT", "SYK", "BSX", "EW", "ISRG"],
    "JETS (航空旅運)": ["DAL", "UAL", "AAL", "LUV", "BKNG", "EXPE", "CCL", "RCL"],
    "META (元宇宙/通訊)": ["META", "GOOGL", "NFLX", "DIS", "ROKU", "SNAP"],
}
INDUSTRY_ETFS = list(INDUSTRY_MAPPING.keys())

# --- 輔助函數 ---
def format_large_number(num):
    """將數字轉換為 K, M, B (千, 百萬, 十億)"""
    if num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B" # Billions
    elif num >= 1_000_000:
        return f"${num / 1_000_000:.2f}M" # Millions
    else:
        return f"${num:.2f}"

def get_sector_money_flow():
    """計算板塊資金流向與漲跌幅"""
    tickers = [s.split()[0] for s in INDUSTRY_ETFS]
    try:
        data = yf.download(tickers, period="5d", progress=False)
        if data.empty: return pd.DataFrame()
        
        close = data['Close']
        volume = data['Volume']
        
        # 計算當日數據
        latest_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        latest_vol = volume.iloc[-1]
        
        # 漲跌幅
        pct_change = (latest_close - prev_close) / prev_close * 100
        
        # 估算成交金額 (Money Flow) = 收盤價 * 成交量
        money_flow = latest_close * latest_vol
        
        # 整理成 DataFrame
        results = []
        for code in tickers:
            full_name = next((name for name in INDUSTRY_ETFS if name.startswith(code)), code)
            results.append({
                "Sector": full_name,
                "Ticker": code,
                "Change (%)": pct_change[code],
                "Money Flow ($)": money_flow[code],
                "Raw Money Flow": money_flow[code] # 用於排序
            })
            
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return pd.DataFrame()

def analyze_top_stocks(sector_name):
    """分析特定板塊內的前五大成交個股"""
    tickers = INDUSTRY_MAPPING.get(sector_name, [])
    try:
        # 下載數據 (只抓一天即可，求速度)
        df = yf.download(tickers, period="1d", progress=False)
        
        results = []
        for ticker in tickers:
            try:
                price = df['Close'][ticker].iloc[-1]
                vol = df['Volume'][ticker].iloc[-1]
                turnover = price * vol
                results.append({
                    "Code": ticker,
                    "Price": price,
                    "Volume": vol,
                    "Turnover": turnover
                })
            except:
                continue
        
        # 依成交金額排序，取前 5
        sorted_df = pd.DataFrame(results).sort_values(by="Turnover", ascending=False).head(5)
        return sorted_df
    except:
        return pd.DataFrame()

# --- UI 介面 ---
st.title("📊 美股資金流向戰情室 (Money Flow Dashboard)")
st.markdown("### 1. 全市場資金熱力圖 (Sector Heatmap)")

if st.button("🚀 啟動戰情室分析", type="primary"):
    
    with st.spinner("正在計算全市場資金流向..."):
        df_sector = get_sector_money_flow()
        
    if not df_sector.empty:
        # 1. 繪製熱力圖 (Treemap)
        # 顏色：紅綠 (漲跌)，大小：資金流向
        fig = px.treemap(
            df_sector, 
            path=['Sector'], 
            values='Raw Money Flow',
            color='Change (%)',
            color_continuous_scale=['red', 'black', 'green'],
            color_continuous_midpoint=0,
            hover_data={'Money Flow ($)': True, 'Change (%)': ':.2f'},
            title="板塊資金熱力圖 (方塊越大=錢越多, 越綠=漲越兇)"
        )
        # 顯示金額格式
        df_sector['Money Flow Label'] = df_sector['Raw Money Flow'].apply(format_large_number)
        fig.data[0].customdata = df_sector[['Money Flow Label', 'Change (%)']]
        fig.data[0].texttemplate = "%{label}<br>%{customdata[1]:.2f}%<br>%{customdata[0]}"
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. 找出前三大資金流入板塊
        st.divider()
        st.markdown("### 2. 資金集中前三大類別 & 龍頭股")
        
        # 這裡我們依「漲跌幅」排序找強勢，或者依「資金量」排序找熱門
        # 假設策略：找「漲幅前三名」的類別
        top_3_sectors = df_sector.sort_values(by="Change (%)", ascending=False).head(3)
        
        all_top_stocks = [] # 用於收集所有強勢股代碼
        
        cols = st.columns(3)
        for i, (index, row) in enumerate(top_3_sectors.iterrows()):
            sector_name = row['Sector']
            with cols[i]:
                st.subheader(f"🏆 {sector_name}")
                st.markdown(f"**漲幅:** {row['Change (%)']:.2f}% | **資金:** {format_large_number(row['Raw Money Flow'])}")
                st.markdown("---")
                
                # 分析該類別前五大
                top_stocks_df = analyze_top_stocks(sector_name)
                
                if not top_stocks_df.empty:
                    for _, stock in top_stocks_df.iterrows():
                        st.markdown(f"**{stock['Code']}**")
                        st.caption(f"價: ${stock['Price']:.1f} | 量: {format_large_number(stock['Volume'])}")
                        st.caption(f"成交額: {format_large_number(stock['Turnover'])}")
                        all_top_stocks.append(stock['Code'])
                else:
                    st.write("無數據")

        # 3. 生成複製清單 (解決需求 4)
        st.divider()
        st.markdown("### 3. 串接分析 (Export to AI Model)")
        st.info("您可以複製下方的強勢股清單，貼到您的舊版 App 或其他工具進行深度分析。")
        
        # 將代碼轉為字串 "NVDA, AMD, TSM..."
        stock_list_str = ", ".join(all_top_stocks)
        st.code(stock_list_str, language="text")
        
        st.markdown(f"👉 [點擊前往您的舊版 App (Stock-AI-v3)](https://huggingface.co/spaces/AnsonTW/Stock-AI-v3)")
        st.markdown("**操作提示：** 複製上方的代碼，點擊連結開啟舊 App，貼入輸入框即可執行。")
        
    else:
        st.error("無法取得市場數據，請稍後再試。")