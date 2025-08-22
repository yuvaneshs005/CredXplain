import os

# -------------------------------
# Alpha Vantage API Key
# -------------------------------
os.environ["ALPHAVANTAGE_API_KEY"] = "NJSOJQKBF4BHH7Y6"

# -------------------------------
# SEC EDGAR User-Agent
# -------------------------------
os.environ["SEC_USER_AGENT"] = "chillakalyan78@gmail.com HackathonApp/1.0"

# -------------------------------
# Optional: verify
# -------------------------------
print("Alpha Vantage Key set:", bool(os.getenv("ALPHAVANTAGE_API_KEY")))
print("SEC User-Agent set:", os.getenv("SEC_USER_AGENT"))

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "NJSOJQKBF4BHH7Y6")
SEC_UA = os.getenv("SEC_USER_AGENT", "chillakalyan78@gmail.com HackathonApp/1.0")
#verify - OPtional
print("Alpha Vantage Key:", ALPHAVANTAGE_API_KEY)
print("SEC User-Agent:", SEC_UA)

import os, time, json, math, datetime, urllib.parse, re
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from textblob import TextBlob
from pandas_datareader import data as pdr
from ratelimit import limits, sleep_and_retry
from bs4 import BeautifulSoup
import requests

# -------------------------------
# 0️⃣ Set API keys
# -------------------------------
# 579b464db66ec23bdd000001ca253fa7c51947b1572eb310a731e99d
MY_API_KEY = "579b464db66ec23bdd0000016b699a9bd8da4d0e5653cc18839f110d"
os.environ["ALPHAVANTAGE_API_KEY"] = "NJSOJQKBF4BHH7Y6"
os.environ["SEC_USER_AGENT"] = "chillakalyan78@gmail.com HackathonApp/1.0"
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
SEC_UA = os.getenv("SEC_USER_AGENT")

# -------------------------------
# 1️⃣ Company metadata
# -------------------------------
COMPANIES = {
    "INFY.NS": {"name": "Infosys", "adr": "INFY", "cin": "L85110KA1981PLC013115"},
    "RELIANCE.NS": {"name": "Reliance Industries", "adr": None, "cin": "L17110MH1973PLC019786"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "adr": "HDB", "cin": "L65920MH1994PLC080618"},
    "TCS.NS": {"name": "Tata Consultancy Services", "adr": None, "cin": "L22210MH1995PLC084781"},
}

# -------------------------------
# 2️⃣ Helper functions
# -------------------------------
def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def sentiment_of_text(text: str) -> float:
    if not text: return 0.0
    return TextBlob(text).sentiment.polarity

def avg(lst: List[float], default=0.0):
    return sum(lst)/len(lst) if lst else default

# -------------------------------
# 3️⃣ Yahoo Finance
# -------------------------------
def get_financials_yahoo(ticker_ns: str) -> Dict[str, Any]:
    try:
        stock = yf.Ticker(ticker_ns)
        info = stock.info or {}
        return {
            "Ticker": ticker_ns,
            "MarketCap": info.get("marketCap", 0),
            "PE_Ratio": info.get("trailingPE", 0),
            "DebtToEquity": info.get("debtToEquity", 0),
            "ProfitMargins": info.get("profitMargins", 0)
        }
    except:
        return {"Ticker": ticker_ns, "MarketCap": 0, "PE_Ratio": 0, "DebtToEquity": 0, "ProfitMargins": 0}

# -------------------------------
# 4️⃣ FRED + World Bank
# -------------------------------
def get_macro_fred_worldbank(country_code="IND") -> Dict[str, Any]:
    out = {}
    try:
        start = datetime.datetime(2018,1,1)
        end = datetime.datetime.today()
        gdp = pdr.DataReader('GDP', 'fred', start, end)
        out["FRED_GDP"] = safe_float(gdp.iloc[-1].values[0])
    except: out["FRED_GDP"] = 0.0
    try:
        wb_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/FP.CPI.TOTL.ZG?format=json&per_page=5"
        r = requests.get(wb_url, timeout=15)
        data = r.json()
        val = None
        if isinstance(data, list) and len(data) == 2:
            for row in data[1]:
                if row.get("value") is not None:
                    val = row["value"]; break
        out["WB_CPI_YoY"] = safe_float(val, 0.0)
    except: out["WB_CPI_YoY"] = 0.0
    return out

# -------------------------------
# 5️⃣ Alpha Vantage
# -------------------------------
def get_alpha_indicators(symbol="INFY.BSE") -> Dict[str, Any]:
    if not ALPHAVANTAGE_API_KEY:
        return {"ALPHA_used": False, "ALPHA_msg": "No API key set"}
    try:
        url = ("https://www.alphavantage.co/query"
               f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=compact&apikey={ALPHAVANTAGE_API_KEY}")
        r = requests.get(url, timeout=20)
        j = r.json()
        ts_key = [k for k in j.keys() if "Time Series" in k]
        if not ts_key: return {"ALPHA_used": True, "ALPHA_vol_30d": 0.0}
        series = j[ts_key[0]]
        closes = [safe_float(vals.get("5. adjusted close", vals.get("4. close", 0))) for d, vals in list(series.items())[:60]]
        closes = list(reversed(closes))
        rets = [math.log(closes[i+1]/closes[i]) for i in range(len(closes)-1) if closes[i]>0 and closes[i+1]>0]
        vol_30d = np.std(rets[-30:]) * math.sqrt(252) if len(rets)>=30 else (np.std(rets)*math.sqrt(252) if rets else 0.0)
        return {"ALPHA_used": True, "ALPHA_vol_30d": vol_30d}
    except: return {"ALPHA_used": True, "ALPHA_vol_30d": 0.0}

# -------------------------------
# 6️⃣ SEC EDGAR filings
# -------------------------------
@sleep_and_retry
@limits(calls=8, period=1)
def _sec_get(url):
    return requests.get(url, headers={"User-Agent": SEC_UA}, timeout=15)

def get_sec_recent_filings(ticker_or_adr: str) -> Dict[str, Any]:
    if not ticker_or_adr: return {"SEC_used": False, "SEC_msg": "No US ticker/ADR"}
    try:
        m = _sec_get("https://www.sec.gov/files/company_tickers.json").json()
        cik = None
        for row in m.values():
            if row.get("ticker","").lower() == ticker_or_adr.lower():
                cik = str(row.get("cik_str")).zfill(10)
                break
        if not cik: return {"SEC_used": False, "SEC_msg": f"CIK not found for {ticker_or_adr}"}
        j = _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
        forms = j.get("filings", {}).get("recent", {})
        form_list, dates = list(forms.get("form", [])), list(forms.get("filingDate", []))
        cutoff = datetime.date.today() - datetime.timedelta(days=180)
        recent_count, recent_forms = 0, []
        for f, d in zip(form_list, dates):
            try:
                dt = datetime.datetime.strptime(d, "%Y-%m-%d").date()
                if dt >= cutoff: recent_count+=1; recent_forms.append(f)
            except: pass
        return {"SEC_used": True, "SEC_recent_180d": recent_count, "SEC_recent_forms": recent_forms[:5]}
    except: return {"SEC_used": True, "SEC_recent_180d": 0, "SEC_recent_forms": []}

# -------------------------------
# 7️⃣ MCA data
# -------------------------------
def get_mca_data(cin: str) -> Dict[str, Any]:
    try:
        url = f"https://apisetu.gov.in/api/mca/companies/{cin}"
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            j = res.json()
            return {
                "MCA_used": True,
                "MCA_CompanyName": j.get("companyName"),
                "MCA_Status": j.get("companyStatus"),
                "MCA_Class": j.get("companyClass"),
                "MCA_Category": j.get("companyCategory"),
                "MCA_SubCategory": j.get("companySubCategory"),
                "MCA_PaidUpCapital": j.get("paidUpCapital"),
                "MCA_DateIncorp": j.get("dateOfIncorporation")
            }
        else: return {"MCA_used": False, "MCA_error": f"HTTP {res.status_code}"}
    except: return {"MCA_used": False, "MCA_error": "Request failed"}

# -------------------------------
# 8️⃣ Sector signals
# -------------------------------
def get_sector_signals(company_name: str, ticker_ns: str) -> Dict[str, Any]:
    out = {}
    try:
        stock_price = yf.Ticker(ticker_ns).history(period="1d")["Close"].iloc[-1]
        out["StockPrice"] = float(stock_price)
        if company_name == "Reliance Industries":
            out["BrentOil"] = float(yf.Ticker("BZ=F").history(period="1mo")["Close"].iloc[-1])
            out["ShippingProxyBDRY"] = float(yf.Ticker("BDRY").history(period="1mo")["Close"].iloc[-1])
        elif company_name in ["Infosys", "Tata Consultancy Services"]:
            out["USDINR"] = float(yf.Ticker("INR=X").history(period="1mo")["Close"].iloc[-1])
        elif company_name == "HDFC Bank":
            out["NSEBankIndex"] = float(yf.Ticker("^NSEBANK").history(period="1mo")["Close"].iloc[-1])
    except: pass
    return out

# -------------------------------
# 9️⃣ News & Press Releases
# -------------------------------
def get_news_sentiment(company: str, max_items=10) -> Dict[str, Any]:
    try:
        q = urllib.parse.quote(company)
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}")
        titles = [e.title for e in feed.entries[:max_items]]
        sents  = [sentiment_of_text(t) for t in titles]
        return {"News_count": len(titles), "News_sent": avg(sents)}
    except: return {"News_count":0,"News_sent":0.0}

def get_press_release_sentiment(company: str) -> Dict[str, Any]:
    try:
        q = urllib.parse.quote(f'{company} site:{company.split()[0].lower()}.com "press release" OR newsroom OR media')
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}")
        titles = [e.title for e in feed.entries[:8]]
        sents  = [sentiment_of_text(t) for t in titles]
        return {"PR_count": len(titles), "PR_sent": avg(sents)}
    except: return {"PR_count":0,"PR_sent":0.0}

def compute_score(data: Dict[str, Any]) -> Dict[str, Any]:
    score = 50  # base score

    # ✅ Financial health
    if data.get("DebtToEquity", 0) < 1: score += 10
    elif data.get("DebtToEquity", 0) > 2: score -= 10

    if data.get("ProfitMargins", 0) > 0.15: score += 10
    else: score -= 5

    # ✅ Sentiment
    score += 5 * data.get("News_sent", 0)
    score += 5 * data.get("PR_sent", 0)

    # ✅ Volatility (AlphaVantage)
    alpha = data.get("AlphaVantage", {})
    if alpha and alpha.get("ALPHA_vol_30d", 0) < 0.2:
        score += 5
    else:
        score -= 5

    # ✅ Bound the score 0–100
    score = max(0, min(100, score))

    return {"Score": score}


# -------------------------------
# 10️⃣ Aggregator (Twitter removed)
# -------------------------------

def fetch_all_sources(ticker_ns: str, meta: Dict[str, Any], api_key: str = MY_API_KEY) -> Dict[str, Any]:
    name, adr = meta["name"], meta.get("adr")
    res: Dict[str, Any] = {"Company": name, "Ticker": ticker_ns}
    res.update(get_financials_yahoo(ticker_ns))
    res.update(get_macro_fred_worldbank("IND"))
    res.update({"AlphaVantage": get_alpha_indicators(symbol=ticker_ns.replace(".NS",""))})
    res.update(get_sec_recent_filings(adr))
    res.update(get_mca_data(meta.get("cin")))
    res.update(get_sector_signals(name, ticker_ns))
    res.update(get_news_sentiment(name))
    res.update(get_press_release_sentiment(name))

    # ⭐ Add scoring here
    res.update(compute_score(res))
    return res



# -------------------------------
# 11️⃣ Main CSV-focused execution
# -------------------------------
if __name__ == "__main__":
    all_data = []
    for ticker, meta in COMPANIES.items():
        print(f"Fetching data for {meta['name']}...")
        data = fetch_all_sources(ticker, meta, api_key=MY_API_KEY)
        all_data.append(data)

    df = pd.DataFrame(all_data)
    print(df.head())

    # Save to CSV only
    df.to_csv("all_company_data_MCA.csv", index=False)
    print("\n✅ Saved: all_company_data.csv")

import requests

url = "https://api.data.gov.in/resource/4dbe5667-7b6b-41d7-82af-211562424d9a"
params = {
    "api-key": "579b464db66ec23bdd00001ca253fa7c51947b1572eb310a731e99d",
    "format": "json"
}

response = requests.get(url, params=params)
print(response.status_code)
print(response.json())

# Libraries to install:
# pip install yfinance feedparser textblob pandas pandas-datareader

# import yfinance as yf
# import feedparser
# from textblob import TextBlob
# import pandas as pd
# import datetime
# import pandas_datareader.data as web
# import urllib.parse


# def get_financial_data(ticker):
#     stock = yf.Ticker(ticker)
#     info = stock.info  # use .info, not fast_info
#     data = {
#         "Ticker": ticker,
#         "MarketCap": info.get("marketCap", 0),
#         "PE_Ratio": info.get("trailingPE", 0),
#         "DebtToEquity": info.get("debtToEquity", 0),  # actual value
#         "ProfitMargins": info.get("profitMargins", 0)  # actual value
#     }
#     return data


# # -----------------------------
# # Structured Data: FRED GDP
# # -----------------------------
# def get_macro_data():
#     start = datetime.datetime(2020, 1, 1)
#     end = datetime.datetime.today()
#     try:
#         gdp = web.DataReader('GDP', 'fred', start, end)
#         latest_gdp = gdp.iloc[-1].values[0] if not gdp.empty else 0
#     except:
#         latest_gdp = 0
#     return {"GDP": latest_gdp}

# # -----------------------------
# # Structured Data: Sector Signals
# # -----------------------------
# def get_sector_data(company):
#     try:
#         if company == "Reliance Industries":
#             data = yf.Ticker("BZ=F").history(period="1mo")
#             return {"BrentOil": float(data["Close"].iloc[-1])}
#         elif company in ["Infosys", "Tata Consultancy Services"]:
#             data = yf.Ticker("INR=X").history(period="1mo")
#             return {"USDINR": float(data["Close"].iloc[-1])}
#         elif company == "HDFC Bank":
#             data = yf.Ticker("^NSEBANK").history(period="1mo")
#             return {"BankIndex": float(data["Close"].iloc[-1])}
#         else:
#             return {}
#     except:
#         return {}


# # -----------------------------
# # Unstructured Data: News + Sentiment
# # -----------------------------
# import requests

# def get_news_sentiment(company):
#     query = urllib.parse.quote(company)
#     feed_url = f"https://news.google.com/rss/search?q={query}"

#     try:
#         # use requests with headers to avoid 403/RemoteDisconnected
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(feed_url, headers=headers, timeout=10)
#         response.raise_for_status()

#         feed = feedparser.parse(response.content)

#         sentiments = []
#         headlines = []
#         for entry in feed.entries[:5]:  # top 5 headlines
#             text = entry.title
#             sentiment = TextBlob(text).sentiment.polarity
#             sentiments.append(sentiment)
#             headlines.append(text)

#         avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
#         return {"Sentiment": avg_sentiment, "Headlines": headlines}

#     except Exception as e:
#         print(f"[WARN] Could not fetch news for {company}: {e}")
#         return {"Sentiment": 0, "Headlines": []}


# # -----------------------------
# # Unstructured Data: Press Releases (simulated)
# # -----------------------------
# def get_press_release_signal(company):
#     # Simulated rule-based impact (placeholder)
#     # In real system: scrape RSS/official filings
#     dummy_data = {
#         "Infosys": "Signed major AI deal",
#         "Reliance Industries": "Raised debt for expansion",
#         "HDFC Bank": "Issued bonus shares",
#         "Tata Consultancy Services": "Announced layoffs due to automation"
#     }
#     headline = dummy_data.get(company, "No major update")
#     sentiment = TextBlob(headline).sentiment.polarity
#     return {"PressHeadline": headline, "PressSentiment": sentiment}


# #     return round(score, 2)
# def compute_credit_score(financials, macro, news, sector, press, company_name=""):
#     # ✅ Base score increased for overall stronger rating
#     # base = 85
#     # score = base
#     base = 75
#     if company_name == "Reliance Industries":
#         base = 85
#     score = base


#     # ---------------------------
#     # Financial Health Adjustments
#     # ---------------------------
#     # Debt-to-Equity
#     dte = financials.get("DebtToEquity") or 0
#     if dte < 0.5:  # very low debt is good
#         score += 10
#     elif dte > 2:  # high debt is bad
#         score -= 10
#     else:  # moderate debt
#         score += 5 - dte*2

#     # PE Ratio
#     pe = financials.get("PE_Ratio") or 0
#     if pe > 25:
#         score -= min((pe - 15)/2, 12)  # penalize high PE
#     elif pe < 10:
#         score += 5  # reward very low PE

#     # Profit Margins
#     pm = financials.get("ProfitMargins") or 0
#     score += min(pm * 50, 15)  # higher weight than before

#     # ---------------------------
#     # Sentiment Adjustments
#     # ---------------------------
#     news_sent = news.get("Sentiment") or 0
#     press_sent = press.get("PressSentiment") or 0
#     score += max(-5, min(5, news_sent * 15))  # stronger impact
#     score += max(-3, min(3, press_sent * 10))

#     # ---------------------------
#     # Macro GDP Adjustment
#     # ---------------------------
#     score += (float(macro.get("GDP") or 0) % 3)

#     # ---------------------------
#     # Sectoral Adjustments
#     # ---------------------------
#     if company_name == "Reliance Industries":
#         # Brent Oil: higher is positive
#         if "BrentOil" in sector:
#             score += (sector["BrentOil"] - 70) / 5  # more sensitive
#         # Shipping proxy: lower costs better
#         if "ShippingProxyBDRY" in sector:
#             score += max(0, 5 - (sector["ShippingProxyBDRY"] - 10)/5)
#         # USD/INR
#         if "USDINR" in sector:
#             score += (sector["USDINR"] - 80) / 2

#     elif company_name in ["Infosys", "Tata Consultancy Services"]:
#         if "USDINR" in sector:
#             score += (sector["USDINR"] - 80) / 5

#     elif company_name == "HDFC Bank":
#         if "BankIndex" in sector:
#             score += (sector["BankIndex"] - 30000) / 5000

#     # ---------------------------
#     # Clamp the Score
#     # ---------------------------
#     # score = max(50, min(95, score))  # allow higher top scores
#     score = max(60, min(90, score))

#     # ---------------------------
#     # Floor for Blue-Chip Companies
#     # ---------------------------
#     if company_name in ["Reliance Industries", "Infosys", "Tata Consultancy Services", "HDFC Bank"]:
#         score = max(score, 70)

#     return round(score, 2)


# -----------------------------
# Driver: Multi-Company Analysis
# -----------------------------
if __name__ == "__main__":
    companies = {
        "INFY.NS": "Infosys",
        "RELIANCE.NS": "Reliance Industries",
        "HDFCBANK.NS": "HDFC Bank",
        "TCS.NS": "Tata Consultancy Services"
    }

    results = []
    macro = get_macro_data()

    for ticker, name in companies.items():
        financials = get_financial_data(ticker)
        news = get_news_sentiment(name)
        sector = get_sector_data(name)
        press = get_press_release_signal(name)

        score = compute_credit_score(financials, macro, news, sector, press)

        results.append({
            "Company": name,
            "Ticker": ticker,
            "Score": score,
            "Sentiment": news["Sentiment"],
            "DebtToEquity": financials["DebtToEquity"],
            "PE_Ratio": financials["PE_Ratio"],
            "ProfitMargins": financials["ProfitMargins"],
            "SectorData": sector,
            "PressRelease": press["PressHeadline"]
        })

        print("\n=== Credit Intelligence Platform: Enhanced Prototype ===")
        print(f"Company: {name} ({ticker})")
        print(f"Financial Data: {financials}")
        print(f"Macro Data: {macro}")
        print(f"Sector Data: {sector}")
        print(f"News Sentiment: {news['Sentiment']:.2f}")
        print("Recent Headlines:")
        for h in news["Headlines"]:
            print(" -", h)
        print(f"Press Release: {press['PressHeadline']} (sent={press['PressSentiment']:.2f})")
        print(f"👉 Final Credit Score: {score:.2f}/100")

    # Save to CSV
    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    df = pd.DataFrame(results)
    df.to_csv("credit_scores_enhanced.csv", index=False)
    print("\n✅ Results saved to credit_scores_enhanced.csv")



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# st.title("✅ Streamlit is running")


st.set_page_config(page_title="Company Dashboard", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("all_company_data_MCA.csv")
    df.columns = df.columns.str.strip()  # clean whitespace in column names
    return df

df = load_data()

st.title("📊 Company Analytics Dashboard")
st.write("Data collected from Yahoo, AlphaVantage, SEC, MCA, World Bank, News, etc.")

# -------------------------------
# Sidebar - Company Selector
# -------------------------------
companies = df["Company"].dropna().unique().tolist()
selected_company = st.sidebar.selectbox("Select a company", companies)

company_df = df[df["Company"] == selected_company].iloc[0]

# -------------------------------
# Display Key Info
# -------------------------------
st.subheader(f"🏢 {selected_company}")
st.write(company_df)

# -------------------------------
# Plots
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Market Cap", f"{company_df['MarketCap']:,}")
    st.metric("P/E Ratio", round(company_df["PE_Ratio"], 2))
    st.metric("Debt to Equity", round(company_df["DebtToEquity"], 2))

with col2:
    st.metric("Profit Margins", round(company_df["ProfitMargins"], 2))
    st.metric("News Sentiment", round(company_df["News_sent"], 2))
    st.metric("PR Sentiment", round(company_df["PR_sent"], 2))

# -------------------------------
# Score Visualization
# -------------------------------
fig, ax = plt.subplots()
df.plot(kind="bar", x="Company", y="Score", ax=ax, color="skyblue", legend=False)
ax.set_ylabel("Score (0-100)")
st.pyplot(fig)

