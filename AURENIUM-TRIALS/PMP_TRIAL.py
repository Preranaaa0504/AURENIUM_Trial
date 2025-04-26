import streamlit as st
import requests
import pandas as pd
import json
import re
from datetime import datetime

# Constants
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'

st.title('Financial Insights Assistant')

# Sidebar for stock selection
with st.sidebar:
    st.header('Stock Selection')
    symbol = st.text_input('Ticker Symbol:', value='AAPL')
    st.caption('Example: AAPL, MSFT, GOOGL, AMZN')

# Cache function to avoid redundant API calls
@st.cache_data(ttl=3600)
def fetch_financial_data(endpoint, ticker):
    url = f'{BASE_URL}/{API_VERSION}/{endpoint}/{ticker}?apikey={API_KEY}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

# Function to get stock quote
def get_stock_quote(ticker):
    return fetch_financial_data('quote', ticker)

# Function to get income statement
def get_income_statement(ticker):
    return fetch_financial_data('income-statement', ticker)

# Function to get balance sheet
def get_balance_sheet(ticker):
    return fetch_financial_data('balance-sheet-statement', ticker)

# Function to get cash flow statement
def get_cash_flow(ticker):
    return fetch_financial_data('cash-flow-statement', ticker)

# Function to get key metrics
def get_key_metrics(ticker):
    return fetch_financial_data('key-metrics', ticker)

# Function to get ratios
def get_ratios(ticker):
    return fetch_financial_data('ratios', ticker)

# Function to get company profile
def get_company_profile(ticker):
    return fetch_financial_data('profile', ticker)

# Function to get stock price history
def get_price_history(ticker):
    return fetch_financial_data('historical-price-full', ticker)

# Function to get analyst ratings
def get_ratings(ticker):
    return fetch_financial_data('rating', ticker)

# Function to format financial numbers
def format_number(num):
    if num is None:
        return "N/A"
    
    if isinstance(num, str):
        try:
            num = float(num)
        except:
            return num
    
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

# Function to process user query and generate response
def process_query(query, ticker):
    query = query.lower()
    
    # Initialize data sources for potential use (lazy loading)
    profile_data = None
    quote_data = None
    income_data = None
    balance_data = None
    cash_flow_data = None
    metrics_data = None
    ratios_data = None
    ratings_data = None
    
    response = ""
    
    # General company information
    if any(term in query for term in ["who", "what", "about", "company", "business", "overview"]):
        profile_data = profile_data or get_company_profile(ticker)
        if profile_data and len(profile_data) > 0:
            company = profile_data[0]
            response = f"{company.get('companyName', ticker)} is a {company.get('industry', 'company')} in the {company.get('sector', '')} sector. "
            response += f"They are headquartered in {company.get('city', '')}, {company.get('state', '')}, {company.get('country', '')}. "
            description = company.get('description', '')
            if description:
                response += f"\n\n{description[:200]}..." if len(description) > 200 else description
        else:
            response = f"I couldn't find company information for {ticker}."
    
    # Current stock price and basic metrics
    elif any(term in query for term in ["price", "worth", "value", "stock", "trading at"]):
        quote_data = quote_data or get_stock_quote(ticker)
        if quote_data and len(quote_data) > 0:
            quote = quote_data[0]
            current_price = quote.get('price', 'N/A')
            change = quote.get('change', 0)
            change_percent = quote.get('changesPercentage', 0)
            
            response = f"The current price of {ticker} is ${current_price}. "
            
            if change is not None:
                direction = "up" if change > 0 else "down"
                response += f"It's {direction} {abs(change):.2f} points ({abs(change_percent):.2f}%) today. "
            
            response += f"\n\nTrading Range: ${quote.get('dayLow', 'N/A')} - ${quote.get('dayHigh', 'N/A')} today"
            response += f"\nMarket Cap: {format_number(quote.get('marketCap', 'N/A'))}"
            response += f"\nP/E Ratio: {quote.get('pe', 'N/A')}"
        else:
            response = f"I couldn't find current stock price information for {ticker}."
    
    # Financial performance
    elif any(term in query for term in ["revenue", "sales", "earning", "profit", "income", "performance"]):
        income_data = income_data or get_income_statement(ticker)
        
        if income_data and len(income_data) > 0:
            latest = income_data[0]
            previous = income_data[1] if len(income_data) > 1 else None
            
            revenue = latest.get('revenue', 0)
            net_income = latest.get('netIncome', 0)
            period = latest.get('date', '')
            
            response = f"For the period ending {period}, {ticker} reported:"
            response += f"\n- Revenue: {format_number(revenue)}"
            response += f"\n- Net Income: {format_number(net_income)}"
            
            if previous:
                rev_growth = ((revenue - previous.get('revenue', 0)) / previous.get('revenue', 1)) * 100
                income_growth = ((net_income - previous.get('netIncome', 0)) / previous.get('netIncome', 1)) * 100
                
                response += f"\n\nCompared to the previous period:"
                response += f"\n- Revenue Growth: {rev_growth:.2f}%"
                response += f"\n- Net Income Growth: {income_growth:.2f}%"
        else:
            response = f"I couldn't find financial performance data for {ticker}."
    
    # Financial health and balance sheet
    elif any(term in query for term in ["balance", "debt", "asset", "liability", "health", "equity"]):
        balance_data = balance_data or get_balance_sheet(ticker)
        
        if balance_data and len(balance_data) > 0:
            latest = balance_data[0]
            
            total_assets = latest.get('totalAssets', 0)
            total_liabilities = latest.get('totalLiabilities', 0)
            equity = latest.get('totalStockholdersEquity', 0)
            period = latest.get('date', '')
            
            response = f"As of {period}, {ticker}'s balance sheet shows:"
            response += f"\n- Total Assets: {format_number(total_assets)}"
            response += f"\n- Total Liabilities: {format_number(total_liabilities)}"
            response += f"\n- Stockholders' Equity: {format_number(equity)}"
            
            if total_assets > 0:
                debt_to_assets = (total_liabilities / total_assets) * 100
                response += f"\n\nThe debt-to-assets ratio is {debt_to_assets:.2f}%, "
                
                if debt_to_assets < 30:
                    response += "which indicates a strong financial position with low leverage."
                elif debt_to_assets < 60:
                    response += "which suggests a moderate level of leverage."
                else:
                    response += "which indicates significant leverage that may present financial risks."
        else:
            response = f"I couldn't find balance sheet information for {ticker}."
    
    # Cash flow
    elif any(term in query for term in ["cash flow", "cash", "free cash", "dividend", "capex", "capital expenditure"]):
        cash_flow_data = cash_flow_data or get_cash_flow(ticker)
        
        if cash_flow_data and len(cash_flow_data) > 0:
            latest = cash_flow_data[0]
            
            operating_cf = latest.get('operatingCashFlow', 0)
            investing_cf = latest.get('netCashUsedForInvestingActivities', 0)
            financing_cf = latest.get('netCashUsedProvidedByFinancingActivities', 0)
            period = latest.get('date', '')
            
            response = f"For the period ending {period}, {ticker}'s cash flows were:"
            response += f"\n- Operating Cash Flow: {format_number(operating_cf)}"
            response += f"\n- Investing Cash Flow: {format_number(investing_cf)}"
            response += f"\n- Financing Cash Flow: {format_number(financing_cf)}"
            
            if 'freeCashFlow' in latest:
                free_cf = latest.get('freeCashFlow', 0)
                response += f"\n- Free Cash Flow: {format_number(free_cf)}"
                
                if free_cf > 0:
                    response += "\n\nPositive free cash flow indicates the company is generating more cash than it's spending on capital expenditures."
                else:
                    response += "\n\nNegative free cash flow suggests the company is investing heavily in its future growth or facing challenges in cash generation."
        else:
            response = f"I couldn't find cash flow information for {ticker}."
    
    # Valuation and ratios
    elif any(term in query for term in ["pe", "ratio", "valuation", "undervalued", "overvalued", "p/e", "eps", "ebitda"]):
        ratios_data = ratios_data or get_ratios(ticker)
        quote_data = quote_data or get_stock_quote(ticker)
        
        if ratios_data and len(ratios_data) > 0 and quote_data and len(quote_data) > 0:
            latest_ratio = ratios_data[0]
            quote = quote_data[0]
            
            pe_ratio = quote.get('pe', 'N/A')
            pb_ratio = latest_ratio.get('priceToBookRatio', 'N/A')
            ps_ratio = latest_ratio.get('priceToSalesRatio', 'N/A')
            dividend_yield = latest_ratio.get('dividendYield', 'N/A')
            
            response = f"{ticker}'s current valuation metrics:"
            response += f"\n- Price-to-Earnings (P/E): {pe_ratio}"
            response += f"\n- Price-to-Book (P/B): {pb_ratio}"
            response += f"\n- Price-to-Sales (P/S): {ps_ratio}"
            response += f"\n- Dividend Yield: {dividend_yield if dividend_yield != 'N/A' else 'N/A'}"
            
            # Add some interpretation
            if pe_ratio != 'N/A' and isinstance(pe_ratio, (int, float)):
                if pe_ratio < 15:
                    response += "\n\nThe P/E ratio suggests the stock may be undervalued compared to average market valuations."
                elif pe_ratio > 25:
                    response += "\n\nThe P/E ratio is relatively high, which might indicate investor optimism about future growth."
                else:
                    response += "\n\nThe P/E ratio is in a moderate range compared to historical market averages."
        else:
            response = f"I couldn't find valuation metrics for {ticker}."
            
    # Analyst recommendations
    elif any(term in query for term in ["recommend", "analyst", "rating", "buy", "sell", "hold", "target"]):
        ratings_data = ratings_data or get_ratings(ticker)
        
        if ratings_data and len(ratings_data) > 0:
            rating = ratings_data[0]
            
            score = rating.get('ratingScore', 'N/A')
            recommendation = rating.get('recommendation', 'N/A')
            
            response = f"For {ticker}, the current analyst consensus is: {recommendation}."
            response += f"\nRating Score: {score}/5"
            
            details = []
            if 'ratingDetailsDCFScore' in rating:
                details.append(f"DCF Score: {rating.get('ratingDetailsDCFScore')}/5")
            if 'ratingDetailsROEScore' in rating:
                details.append(f"ROE Score: {rating.get('ratingDetailsROEScore')}/5")
            if 'ratingDetailsROAScore' in rating:
                details.append(f"ROA Score: {rating.get('ratingDetailsROAScore')}/5")
            if 'ratingDetailsDEScore' in rating:
                details.append(f"D/E Score: {rating.get('ratingDetailsDEScore')}/5")
            if 'ratingDetailsPEScore' in rating:
                details.append(f"P/E Score: {rating.get('ratingDetailsPEScore')}/5")
            if 'ratingDetailsPBScore' in rating:
                details.append(f"P/B Score: {rating.get('ratingDetailsPBScore')}/5")
                
            if details:
                response += "\n\nBreakdown of the rating:"
                response += "\n- " + "\n- ".join(details)
        else:
            response = f"I couldn't find analyst recommendations for {ticker}."
    
    # Default response for unknown queries
    else:
        response = f"I'm not sure what financial information about {ticker} you're looking for. You can ask about:"
        response += "\n- Company information and overview"
        response += "\n- Current stock price and trading data"
        response += "\n- Revenue, earnings, and financial performance"
        response += "\n- Balance sheet and financial health"
        response += "\n- Cash flow details"
        response += "\n- Valuation metrics and ratios"
        response += "\n- Analyst recommendations"
    
    return response

# Main layout
st.header(f"Ask me about {symbol}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if company data exists
quote_data = get_stock_quote(symbol)
if not quote_data:
    st.warning(f"Could not find data for ticker symbol '{symbol}'. Please verify the ticker is correct.")

# Accept user input
if prompt := st.chat_input("What would you like to know about this stock?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = process_query(prompt, symbol)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add basic instructions for first-time users
if not st.session_state.messages:
    st.info("👋 Hello! I'm your financial data assistant. Enter a ticker symbol in the sidebar and ask me questions about that company's financials.")
    st.markdown("""
    **Example questions you can ask:**
    - What does this company do?
    - What's the current stock price?
    - How has their revenue been performing?
    - What's their financial health like?
    - How's their cash flow situation?
    - Is the stock undervalued?
    - What do analysts recommend?
    """)

# Add a data disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("Data provided by Financial Modeling Prep API. The information is for educational purposes only and not financial advice.")