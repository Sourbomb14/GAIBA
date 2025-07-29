import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai
import os
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing & Finance Insight Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Key Configuration ---
# It's recommended to use st.secrets for production, but for quick demo,
# we'll use text input or environment variables.
st.sidebar.header("API Key Configuration")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password", help="Get your key from Google AI Studio.")

# Set the API key for the generative AI model
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    st.sidebar.warning("Please enter your Gemini API Key to use the chatbot and AI features.")

# --- Helper Functions ---

# Function to generate dummy marketing data (since direct Kaggle API integration is complex for a simple demo)
@st.cache_data
def generate_marketing_data(num_records=1000):
    np.random.seed(42) # for reproducibility
    data = {
        'CustomerID': range(1, num_records + 1),
        'Age': np.random.randint(18, 70, num_records),
        'Income': np.random.randint(30000, 150000, num_records),
        'Education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], num_records, p=[0.25, 0.4, 0.2, 0.15]),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], num_records, p=[0.3, 0.5, 0.15, 0.05]),
        'CampaignType': np.random.choice(['Email', 'Social Media', 'Direct Mail', 'TV Ad'], num_records, p=[0.35, 0.3, 0.2, 0.15]),
        'Spend': np.random.randint(50, 2000, num_records),
        'Clicks': np.random.randint(5, 100, num_records),
        'Conversions': np.random.randint(0, 5, num_records), # Number of conversions
        'PurchaseValue': np.random.randint(100, 5000, num_records)
    }
    df = pd.DataFrame(data)
    # Simulate conversion rate based on spend and campaign type
    df['ConversionRate'] = (df['Conversions'] / df['Clicks'] * 100).fillna(0)
    df.loc[df['ConversionRate'] > 100, 'ConversionRate'] = 100 # Cap at 100%
    df['ROI'] = ((df['PurchaseValue'] * df['Conversions']) - df['Spend']) / df['Spend'] * 100
    df.loc[df['Spend'] == 0, 'ROI'] = 0 # Handle division by zero for ROI
    return df

# Function to fetch financial data
@st.cache_data
def get_stock_data(ticker_symbol, period='1y'):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            st.error(f"No data found for {ticker_symbol}. Please check the ticker symbol.")
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

# Function to calculate financial metrics
def calculate_financial_metrics(df):
    if df is None or df.empty:
        return {}
    metrics = {}
    metrics['Current Price'] = df['Close'].iloc[-1]
    metrics['Previous Close'] = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1]
    metrics['Daily Change (%)'] = ((metrics['Current Price'] - metrics['Previous Close']) / metrics['Previous Close']) * 100
    metrics['52 Week High'] = df['High'].max()
    metrics['52 Week Low'] = df['Low'].min()
    metrics['Average Volume'] = df['Volume'].mean()
    metrics['Market Cap'] = yf.Ticker(df.index.name).info.get('marketCap') if df.index.name else 'N/A' # Get market cap from info
    return metrics

# --- Main Application Layout ---
st.title("📈 Marketing & Finance Insight Hub")
st.markdown("""
Welcome to your integrated dashboard for marketing performance and financial market analysis.
Explore key metrics, visualize trends, and leverage AI for strategic recommendations.
""")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Marketing Analytics", "Financial Analytics", "AI Chatbot"])

# --- Marketing Analytics Section ---
if section == "Marketing Analytics":
    st.header("📊 Marketing Analytics Dashboard")
    st.markdown("Analyze your marketing campaign performance and customer segments.")

    marketing_df = generate_marketing_data()

    if marketing_df is not None:
        # Slicers for Marketing Data
        st.sidebar.subheader("Marketing Data Filters")
        selected_campaign_type = st.sidebar.multiselect(
            "Filter by Campaign Type",
            options=marketing_df['CampaignType'].unique(),
            default=marketing_df['CampaignType'].unique()
        )
        min_age, max_age = int(marketing_df['Age'].min()), int(marketing_df['Age'].max())
        age_range = st.sidebar.slider("Filter by Age Range", min_age, max_age, (min_age, max_age))
        min_income, max_income = int(marketing_df['Income'].min()), int(marketing_df['Income'].max())
        income_range = st.sidebar.slider("Filter by Income Range", min_income, max_income, (min_income, max_income))

        filtered_marketing_df = marketing_df[
            (marketing_df['CampaignType'].isin(selected_campaign_type)) &
            (marketing_df['Age'].between(age_range[0], age_range[1])) &
            (marketing_df['Income'].between(income_range[0], income_range[1]))
        ]

        st.subheader("Key Marketing Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spend", f"${filtered_marketing_df['Spend'].sum():,.2f}")
        with col2:
            st.metric("Avg. Conversion Rate", f"{filtered_marketing_df['ConversionRate'].mean():.2f}%")
        with col3:
            st.metric("Avg. ROI", f"{filtered_marketing_df['ROI'].mean():.2f}%")
        with col4:
            st.metric("Total Customers", f"{filtered_marketing_df['CustomerID'].nunique()}")

        st.subheader("Marketing Performance Visualizations")

        # Campaign Type Performance
        campaign_perf = filtered_marketing_df.groupby('CampaignType').agg(
            AvgSpend=('Spend', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean'),
            AvgROI=('ROI', 'mean'),
            TotalCustomers=('CustomerID', 'nunique')
        ).reset_index()

        fig_campaign_cr = px.bar(
            campaign_perf,
            x='CampaignType',
            y='AvgConversionRate',
            title='Average Conversion Rate by Campaign Type',
            labels={'AvgConversionRate': 'Avg. Conversion Rate (%)'},
            color='AvgConversionRate',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_campaign_cr, use_container_width=True)

        fig_campaign_roi = px.bar(
            campaign_perf,
            x='CampaignType',
            y='AvgROI',
            title='Average ROI by Campaign Type',
            labels={'AvgROI': 'Avg. ROI (%)'},
            color='AvgROI',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_campaign_roi, use_container_width=True)

        # Customer Demographics
        st.subheader("Customer Demographics")
        fig_age_dist = px.histogram(
            filtered_marketing_df,
            x='Age',
            nbins=20,
            title='Distribution of Customer Age',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_age_dist, use_container_width=True)

        fig_income_dist = px.histogram(
            filtered_marketing_df,
            x='Income',
            nbins=20,
            title='Distribution of Customer Income',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_income_dist, use_container_width=True)

        # AI-powered Marketing Campaign Recommendations
        st.subheader("💡 AI-Powered Marketing Campaign Recommendations")
        if gemini_api_key:
            if st.button("Generate Marketing Campaign Ideas"):
                with st.spinner("Generating recommendations..."):
                    try:
                        # Prepare context for Gemini
                        campaign_summary = campaign_perf.to_string()
                        prompt = f"""
                        Based on the following marketing campaign performance data:
                        {campaign_summary}

                        And considering the filtered customer demographics:
                        - Age range: {age_range[0]}-{age_range[1]}
                        - Income range: ${income_range[0]:,.0f}-${income_range[1]:,.0f}
                        - Selected campaign types: {', '.join(selected_campaign_type)}

                        Please provide actionable marketing campaign recommendations. Focus on strategies to improve conversion rates and ROI for the most promising segments or campaigns, and suggest new creative angles.
                        Provide the recommendations in bullet points, with a brief explanation for each.
                        """
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
                        st.info("Please ensure your Gemini API key is valid and you have sufficient quota.")
        else:
            st.info("Enter your Gemini API Key in the sidebar to generate marketing campaign ideas.")

# --- Financial Analytics Section ---
elif section == "Financial Analytics":
    st.header("💰 Financial Analytics Dashboard")
    st.markdown("Track stock performance and key financial metrics.")

    ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, GOOGL, MSFT)", "AAPL").upper()
    period_options = {
        "1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", "3 Months": "3mo",
        "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y",
        "10 Years": "10y", "YTD": "ytd", "Max": "max"
    }
    selected_period_name = st.selectbox("Select Data Period", list(period_options.keys()), index=5)
    selected_period = period_options[selected_period_name]

    if ticker_symbol:
        stock_df = get_stock_data(ticker_symbol, selected_period)

        if stock_df is not None:
            st.subheader(f"Key Financial Indicators for {ticker_symbol}")
            metrics = calculate_financial_metrics(stock_df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${metrics.get('Current Price', 0):,.2f}")
            with col2:
                st.metric("Daily Change", f"{metrics.get('Daily Change (%)', 0):.2f}%")
            with col3:
                st.metric("52 Week High", f"${metrics.get('52 Week High', 0):,.2f}")
            with col4:
                st.metric("52 Week Low", f"${metrics.get('52 Week Low', 0):,.2f}")

            st.metric("Market Cap", f"${metrics.get('Market Cap', 'N/A'):,}" if isinstance(metrics.get('Market Cap'), (int, float)) else metrics.get('Market Cap', 'N/A'))


            st.subheader(f"Stock Price Chart for {ticker_symbol}")
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(
                x=stock_df.index,
                open=stock_df['Open'],
                high=stock_df['High'],
                low=stock_df['Low'],
                close=stock_df['Close'],
                name='Price'
            ))
            fig_price.update_layout(
                title=f'{ticker_symbol} Stock Price',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False # Hide the range slider for cleaner view
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # Volume Chart
            st.subheader(f"Volume Chart for {ticker_symbol}")
            fig_volume = px.bar(
                stock_df,
                x=stock_df.index,
                y='Volume',
                title=f'{ticker_symbol} Trading Volume',
                labels={'Volume': 'Volume'},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_volume, use_container_width=True)

# --- AI Chatbot Section ---
elif section == "AI Chatbot":
    st.header("💬 AI Chatbot (Powered by Gemini)")
    st.markdown("""
    Ask me anything about marketing, finance, or general knowledge!
    **Note**: For voice input, you would typically integrate with a Speech-to-Text (STT) service or library (e.g., `SpeechRecognition` in Python, or browser's Web Speech API for web apps). This demo uses text input for simplicity.
    """)

    if gemini_api_key:
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response from Gemini
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    chat = model.start_chat(history=[]) # Start a new chat for each interaction for simplicity
                                                        # For persistent chat, pass st.session_state.messages as history
                    response = chat.send_message(prompt, stream=True)
                    for chunk in response:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"Error: {e}. Please check your API key and try again."
                    st.error(full_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.info("Please enter your Gemini API Key in the sidebar to use the chatbot.")

st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit, yfinance, Plotly, and Google Gemini API.")
