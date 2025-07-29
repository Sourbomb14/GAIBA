import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import kaggle # Import the kaggle library

# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing & Finance Insight Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Key Configuration ---
st.sidebar.header("API Key Configuration")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password", help="Get your key from Google AI Studio.")

# Set the API key for the generative AI model
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    st.sidebar.warning("Please enter your Gemini API Key to use the chatbot and AI features.")

# --- Helper Functions ---

# Function to download and load marketing data from Kaggle
@st.cache_data
def load_marketing_data_from_kaggle(dataset_path="manishabhatt22/marketing-campaign-performance-dataset"):
    """
    Downloads the specified Kaggle dataset and loads it into a pandas DataFrame.
    Assumes Kaggle API credentials are set up (KAGGLE_USERNAME, KAGGLE_KEY env vars).
    """
    try:
        # Initialize Kaggle API
        # Kaggle API expects credentials in ~/.kaggle/kaggle.json or environment variables
        # For Streamlit Cloud, you'd set these as secrets.
        # For local, ensure KAGGLE_USERNAME and KAGGLE_KEY are set in your environment.
        kaggle.api.authenticate()

        # Define download path
        download_dir = "./kaggle_data"
        os.makedirs(download_dir, exist_ok=True)

        st.info(f"Attempting to download Kaggle dataset: {dataset_path} to {download_dir}")
        kaggle.api.dataset_download_files(dataset_path, path=download_dir, unzip=True)
        st.success("Kaggle dataset downloaded successfully!")

        # Assuming the CSV file name inside the zip is 'marketing_campaign_performance.csv'
        # You might need to adjust this based on the actual file name in the zip.
        # Let's list files in the downloaded directory to find the correct CSV.
        downloaded_files = os.listdir(download_dir)
        csv_file = None
        for f in downloaded_files:
            if f.endswith('.csv'):
                csv_file = os.path.join(download_dir, f)
                break

        if csv_file:
            df = pd.read_csv(csv_file)
            st.success(f"Loaded data from {os.path.basename(csv_file)}")

            # Perform necessary data cleaning/preparation for the specific dataset
            # Based on the dataset name, it might have columns like 'Spend', 'ConversionRate', 'ROI'
            # Let's ensure these columns exist or create dummy ones if not for consistency with the app.
            if 'Spend' not in df.columns:
                df['Spend'] = np.random.randint(50, 2000, len(df))
            if 'Clicks' not in df.columns:
                df['Clicks'] = np.random.randint(5, 100, len(df))
            if 'Conversions' not in df.columns:
                df['Conversions'] = np.random.randint(0, 5, len(df))
            if 'PurchaseValue' not in df.columns:
                df['PurchaseValue'] = np.random.randint(100, 5000, len(df))

            df['ConversionRate'] = (df['Conversions'] / df['Clicks'] * 100).fillna(0)
            df.loc[df['ConversionRate'] > 100, 'ConversionRate'] = 100 # Cap at 100%
            df['ROI'] = ((df['PurchaseValue'] * df['Conversions']) - df['Spend']) / df['Spend'] * 100
            df.loc[df['Spend'] == 0, 'ROI'] = 0 # Handle division by zero for ROI

            # Ensure CampaignType exists or create a dummy one
            if 'CampaignType' not in df.columns:
                 df['CampaignType'] = np.random.choice(['Email', 'Social Media', 'Direct Mail', 'TV Ad'], len(df), p=[0.35, 0.3, 0.2, 0.15])
            # Ensure Age and Income exist or create dummy ones
            if 'Age' not in df.columns:
                df['Age'] = np.random.randint(18, 70, len(df))
            if 'Income' not in df.columns:
                df['Income'] = np.random.randint(30000, 150000, len(df))
            if 'CustomerID' not in df.columns:
                df['CustomerID'] = range(1, len(df) + 1)


            return df
        else:
            st.error("No CSV file found in the downloaded Kaggle dataset.")
            return pd.DataFrame() # Return empty DataFrame on failure
    except Exception as e:
        st.error(f"Error loading marketing data from Kaggle: {e}")
        st.warning("Please ensure your Kaggle API credentials (KAGGLE_USERNAME and KAGGLE_KEY) are set as environment variables or in ~/.kaggle/kaggle.json.")
        return pd.DataFrame() # Return empty DataFrame on error

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
    # Attempt to get market cap from ticker info, handle cases where index is not a ticker
    try:
        if df.index.name: # Check if index has a name (which would be the ticker symbol)
            info = yf.Ticker(df.index.name).info
            metrics['Market Cap'] = info.get('marketCap')
        else:
            metrics['Market Cap'] = 'N/A'
    except Exception:
        metrics['Market Cap'] = 'N/A'
    return metrics

# Function to generate Instagram post content using Gemini
def generate_instagram_post(gemini_api_key, context_prompt):
    if not gemini_api_key:
        return "Please enter your Gemini API Key to generate Instagram content."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Generate an Instagram post for a marketing campaign. Provide:
        1. A catchy Instagram caption (aim for concise and impactful, max 2200 characters).
        2. 5-10 relevant and trending hashtags.
        3. A brief description of an ideal image or video concept for this post.

        Consider the following context for the campaign:
        {context_prompt}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating Instagram content: {e}. Please ensure your Gemini API key is valid and you have sufficient quota."

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

    # Load data from Kaggle
    marketing_df = load_marketing_data_from_kaggle()

    if not marketing_df.empty:
        # Slicers for Marketing Data
        st.sidebar.subheader("Marketing Data Filters")
        # Ensure 'CampaignType' exists in the loaded data
        if 'CampaignType' in marketing_df.columns:
            selected_campaign_type = st.sidebar.multiselect(
                "Filter by Campaign Type",
                options=marketing_df['CampaignType'].unique(),
                default=marketing_df['CampaignType'].unique()
            )
        else:
            selected_campaign_type = [] # No campaign type to filter by if column missing
            st.sidebar.warning("No 'CampaignType' column found in dataset. Filtering disabled.")

        # Ensure 'Age' and 'Income' exist
        min_age, max_age = 18, 70 # Default range if not found
        if 'Age' in marketing_df.columns:
            min_age, max_age = int(marketing_df['Age'].min()), int(marketing_df['Age'].max())
        age_range = st.sidebar.slider("Filter by Age Range", min_age, max_age, (min_age, max_age))

        min_income, max_income = 30000, 150000 # Default range if not found
        if 'Income' in marketing_df.columns:
            min_income, max_income = int(marketing_df['Income'].min()), int(marketing_df['Income'].max())
        income_range = st.sidebar.slider("Filter by Income Range", min_income, max_income, (min_income, max_income))

        # Apply filters
        filtered_marketing_df = marketing_df
        if selected_campaign_type and 'CampaignType' in filtered_marketing_df.columns:
            filtered_marketing_df = filtered_marketing_df[filtered_marketing_df['CampaignType'].isin(selected_campaign_type)]
        if 'Age' in filtered_marketing_df.columns:
            filtered_marketing_df = filtered_marketing_df[filtered_marketing_df['Age'].between(age_range[0], age_range[1])]
        if 'Income' in filtered_marketing_df.columns:
            filtered_marketing_df = filtered_marketing_df[filtered_marketing_df['Income'].between(income_range[0], income_range[1])]

        if filtered_marketing_df.empty:
            st.warning("No data matches the selected filters. Please adjust your selections.")
        else:
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
            if 'CampaignType' in filtered_marketing_df.columns:
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
            else:
                st.info("Campaign Type visualizations are not available as 'CampaignType' column is missing.")

            # Customer Demographics
            st.subheader("Customer Demographics")
            if 'Age' in filtered_marketing_df.columns:
                fig_age_dist = px.histogram(
                    filtered_marketing_df,
                    x='Age',
                    nbins=20,
                    title='Distribution of Customer Age',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_age_dist, use_container_width=True)
            else:
                st.info("Customer Age distribution is not available as 'Age' column is missing.")

            if 'Income' in filtered_marketing_df.columns:
                fig_income_dist = px.histogram(
                    filtered_marketing_df,
                    x='Income',
                    nbins=20,
                    title='Distribution of Customer Income',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_income_dist, use_container_width=True)
            else:
                st.info("Customer Income distribution is not available as 'Income' column is missing.")

            # AI-powered Marketing Campaign Recommendations
            st.subheader("💡 AI-Powered Marketing Campaign Recommendations")
            if gemini_api_key:
                if st.button("Generate Marketing Campaign Ideas"):
                    with st.spinner("Generating recommendations..."):
                        try:
                            # Prepare context for Gemini
                            campaign_summary = campaign_perf.to_string() if 'CampaignType' in filtered_marketing_df.columns else "No campaign type data available."
                            prompt = f"""
                            Based on the following marketing campaign performance data:
                            {campaign_summary}

                            And considering the filtered customer demographics:
                            - Age range: {age_range[0]}-{age_range[1]}
                            - Income range: ${income_range[0]:,.0f}-${income_range[1]:,.0f}
                            - Selected campaign types: {', '.join(selected_campaign_type) if selected_campaign_type else 'All available'}

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

            # Instagram Campaign Creator
            st.subheader("📸 Instagram Campaign Creator")
            st.markdown("Generate content for your next Instagram post based on your filtered marketing insights.")
            if gemini_api_key:
                if st.button("Generate Instagram Post Content"):
                    with st.spinner("Generating Instagram content..."):
                        context_prompt = f"""
                        Target Audience Age: {age_range[0]}-{age_range[1]}
                        Target Audience Income: ${income_range[0]:,.0f}-${income_range[1]:,.0f}
                        Selected Campaign Types: {', '.join(selected_campaign_type) if selected_campaign_type else 'All available'}
                        Overall Marketing Performance Summary:\n{campaign_perf.to_string() if 'CampaignType' in filtered_marketing_df.columns else 'No campaign type data for summary.'}
                        """
                        instagram_content = generate_instagram_post(gemini_api_key, context_prompt)
                        st.markdown(instagram_content)

                        st.markdown("---")
                        st.info("**Note on Instagram Posting:** Direct posting to Instagram from a web application requires complex API setup, including Facebook Developer App registration, specific permissions, and user authentication flows. This feature provides the content, which you can then manually post.")
                        st.button("Simulate Post to Instagram (Placeholder)", disabled=True, help="This button is a placeholder. Real Instagram integration requires advanced API setup.")
            else:
                st.info("Enter your Gemini API Key in the sidebar to generate Instagram post content.")
    else:
        st.error("Could not load marketing data. Please check Kaggle API credentials and dataset availability.")


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
    You can also ask me to **"create an Instagram campaign"** to get content ideas.
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

                    # Check if the user is asking for an Instagram campaign
                    if "instagram campaign" in prompt.lower() or "create post" in prompt.lower() or "social media campaign" in prompt.lower():
                        # For chatbot, we don't have filtered marketing data directly.
                        # We'll ask Gemini to generate general Instagram content or based on the prompt's context.
                        context_for_instagram_bot = f"User request: '{prompt}'. Generate an Instagram post based on this."
                        instagram_content = generate_instagram_post(gemini_api_key, context_for_instagram_bot)
                        full_response = instagram_content + "\n\n"
                        full_response += "**Note on Instagram Posting:** Direct posting to Instagram from a web application requires complex API setup. This provides the content, which you can then manually post."
                    else:
                        # General chat response
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
