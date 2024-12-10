import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import auc, mean_squared_error, roc_curve
import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide",  # Full-width layout
)


# Fresh CSS Styling
st.markdown("""
    <style>
    /* Background for the entire app */
    .main {
        background-color: #0e1117;  /* Uniform dark color for the app */
        color: #cccccc;  /* Light gray text for readability */
        padding: 20px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0e1117 !important;  /* Dark color for sidebar */
        color: #cccccc !important;  /* Light gray text */
    }

    /* Sidebar input box */
    .css-1w3z6i7 {
        background-color: #333333 !important;  /* Slightly lighter input background */
        color: #cccccc !important;  /* Light gray text */
    }

    /* Button styling */
    .stButton>button {
        background-color: #333333;  /* Dark button */
        color: #cccccc;  /* Light gray text */
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
    }

    /* Button hover effect */
    .stButton>button:hover {
        background-color: #4d4d4d;  /* Slightly lighter on hover */
        color: #cccccc;  /* Keep text color consistent */
    }

    /* Table styling */
    .stDataFrame {
        background-color: #0e1117;  /* Dark background for the table */
        color: #cccccc;  /* Light gray text for table */
        border-radius: 8px;
        border: 1px solid #333333;
        padding: 10px;
    }

    /* Header styling - same as background */
    h1, h2, h3 {
        color: #cccccc;  /* Light gray text for headers */
        font-family: 'Arial', sans-serif;
    }

    /* Padding and border for better visuals */
    .block-container {
        padding: 2rem;
        background-color: #0e1117;  /* Dark background for content */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


API_KEY = '3774ae55ff63444aa99e6c2298574484'

# Function to fetch news articles
def fetch_news(query="stocks"):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        news_data = response.json()
        
        # Debug: Print the full response if 'articles' key is missing
        if 'articles' not in news_data:
            st.error("Error fetching news: 'articles' key missing.")
            st.write(news_data)  # Print the full response for debugging
            return []
        
        return news_data['articles']
    else:
        st.error(f"Failed to fetch news. Status code: {response.status_code}")
        return []



# List of stock tickers for users to select from
st.sidebar.header("Select Stock")
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'FB', 'NVDA', 'DIS', 'BABA', 'V', 'JNJ', 'JPM', 'WMT', 'PG', 'MA', 'KO', 'PEP', 'CSCO', 'INTC']
selected_stock = st.sidebar.selectbox('Select a stock to predict', stocks)

# Date
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-09-19'))


# News Section in the Sidebar
st.sidebar.subheader("Live News Updates")
news_query = "stock market"

# Fetch and display news articles in the sidebar
if news_query:
    news_articles = fetch_news(news_query)
    
    if not news_articles:
        st.sidebar.write("No news found or an error occurred.")
    
    for article in news_articles[:5]:  # Display only top 5 articles for simplicity
        st.sidebar.write(f"**{article['title']}**")
        st.sidebar.write(f"Published at: {article['publishedAt']}")
        st.sidebar.write(f"[Read more]({article['url']})")
        st.sidebar.write("---")

# User input for future prediction days
st.subheader(f'Prediction Range (in Days)')
future_days = st.slider('', 1, 60, 30)

# Load the stock data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

data = load_data(selected_stock)



# Display raw data
st.subheader(f'Recent data for {selected_stock}')
st.dataframe(data.tail(10), use_container_width=True)

col1, col2 = st.columns(2)

# Candlestick
with col1:
    st.subheader(f'Candlestick Chart of {selected_stock}')
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])
    fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    xaxis_rangeslider_visible=True,  # Enable the range slider for better zooming
    template="plotly_dark",  # Dark theme
    plot_bgcolor='#0e1117',  # Background color
    paper_bgcolor='#0e1117',  # Page background color
    font=dict(color='white')  # Text color
    )
    st.plotly_chart(fig)


# RSI calculation function
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Add RSI to the data
data['RSI'] = calculate_rsi(data)

# Handle missing values by dropping rows with NaN values
data.dropna(inplace=True)

# Prepare the data
data = data[['Close', 'RSI']]
data['Prediction'] = data[['Close']].shift(-future_days)

# Create independent dataset (X) and dependent dataset (y)
X = np.array(data.drop(['Prediction'], axis=1))[:-future_days]
y = np.array(data['Prediction'])[:-future_days]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model and predict stock prices
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

# Predict the next 'n' days
X_future = np.array(data.drop(['Prediction'], axis=1))[-future_days:]
future_predictions = model.predict(X_future)

# Visualize the predicted stock prices
st.subheader(f'Predicted Prices for {selected_stock} over next {future_days} days')

# Ensure the future index is created for predictions
future_index = pd.date_range(start=data.index[-1], periods=future_days + 1, freq='B')[1:]  # Creating future dates

# Plot historical actual prices and future predicted prices separately
fig, ax = plt.subplots(figsize=(10, 6))

# Set white axes and labels
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

# Plot the actual closing prices
ax.plot(data['Close'], label='Actual Prices', color='blue')

# Plot predicted future prices on the new future date range
ax.plot(future_index, future_predictions, label='Predicted Prices', color='red', linestyle='--')

ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
st.pyplot(fig)


# Moving Averages
data['MA100'] = data['Close'].rolling(window=100).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()





# Closing Price vs Time
with col2:
    st.subheader(f'Closing Price vs Time for {selected_stock}')
    st.write('<div style="height: 60px;"></div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.plot(data['Close'], label='Closing Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    fig1.patch.set_alpha(0)
    ax1.patch.set_alpha(0)
    st.pyplot(fig1)

col3, col4 = st.columns(2) 

# Visualize the RSI
with col3:
    st.subheader(f'RSI for {selected_stock}')
    st.write('<div style="height: 85px;"></div>', unsafe_allow_html=True)
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
    ax_rsi.spines['bottom'].set_color('white')
    ax_rsi.spines['left'].set_color('white')
    ax_rsi.spines['top'].set_color('white')
    ax_rsi.spines['right'].set_color('white')
    ax_rsi.tick_params(axis='x', colors='white')
    ax_rsi.tick_params(axis='y', colors='white')
    ax_rsi.xaxis.label.set_color('white')
    ax_rsi.yaxis.label.set_color('white')
    ax_rsi.plot(data['RSI'], label='RSI', color='blue')
    ax_rsi.axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought (70)')
    ax_rsi.axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold (30)')
    ax_rsi.set_xlabel('Date')
    ax_rsi.set_ylabel('RSI')
    ax_rsi.legend()
    fig_rsi.patch.set_alpha(0)
    ax_rsi.patch.set_alpha(0) 
    st.pyplot(fig_rsi)

# Closing Price vs Time with 100-day and 200-day MA
with col4:
    st.subheader(f'Closing Price vs Time with 100-Day and 200-Day MA for {selected_stock}')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.plot(data['Close'], label='Closing Price')
    ax2.plot(data['MA100'], label='100-Day Moving Average', color='orange')
    ax2.plot(data['MA200'], label='200-Day Moving Average', color='green')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()    
    fig2.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    st.pyplot(fig2)

import pandas as pd

st.subheader('Select multiple stocks to compare')
selected_stocks = st.multiselect('', stocks, default=['AAPL', 'GOOGL'])

# Create an empty DataFrame to store the closing prices of all selected stocks
combined_data = pd.DataFrame()

# Loop through selected stocks and add the closing prices to the combined DataFrame
for stock in selected_stocks:
    data = load_data(stock)
    combined_data[stock] = data['Close']  # Store closing prices for each stock

# Plot the combined data on the same chart
st.subheader('Closing Price Comparison')
st.line_chart(combined_data)

# Preparing Data for Classification
data['RSI'] = calculate_rsi(data)
data['Price_Change'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Create target variable as binary outcome (1 if price goes up, 0 if it goes down)
data['Target'] = (data['Price_Change'] > 0).astype(int)

# Define features and target for classification
X = data[['Close', 'RSI']][:-future_days]  # Exclude last future_days for prediction
y = data['Target'][:-future_days]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classification model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict probabilities
y_proba = classifier.predict_proba(X_test)[:, 1]

# Calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
st.write(f"AUC Score: {roc_auc:.2f}")

# Plot ROC Curve
st.subheader(f'ROC Curve for {selected_stock}')
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the ROC curve
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Set the labels and title with white color
ax.set_xlabel('False Positive Rate', color='white')
ax.set_ylabel('True Positive Rate', color='white')
ax.set_title('Receiver Operating Characteristic (ROC) Curve', color='white')

# Set the color of the ticks to white
ax.tick_params(axis='x', colors='white')  # X-axis ticks
ax.tick_params(axis='y', colors='white')  # Y-axis ticks

# Add the legend with white text
ax.legend(loc='lower right', facecolor='black', edgecolor='white', fontsize=10, labelcolor='white')

# Remove background color
fig.patch.set_facecolor('none')  # Make the figure background transparent
ax.set_facecolor('none')  # Make the axes background transparent

# Set gridlines to white for better visibility
ax.grid(True, color='white', linestyle='--', linewidth=0.5)

# Plot the figure
st.pyplot(fig)

