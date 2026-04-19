import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
from utils import (
    load_model_and_scaler,
    fetch_binance_or_coingecko,
    get_latest_sequence,
    predict_price,
    calculate_change_percent,
    format_price,
    get_current_price,
    prepare_chart_data
)

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="BNB Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .up {
        color: #28a745 !important;
    }
    .down {
        color: #dc3545 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Session State Initialization ====================

@st.cache_resource
def init_model():
    """Load and cache ONNX model and scaler"""
    try:
        session, scaler = load_model_and_scaler()
        return session, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def init_session_state():
    """Initialize session state variables"""
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'update_interval' not in st.session_state:
        st.session_state.update_interval = 60  # seconds
    if 'data_source' not in st.session_state:
        st.session_state.data_source = 'unknown'


# ==================== Main Application ====================

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Load model and scaler
    session, scaler = init_model()
    if session is None or scaler is None:
        st.error("❌ Failed to load model. Please check if model files exist.")
        return
    
    # ==================== Header ====================
    st.title("📈 BNB Price Prediction")
    st.markdown("""
    Real-time BNB/USDT price prediction using CNN-LSTM Neural Network with Genetic Algorithm optimization.
    Predicts next day's closing price based on 30 days of historical data.
    
    **Model**: ONNX format (converted from TensorFlow Keras)
    **Data Source**: Binance API (with CoinGecko fallback for restricted regions)
    """)
    
    # ==================== Sidebar ====================
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_update = None
            st.rerun()
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto-refresh per minute", value=True)
        
        # Display last update time
        st.markdown("---")
        if st.session_state.last_update:
            time_ago = datetime.datetime.now() - st.session_state.last_update
            minutes = int(time_ago.total_seconds() / 60)
            st.metric("Last Update", f"{minutes} min ago" if minutes > 0 else "Just now")
        else:
            st.metric("Last Update", "Never")
        
        # Display data source
        source_icon = "🌐" if st.session_state.data_source == 'coingecko' else "📊"
        st.metric("Data Source", f"{source_icon} {st.session_state.data_source.title()}")
        
        # Display app info
        st.markdown("---")
        st.info("""
        **Model Info:**
        - Architecture: CNN-LSTM
        - Optimization: Genetic Algorithm
        - Input: 30 days historical data
        - Output: Next day prediction
        
        **Data Source:**
        - Binance API (BNB/USDT)
        - Update: Per minute
        """)
    
    # ==================== Data Fetching & Prediction ====================
    
    # Check if we need to update data
    should_update = False
    if st.session_state.data is None:
        should_update = True
    elif st.session_state.last_update:
        time_elapsed = (datetime.datetime.now() - st.session_state.last_update).total_seconds()
        if time_elapsed >= st.session_state.update_interval and auto_refresh:
            should_update = True
    
    if should_update:
        with st.spinner("📊 Fetching data from Binance/CoinGecko..."):
            try:
                # Fetch data (tries Binance, fallback to CoinGecko)
                df, source = fetch_binance_or_coingecko(
                    symbol_binance='BNBUSDT',
                    symbol_coingecko='binancecoin',
                    days=30
                )
                
                # Get latest sequence for prediction
                X_recent = get_latest_sequence(df, scaler, seq_length=30)
                
                # Make prediction
                pred = predict_price(X_recent, session, scaler)
                
                # Store in session state
                st.session_state.data = df
                st.session_state.prediction = pred
                st.session_state.last_update = datetime.datetime.now()
                st.session_state.data_source = source
                st.session_state.error_message = None
                
                st.rerun()
            
            except Exception as e:
                st.session_state.error_message = str(e)
                st.error(f"❌ Error: {str(e)}")
    
    # Use cached data if available
    if st.session_state.data is None:
        st.error("No data available. Please click 'Refresh Now' to fetch data.")
        return
    
    df = st.session_state.data
    pred = st.session_state.prediction
    
    # ==================== Metrics Display ====================
    
    st.markdown("---")
    st.subheader("📊 Price Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # Current Price
    current_price = get_current_price(df)
    with col1:
        st.metric(
            label="Current BNB Price",
            value=format_price(current_price),
            delta=None
        )
    
    # Predicted Price
    predicted_price = pred['price']
    change_percent, direction = calculate_change_percent(current_price, predicted_price)
    
    with col2:
        st.metric(
            label="Predicted Price (Next Day)",
            value=format_price(predicted_price),
            delta=None
        )
    
    # Change Percentage
    delta_text = f"{change_percent:.2f}% {'📈' if direction == 'up' else '📉'}"
    with col3:
        color = "green" if direction == 'up' else "red"
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Change %</div>
            <div style="font-size: 32px; font-weight: bold; color: {color};">
                {delta_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== Chart ====================
    
    st.markdown("---")
    st.subheader("📈 Price Chart (30 Days + Prediction)")
    
    # Prepare chart data
    chart_df = prepare_chart_data(df, pred)
    
    # Create interactive chart with Plotly
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=chart_df.index[:-1],
        y=chart_df['Price'].iloc[:-1],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Predicted price (latest point)
    fig.add_trace(go.Scatter(
        x=[chart_df.index[-2], chart_df.index[-1]],
        y=[chart_df['Price'].iloc[-2], chart_df['Price'].iloc[-1]],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=10, color='#ff7f0e'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='BNB/USDT Price Trend',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== Detailed Predictions ====================
    
    st.markdown("---")
    st.subheader("🎯 Detailed Predictions (Next Day)")
    
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    
    with pred_col1:
        st.metric(
            label="Opening Price",
            value=format_price(pred['open']),
            delta=None
        )
    
    with pred_col2:
        st.metric(
            label="Highest Price",
            value=format_price(pred['high']),
            delta=None
        )
    
    with pred_col3:
        st.metric(
            label="Lowest Price",
            value=format_price(pred['low']),
            delta=None
        )
    
    with pred_col4:
        volume_m = pred['volume'] / 1e6  # Convert to millions
        st.metric(
            label="Volume (M)",
            value=f"{volume_m:,.0f}M",
            delta=None
        )
    
    # ==================== Recent Data Table ====================
    
    st.markdown("---")
    st.subheader("📋 Recent Historical Data (Last 7 Days)")
    
    recent_data = df.tail(7)[['Open', 'High', 'Low', 'Price', 'Volume']].copy()
    recent_data.index = recent_data.index.strftime('%Y-%m-%d')
    
    # Format for display
    display_df = recent_data.copy()
    for col in ['Open', 'High', 'Low', 'Price']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
    display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x/1e6:,.0f}M")
    
    st.dataframe(display_df, use_container_width=True)
    
    # ==================== Auto-refresh logic ====================
    
    if auto_refresh:
        st.markdown("---")
        # Use placeholder for countdown timer
        timer_placeholder = st.empty()
        
        while auto_refresh and (datetime.datetime.now() - st.session_state.last_update).total_seconds() < st.session_state.update_interval:
            seconds_left = st.session_state.update_interval - (datetime.datetime.now() - st.session_state.last_update).total_seconds()
            timer_placeholder.info(f"⏱️ Next update in {int(seconds_left)} seconds...")
            time.sleep(1)
        
        if auto_refresh:
            st.rerun()


if __name__ == "__main__":
    main()
