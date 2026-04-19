import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings('ignore')

# ==================== Model & Scaler Loading ====================

def load_model_and_scaler(model_path='best_cnn_lstm_model.onnx', scaler_path='data_scaler.joblib'):
    """
    Load trained CNN-LSTM model (ONNX format) and data scaler
    
    Args:
        model_path: Path to .onnx model file
        scaler_path: Path to .joblib scaler file
    
    Returns:
        tuple: (session, scaler)
        - session: ONNX Runtime InferenceSession
        - scaler: Fitted MinMaxScaler object
    """
    try:
        # Load ONNX model using ONNX Runtime
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        scaler = load(scaler_path)
        return session, scaler
    except Exception as e:
        raise Exception(f"Error loading model or scaler: {str(e)}")


# ==================== Data Fetching (yfinance) ====================

def fetch_bnb_data(ticker='BNB-USD', days=30):
    """
    Fetch historical OHLCV data from Yahoo Finance (yfinance)
    
    NO GEO-RESTRICTIONS, NO API KEY NEEDED, WORLDWIDE ACCESS
    Works from any country including Indonesia
    
    Args:
        ticker: Ticker symbol (default: 'BNB-USD' for BNB in USD)
        days: Number of days to fetch (default: 30)
    
    Returns:
        tuple: (df, source) where:
            - df: DataFrame with columns [Date, Open, High, Low, Price, Volume]
            - source: 'yfinance'
    """
    try:
        # Calculate date range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days + 5)  # Extra days for margin
        
        # Fetch data using yfinance
        print(f"Fetching {ticker} data from Yahoo Finance...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, interval='1d')
        
        if df.empty:
            raise Exception(f"No data found for {ticker}")
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns: yfinance returns [Date, Open, High, Low, Close, Volume, Adj Close]
        # We rename Close to Price for consistency with our model
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Price', 'Volume']
        
        # Convert to numeric
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Remove NaN rows
        df = df.dropna()
        
        # Keep last 30 days
        df = df.tail(30)
        
        # Sort by date ascending (oldest first)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        print(f"✅ Fetched {len(df)} days of data from Yahoo Finance")
        
        return df, 'yfinance'
    
    except Exception as e:
        raise Exception(f"Error fetching Yahoo Finance data: {str(e)}")


# ==================== Data Preprocessing ====================

def create_sequences(data, seq_length=30):
    """
    Create sequences for LSTM input
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        seq_length: Sequence length (default: 30)
    
    Returns:
        numpy array of shape (n_samples - seq_length, seq_length, n_features)
    """
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)


def preprocess_data(df, scaler, seq_length=30):
    """
    Preprocess data for model prediction
    
    Args:
        df: DataFrame with columns [Open, High, Low, Price, Volume]
        scaler: Fitted MinMaxScaler object
        seq_length: Sequence length for LSTM (default: 30)
    
    Returns:
        tuple: (X_sequences, df_scaled)
               X_sequences: shape (n_samples - seq_length, seq_length, 5)
               df_scaled: scaled DataFrame values
    """
    try:
        # Extract features in the correct order: [Price, Open, High, Low, Volume]
        # This should match the scaler's feature order from training
        data = df[['Price', 'Open', 'High', 'Low', 'Volume']].values
        
        # Normalize using fitted scaler
        data_scaled = scaler.transform(data)
        
        # Create sequences
        X_sequences = create_sequences(data_scaled, seq_length=seq_length)
        
        return X_sequences, data_scaled
    
    except Exception as e:
        raise Exception(f"Error preprocessing data: {str(e)}")


def get_latest_sequence(df, scaler, seq_length=30):
    """
    Get the latest sequence for prediction
    
    Args:
        df: DataFrame with columns [Open, High, Low, Price, Volume]
        scaler: Fitted MinMaxScaler object
        seq_length: Sequence length (default: 30)
    
    Returns:
        numpy array of shape (1, seq_length, 5)
    """
    try:
        # Check data availability
        if len(df) < seq_length:
            raise ValueError(f"Not enough data. Need at least {seq_length} rows, got {len(df)}")
        
        # Extract last 30 days of data
        recent_data = df[['Price', 'Open', 'High', 'Low', 'Volume']].tail(seq_length).values
        
        # Ensure all values are positive (sanity check)
        if np.any(recent_data < 0):
            print("⚠️ Warning: Negative values detected in data. Applying absolute value.")
            recent_data = np.abs(recent_data)
        
        # Normalize using fitted scaler
        # Scaler expects shape (n_samples, n_features)
        recent_data_scaled = scaler.transform(recent_data)
        
        # Ensure normalized data is in valid range [0, 1]
        recent_data_scaled = np.clip(recent_data_scaled, 0.0, 1.0)
        
        # Add batch dimension: (30, 5) -> (1, 30, 5)
        X_recent = recent_data_scaled[np.newaxis, :, :]
        
        return X_recent
    
    except Exception as e:
        raise Exception(f"Error creating latest sequence: {str(e)}")


# ==================== Model Prediction ====================

def predict_price(X_recent, session, scaler):
    """
    Predict next day's values using ONNX model
    
    Args:
        X_recent: Input sequence of shape (1, 30, 5)
        session: ONNX Runtime InferenceSession
        scaler: Fitted MinMaxScaler object
    
    Returns:
        dict: {
            'price': float (predicted closing price),
            'open': float,
            'high': float,
            'low': float,
            'volume': float
        }
    """
    try:
        # Get input and output names from ONNX model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Convert input to float32
        X_recent = X_recent.astype(np.float32)
        
        # Run inference
        prediction_normalized = session.run([output_name], {input_name: X_recent})[0]
        
        # Ensure predictions are in valid range [0, 1]
        prediction_normalized = np.clip(prediction_normalized, 0.0, 1.0)
        
        # Inverse transform to original scale
        prediction_original = scaler.inverse_transform(prediction_normalized)
        
        # Extract values [Price, Open, High, Low, Volume]
        price = float(prediction_original[0, 0])
        open_price = float(prediction_original[0, 1])
        high_price = float(prediction_original[0, 2])
        low_price = float(prediction_original[0, 3])
        volume = float(prediction_original[0, 4])
        
        # Validation: Ensure prices are positive and realistic
        # BNB typically ranges $50-$1000, but use broader bounds for safety
        MIN_PRICE = 1.0  # Minimum valid price
        MAX_PRICE = 100000.0  # Maximum valid price (upper bound)
        
        # If prediction is outside valid range, use last known price as estimate
        if price < MIN_PRICE or price > MAX_PRICE:
            print(f"⚠️ Warning: Unrealistic prediction {price}. Using last known price.")
            # Use median of recent prices as fallback
            price = np.median([open_price, high_price, low_price]) 
            if price < MIN_PRICE or price > MAX_PRICE:
                price = 200.0  # BNB default fallback
        
        # Ensure OHLC relationships are logical
        if open_price < MIN_PRICE or open_price > MAX_PRICE:
            open_price = price
        if high_price < price:
            high_price = price * 1.02  # High should be >= close
        if low_price > price:
            low_price = price * 0.98  # Low should be <= close
        if volume < 0:
            volume = abs(volume)
        
        result = {
            'price': price,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'volume': volume
        }
        
        return result
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")


# ==================== Utility Functions ====================

def calculate_change_percent(current_price, predicted_price):
    """
    Calculate percentage change from current to predicted price
    
    Args:
        current_price: Current BNB price
        predicted_price: Predicted BNB price
    
    Returns:
        tuple: (change_percent: float, direction: str)
               direction: 'up' or 'down'
    """
    change = predicted_price - current_price
    change_percent = (change / current_price) * 100
    direction = 'up' if change_percent >= 0 else 'down'
    
    return abs(change_percent), direction


def format_price(price):
    """Format price as string with 2 decimal places"""
    return f"${price:,.2f}"


def get_current_price(df):
    """Get the current (latest) BNB price"""
    return float(df.iloc[-1]['Price'])


def prepare_chart_data(df, predictions):
    """
    Prepare data for chart visualization
    
    Args:
        df: Historical DataFrame
        predictions: Dict of predicted values
    
    Returns:
        pd.DataFrame: Combined historical + prediction data
    """
    try:
        chart_df = df[['Price']].copy()
        
        # Add prediction for tomorrow
        tomorrow = df.index[-1] + pd.Timedelta(days=1)
        prediction_row = pd.DataFrame(
            {'Price': [predictions['price']]},
            index=[tomorrow]
        )
        
        chart_df = pd.concat([chart_df, prediction_row])
        
        return chart_df
    
    except Exception as e:
        raise Exception(f"Error preparing chart data: {str(e)}")


def validate_and_fix_prediction(pred_dict, current_price):
    """
    Validate prediction and fix if necessary
    Ensures prediction is realistic and positive
    
    Args:
        pred_dict: Prediction dict with price, open, high, low, volume
        current_price: Current market price for reference
    
    Returns:
        dict: Validated prediction
    """
    price = pred_dict['price']
    open_price = pred_dict['open']
    high_price = pred_dict['high']
    low_price = pred_dict['low']
    volume = pred_dict['volume']
    
    # If any value is negative or NaN, fix it
    if not np.isfinite(price) or price < 1.0:
        price = current_price
    if not np.isfinite(open_price) or open_price < 1.0:
        open_price = current_price
    if not np.isfinite(high_price) or high_price < 1.0:
        high_price = current_price
    if not np.isfinite(low_price) or low_price < 1.0:
        low_price = current_price
    if not np.isfinite(volume) or volume < 0:
        volume = 0
    
    # If prediction is too far from current price (> 50% change), use conservative estimate
    if current_price > 0:
        price_change_pct = abs(price - current_price) / current_price * 100
        if price_change_pct > 50:
            print(f"⚠️ Prediction change too extreme ({price_change_pct:.1f}%). Using conservative estimate...")
            # Use weighted average: 80% current, 20% prediction
            price = current_price * 0.8 + price * 0.2
    
    # Ensure OHLC logic (High >= Close >= Low)
    high_price = max(high_price, price, open_price, low_price)
    low_price = min(low_price, price, open_price, high_price)
    
    # Final sanity check
    price = max(float(price), 1.0)
    open_price = max(float(open_price), 1.0)
    high_price = max(float(high_price), 1.0)
    low_price = max(float(low_price), 1.0)
    volume = max(float(volume), 0)
    
    return {
        'price': price,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'volume': volume
    }
