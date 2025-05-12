import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import timedelta

# PostgreSQL connection
engine = create_engine("postgresql://miniproject:31998369@localhost:6543/price_comparison")

def fetch_data(product_name, platform):
    try:
        query = f"""
            SELECT * FROM product_prices4
            WHERE LOWER(product_name) LIKE '%%{product_name.lower()}%%'
            AND LOWER(platform) = '{platform.lower()}'
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        print("Data fetch error:", e)
        return None

def prepare_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.groupby('timestamp').agg({'price': 'mean'}).reset_index()
    df['days'] = (df['timestamp'] - df['timestamp'].min()).dt.days
    return df

def create_future_df(df, days=180):
    max_day = df['days'].max()
    last_date = df['timestamp'].max()
    future_days = np.arange(max_day + 1, max_day + days + 1)
    future_dates = [last_date + timedelta(days=int(i)) for i in range(1, days + 1)]
    return pd.DataFrame({'days': future_days, 'timestamp': future_dates})

def train_regression(df, future_df):
    X = df[['days']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X)
    y_future_lr = lr.predict(future_df[['days']])

    xgboost_model = xgb.XGBRegressor()
    xgboost_model.fit(X_train, y_train)
    y_pred_xgb = xgboost_model.predict(X)
    y_future_xgb = xgboost_model.predict(future_df[['days']])

    return lr, xgboost_model, y_pred_lr, y_pred_xgb, y_future_lr, y_future_xgb

def train_lstm(df, future_df):
    sequence_length = 7
    data = df['price'].values
    X_lstm, y_lstm = [], []

    for i in range(len(data) - sequence_length):
        X_lstm.append(data[i:i + sequence_length])
        y_lstm.append(data[i + sequence_length])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_lstm, y_lstm, epochs=50, verbose=0)

    y_pred_lstm = model.predict(X_lstm).flatten()

    last_sequence = data[-sequence_length:]
    future_prices = []
    current_sequence = last_sequence.copy()

    for _ in range(len(future_df)):
        pred = model.predict(current_sequence.reshape(1, sequence_length, 1), verbose=0)[0][0]
        future_prices.append(pred)
        current_sequence = np.append(current_sequence[1:], pred)

    return model, y_pred_lstm, future_prices

def plot_interactive(df, y_pred_lr, y_pred_xgb, y_pred_lstm,
                     future_df, y_future_lr, y_future_xgb, y_future_lstm,
                     platform):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name='Actual',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=y_pred_lr, mode='lines', name='Linear Regression',line=dict(color='Green')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=y_pred_xgb, mode='lines', name='XGBoost',line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['timestamp'][7:], y=y_pred_lstm, mode='lines', name='LSTM',line=dict(color='purple')))

    fig.add_trace(go.Scatter(x=future_df['timestamp'], y=y_future_lr, mode='lines', name='LR Forecast',line=dict(color='Green')))
    fig.add_trace(go.Scatter(x=future_df['timestamp'], y=y_future_xgb, mode='lines', name='XGB Forecast',line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=future_df['timestamp'], y=y_future_lstm, mode='lines', name='LSTM Forecast',line=dict(color='purple')))

    fig.update_layout(
        title=f"{platform} Price Trend Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    return fig

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)

import os
import traceback  # To print full exception

def generate_price_chart(product_name, platform, chart_dir='static/charts'):
    try:
        # ✅ Ensure the chart directory exists
        os.makedirs(chart_dir, exist_ok=True)

        df = fetch_data(product_name, platform)
        if df is None or df.empty:
            print("[ERROR] No data fetched.")
            return False, [], "Unknown", {}, None

        df = prepare_data(df)
        future_df = create_future_df(df, days=180)

        lr, xgb, y_pred_lr, y_pred_xgb, y_future_lr, y_future_xgb = train_regression(df, future_df)
        lstm_model, y_pred_lstm, y_future_lstm = train_lstm(df, future_df)

        valid_len = min(len(df['price']) - 7, len(y_pred_lstm))
        mae_lstm = mean_absolute_error(df['price'][7:7 + valid_len], y_pred_lstm[:valid_len])
        mae_lr = mean_absolute_error(df['price'], y_pred_lr)
        mae_xgb = mean_absolute_error(df['price'], y_pred_xgb)

        best_model = min(
            [('Linear Regression', mae_lr), ('XGBoost', mae_xgb), ('LSTM', mae_lstm)],
            key=lambda x: x[1]
        )[0]

        if best_model == "Linear Regression":
            final_price = y_future_lr[-1]
        elif best_model == "XGBoost":
            final_price = y_future_xgb[-1]
        else:
           final_price = y_future_lstm[-1]
          

        summary = {
            "MAE_LR": round(mae_lr, 2),
            "MAE_XGB": round(mae_xgb, 2),
            "MAE_LSTM": round(mae_lstm, 2),
            "Best_Model": best_model,
            "predicted_price":round(float(final_price), 2)
        }

        # ✅ Safe filename and path
        filename = f"{sanitize_filename(product_name)}_{sanitize_filename(platform.lower())}_chart.html"
        chart_path = os.path.join(chart_dir, filename)

        print(f"[DEBUG] Saving chart to: {chart_path}")

        fig = plot_interactive(
            df, y_pred_lr, y_pred_xgb, y_pred_lstm,
            future_df, y_future_lr, y_future_xgb, y_future_lstm,
            platform
        )

        fig.write_html(chart_path)

        print("[DEBUG] Chart saved successfully.")

        # ✅ Format predictions
        predictions = []
        for i in range(len(future_df)):
            month_str = future_df['timestamp'].iloc[i].strftime('%B %Y')
            predictions.extend([
                {"month": month_str, "model_name": "Linear Regression", "predicted_price": round(y_future_lr[i], 2)},
                {"month": month_str, "model_name": "XGBoost", "predicted_price": round(y_future_xgb[i], 2)},
                {"month": month_str, "model_name": "LSTM", "predicted_price": round(y_future_lstm[i], 2)},
            ])

        category = "Laptop" if "laptop" in product_name.lower() else "Unknown"

        return True, predictions, category, summary, filename

    except Exception as e:
        print(f"[ERROR] Chart generation failed: {e}")
        traceback.print_exc()  # ✅ Show full traceback
        return False, [], "Unknown", {}, None