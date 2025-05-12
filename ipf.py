import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sqlalchemy import create_engine
import plotly.graph_objects as go
from datetime import timedelta
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ========== 1. FETCH DATA ========== #
def fetch_data(product_name, platform):
    engine = create_engine("postgresql+psycopg2://miniproject:31998369@localhost:6543/price_comparison")
    query = """
    SELECT product_name, price, timestamp, platform 
    FROM product_prices4
    WHERE LOWER(product_name) LIKE %s AND LOWER(platform) = %s 
    ORDER BY timestamp
    """
    df = pd.read_sql(query, engine, params=('%' + product_name.lower() + '%', platform.lower()))
    engine.dispose()

    if df.empty:
        print(f"No data found for '{product_name}' on {platform}")
        return None
    return df

# ========== 2. PREPARE DATA ========== #
def prepare_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
    return df

# ========== 3. CREATE FUTURE DF ========== #
def create_future_df(df, days=180):
    last_date = df['timestamp'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_days_since_start = [(d - df['timestamp'].min()).days for d in future_dates]
    return pd.DataFrame({'timestamp': future_dates, 'days_since_start': future_days_since_start})

# ========== 4. REGRESSION TRAINING + FUTURE PREDICTION ========== #
def train_regression(df, future_df):
    X = df[['days_since_start']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X)
    y_future_lr = lr.predict(future_df[['days_since_start']])

    # XGBoost
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X)
    y_future_xgb = xgb.predict(future_df[['days_since_start']])

    return lr, xgb, y_pred_lr, y_pred_xgb, y_future_lr, y_future_xgb

# ========== 5. LSTM TRAINING + FUTURE PREDICTION ========== #
def train_lstm(df, future_df):
    scaler_price = MinMaxScaler()
    df['price_scaled'] = scaler_price.fit_transform(df[['price']])

    X = np.array(df['days_since_start']).reshape(-1, 1, 1)
    y = np.array(df['price_scaled'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(1, 1)),
        LSTM(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=60, batch_size=8, verbose=0)

    # Predict on full data
    y_pred_lstm = model.predict(X)
    y_pred_lstm = scaler_price.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

    # Predict into future
    X_future = np.array(future_df['days_since_start']).reshape(-1, 1, 1)
    y_future_lstm = model.predict(X_future)
    y_future_lstm = scaler_price.inverse_transform(y_future_lstm.reshape(-1, 1)).flatten()

    return model, y_pred_lstm, y_future_lstm

# ========== 6. INTERACTIVE PLOTTING ========== #
def plot_interactive(df, y_pred_lr, y_pred_xgb, y_pred_lstm,
                     future_df, y_future_lr, y_future_xgb, y_future_lstm,
                     platform):
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=y_pred_lr, mode='lines', name='LR Fit', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=y_pred_xgb, mode='lines', name='XGB Fit', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=y_pred_lstm, mode='lines', name='LSTM Fit', line=dict(color='purple')))

    # Future
    fig.add_trace(go.Scatter(x=future_df['timestamp'], y=y_future_lr, mode='lines', name='LR Future', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=future_df['timestamp'], y=y_future_xgb, mode='lines', name='XGB Future', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=future_df['timestamp'], y=y_future_lstm, mode='lines', name='LSTM Future', line=dict(color='purple', dash='dot')))

    fig.update_layout(
        title=f"Price Prediction (Past + 6 Months Future) - {platform.capitalize()}",
        xaxis_title='Date',
        yaxis_title='Price',
        template="plotly_white",
        legend=dict(x=0, y=1.1, orientation="h")
    )

    fig.show()

# ========== 7. RUN + COMPARE MAEs FOR EACH PLATFORM ========== #
platforms = ["amazon", "flipkart", "ebay", "walmart"]
product_name = "Dell XPS 15"
results = []

for platform in platforms:
    print(f"\n=== {platform.upper()} ===")
    df = fetch_data(product_name, platform)
    if df is not None:
        df = prepare_data(df)
        future_df = create_future_df(df, days=180)

        # Train models and get predictions
        lr, xgb, y_pred_lr, y_pred_xgb, y_future_lr, y_future_xgb = train_regression(df, future_df)
        lstm_model, y_pred_lstm, y_future_lstm = train_lstm(df, future_df)

        # Compute MAEs
        mae_lr = mean_absolute_error(df['price'], y_pred_lr)
        mae_xgb = mean_absolute_error(df['price'], y_pred_xgb)
        mae_lstm = mean_absolute_error(df['price'], y_pred_lstm)

        # Save results
        results.append({
            'Platform': platform,
            'MAE_LR': round(mae_lr, 2),
            'MAE_XGB': round(mae_xgb, 2),
            'MAE_LSTM': round(mae_lstm, 2),
            'Best_Model': min(
                [('Linear Regression', mae_lr),
                 ('XGBoost', mae_xgb),
                 ('LSTM', mae_lstm)],
                key=lambda x: x[1]
            )[0]
        })

        # Plot predictions
        plot_interactive(df, y_pred_lr, y_pred_xgb, y_pred_lstm,
                         future_df, y_future_lr, y_future_xgb, y_future_lstm,
                         platform)

# ========== 8. PRINT SUMMARY ========== #
print("\n========= MODEL PERFORMANCE SUMMARY =========")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# ========== 9. PLOT COMPARISON ACCURACY GRAPH ========== #
def plot_mae_comparison(summary_df):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=summary_df['Platform'],
        y=summary_df['MAE_LR'],
        name='Linear Regression',
        marker_color='red'
    ))

    fig.add_trace(go.Bar(
        x=summary_df['Platform'],
        y=summary_df['MAE_XGB'],
        name='XGBoost',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=summary_df['Platform'],
        y=summary_df['MAE_LSTM'],
        name='LSTM',
        marker_color='purple'
    ))

    fig.update_layout(
        title='Model Comparison (MAE) per Platform',
        xaxis_title='Platform',
        yaxis_title='Mean Absolute Error (MAE)',
        barmode='group',
        template='plotly_white'
    )

    fig.show()

# Call the plot function
plot_mae_comparison(summary_df)
