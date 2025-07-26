import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def train_demand_model(sales_data):
    """Train demand forecasting model with Prophet"""
    
    # Step 1: Rename columns
    df = sales_data.rename(columns={'date': 'ds', 'quantity': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    # Step 2: Prophet model with regressors
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    for reg in ['temperature', 'precipitation']:
        if reg in df.columns:
            model.add_regressor(reg)

    # Step 3: Fit the model
    model.fit(df)

    # Step 4: Future dataframe for 30 days
    future = model.make_future_dataframe(periods=30, freq='D')

    # Step 5: Merge regressors
    for reg in ['temperature', 'precipitation']:
        if reg in df.columns:
            future = future.merge(df[['ds', reg]], on='ds', how='left')
            future[reg] = future[reg].fillna(df[reg].mean())

    # Step 6: Forecast
    forecast = model.predict(future)

    return model, forecast
