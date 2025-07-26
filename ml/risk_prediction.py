import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib


def train_risk_model():
    """Train a risk prediction model using preprocessed supplier+news data"""

    # Load preprocessed supplier-disruption data
    df = pd.read_csv("data/processed/supplier_disruption_view.csv")

    # Features used for training
    features = [
        'lead_time', 'defect_rate', 'on_time_rate',
        'title_sentiment', 'content_sentiment',
        'has_strike', 'has_hurricane', 'has_war', 'has_sanction', 'has_shortage', 'has_delay'
    ]

    # Simulated targets
    y = pd.DataFrame({
        'disruption_prob': np.random.uniform(0, 1, len(df)),
        'delay_days': np.random.randint(0, 14, len(df))
    })

    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

        # Save model to file
    import os
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/risk_model.pkl")
    print("✅ Risk model saved to 'models/risk_model.pkl'")

    return model, df, y


