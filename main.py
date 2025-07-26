# main.py

import pandas as pd
from ml import demand_forecasting, risk_prediction
from app import dashboard
import os
import shutil
def clear_pycache():
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                print(f"🧹 Deleted: {pycache_path}")

def main():
    clear_pycache()
    print("🧼 Cleaned up __pycache__ folders.")
    print("🔁 Loading processed data...")

    # Load preprocessed data
    suppliers_df = pd.read_csv("data/processed/suppliers.csv")
    news_df = pd.read_csv("data/processed/news.csv")
    rss_df = pd.read_csv("data/processed/rss_news.csv")
    combined_news = pd.read_csv("data/processed/combined_news_enriched.csv")

    print("✅ Data loaded.")

    # ---------------- Demand Forecast ----------------
    print("📈 Training demand forecast model...")

    # For demo: Simulated sales data
    sales_sample = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=100),
        "quantity": (50 + (pd.Series(range(100)) % 30) + (5 * pd.Series(range(100)).apply(lambda x: x % 2))).astype(int),
        "temperature": 25 + (pd.Series(range(100)) % 5),
        "precipitation": (pd.Series(range(100)) % 3)
    })

    forecast_model, forecast_df = demand_forecasting.train_demand_model(sales_sample)

    print("✅ Demand forecasting complete.")

    # ---------------- Risk Prediction ----------------
    print("⚠️ Training risk prediction model...")

    risk_model, enriched_suppliers_df, _ = risk_prediction.train_risk_model()

    print("✅ Risk model trained.")

    # ---------------- Dashboard ----------------
    print("📊 Launching dashboard...")

    dashboard.show_dashboard(
        enriched_suppliers_df,
        forecast_df,
        combined_news
    )


if __name__ == "__main__":
    main()
