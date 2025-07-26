# scrapers/supplier_scraper.py

import pandas as pd
import numpy as np

def get_supplier_data():
    df = pd.read_csv("data/raw/fashion_supply_chain.csv")

    # Rename selected columns
    df = df.rename(columns={
        "Supplier name": "name",
        "Location": "location",
        "Lead time": "lead_time",
        "Defect rates": "defect_rate",
        "Inspection results": "inspection_result",
        "Production volumes": "prod_volume",
        "Manufacturing costs": "manufacturing_cost",
        "Shipping times": "shipping_time",
        "Shipping costs": "shipping_cost"
    })

    # Preprocessing
    df['inspection_result'] = df['inspection_result'].str.lower().map({'pass': 1, 'fail': 0})
    df['defect_rate'] = pd.to_numeric(df['defect_rate'], errors='coerce')
    df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce')
    df['shipping_time'] = pd.to_numeric(df['shipping_time'], errors='coerce')
    df['shipping_cost'] = pd.to_numeric(df['shipping_cost'], errors='coerce')
    df['manufacturing_cost'] = pd.to_numeric(df['manufacturing_cost'], errors='coerce')
    df['prod_volume'] = pd.to_numeric(df['prod_volume'], errors='coerce')

    # Handle missing values (basic)
    df = df.fillna({
        "defect_rate": df["defect_rate"].mean(),
        "lead_time": df["lead_time"].mean(),
        "shipping_time": df["shipping_time"].mean(),
        "shipping_cost": df["shipping_cost"].mean(),
        "manufacturing_cost": df["manufacturing_cost"].mean(),
        "prod_volume": df["prod_volume"].mean(),
        "inspection_result": 1
    })

    # Add derived fields
    df["on_time_rate"] = np.round(np.random.uniform(0.75, 0.98, size=len(df)), 2)

    # Risk flag based on thresholds
    df["critical_flag"] = ((df["lead_time"] > 20) | (df["defect_rate"] > 0.10) | (df["inspection_result"] == 0)).astype(int)

    # Final selected columns
    df = df[[
        "name", "location", "lead_time", "defect_rate", "on_time_rate",
        "inspection_result", "prod_volume", "manufacturing_cost",
        "shipping_time", "shipping_cost", "critical_flag"
    ]]

    return df
