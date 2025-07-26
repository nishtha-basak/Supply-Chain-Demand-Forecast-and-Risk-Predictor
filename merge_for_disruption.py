import pandas as pd
import numpy as np
import os

# Load processed datasets
suppliers = pd.read_csv("data/processed/suppliers.csv")
news = pd.read_csv("data/processed/combined_news_enriched.csv")
news["published"] = pd.to_datetime(news["published"], errors="coerce")

# Define mapping of cities (in suppliers) to check in news titles/text
cities = suppliers["location"].unique()

# Tag news articles with matched city
def match_city(text):
    text = str(text).lower()
    for city in cities:
        if city.lower() in text:
            return city
    return None

news["matched_city"] = news["title"].apply(match_city)

# Drop unrelated news
city_news = news.dropna(subset=["matched_city"])

# Consider only recent news (last 5 days)
latest_date = city_news["published"].max()
city_news = city_news[city_news["published"] >= latest_date - pd.Timedelta(days=5)]

# Aggregate risk/sentiment by city
agg_funcs = {
    "title_sentiment": "mean",
    "content_sentiment": "mean",
    "has_strike": "sum",
    "has_war": "sum",
    "has_hurricane": "sum",
    "has_delay": "sum",
    "has_shortage": "sum",
    "has_sanction": "sum"
}
risk_summary = city_news.groupby("matched_city").agg(agg_funcs).reset_index()
risk_summary = risk_summary.rename(columns={"matched_city": "location"})

# Compute simple disruption probability
def compute_disruption_prob(row):
    risk_score = sum([
        row.get("has_strike", 0),
        row.get("has_war", 0),
        row.get("has_hurricane", 0),
        row.get("has_shortage", 0),
        row.get("has_delay", 0),
        row.get("has_sanction", 0)
    ])
    sentiment_score = (1 - row["title_sentiment"] + 1 - row["content_sentiment"]) / 2
    prob = min(1.0, (risk_score * 0.1) + (sentiment_score * 0.2))
    return round(prob, 2)

risk_summary["disruption_prob"] = risk_summary.apply(compute_disruption_prob, axis=1)

# Merge with supplier data
final_df = suppliers.merge(risk_summary, on="location", how="left")

# Fill missing risk values (for cities with no recent news)
risk_cols = [col for col in final_df.columns if "has_" in col or "sentiment" in col]
final_df[risk_cols] = final_df[risk_cols].fillna(0)
final_df["disruption_prob"] = final_df["disruption_prob"].fillna(0.1)

# Save
output_path = "data/processed/supplier_disruption_view.csv"
final_df.to_csv(output_path, index=False)

print(f"✅ Disruption-aware supplier dataset saved → {output_path}")
