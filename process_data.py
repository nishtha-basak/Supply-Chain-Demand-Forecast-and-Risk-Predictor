import pandas as pd
import numpy as np
import os
from textblob import TextBlob

# Define file paths
fashion_path = "data/raw/fashion_supply_chain.csv"
newsapi_path = "data/raw/news.csv"
rss_path = "data/raw/rss_news.csv"
legacy_suppliers_path = "data/raw/suppliers.csv"

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# ------------------------ 1. FASHION SUPPLIER DATA ------------------------
fashion_df = pd.read_csv(fashion_path)

# Rename and clean essential columns
fashion_df = fashion_df.rename(columns={
    "Supplier name": "name",
    "Location": "location",
    "Lead time": "lead_time",
    "Defect rates": "defect_rate"
})

# Convert lead_time and defect_rate to numeric
fashion_df["lead_time"] = pd.to_numeric(fashion_df["lead_time"], errors="coerce").fillna(10).astype(int)
fashion_df["defect_rate"] = pd.to_numeric(fashion_df["defect_rate"], errors="coerce").fillna(0.05).round(3)

# Simulated on-time rate
fashion_df["on_time_rate"] = np.round(np.random.uniform(0.75, 0.98, len(fashion_df)), 2)

# Add geolocation for Indian cities
city_coords = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639)
}
fashion_df["latitude"] = fashion_df["location"].map(lambda loc: city_coords.get(str(loc).strip(), (np.nan, np.nan))[0])
fashion_df["longitude"] = fashion_df["location"].map(lambda loc: city_coords.get(str(loc).strip(), (np.nan, np.nan))[1])

# Drop suppliers without geolocation
fashion_df = fashion_df.dropna(subset=["latitude", "longitude"])

# Select final columns
fashion_clean = fashion_df[[
    "name", "location", "lead_time", "defect_rate", "on_time_rate", "latitude", "longitude"
]]
fashion_clean.to_csv("data/processed/suppliers.csv", index=False)

# ------------------------ 2. NEWSAPI DATA ------------------------
news_df = pd.read_csv(newsapi_path)
news_df["published_at"] = pd.to_datetime(news_df["published_at"], errors="coerce")
news_df["title"] = news_df["title"].fillna("").str.strip()
news_df["content"] = news_df["content"].fillna("").str.strip()
news_df = news_df.dropna(subset=["published_at"])
news_df.to_csv("data/processed/news.csv", index=False)

# ------------------------ 3. RSS NEWS DATA ------------------------
rss_df = pd.read_csv(rss_path)
rss_df["published"] = pd.to_datetime(rss_df["published"], errors="coerce")
rss_df["title"] = rss_df["title"].fillna("").str.strip()
rss_df["summary"] = rss_df["summary"].fillna("").str.strip()
rss_df = rss_df.dropna(subset=["published"])
rss_df.to_csv("data/processed/rss_news.csv", index=False)

# ------------------------ 4. LEGACY SUPPLIER DATA ------------------------
legacy_df = pd.read_csv(legacy_suppliers_path)
legacy_df.to_csv("data/processed/suppliers_legacy.csv", index=False)

# ------------------------ 5. COMBINE NEWS SOURCES ------------------------
combined_news = pd.concat([
    news_df.rename(columns={"published_at": "published", "content": "text"})[["published", "title", "text"]],
    rss_df.rename(columns={"summary": "text"})[["published", "title", "text"]]
], ignore_index=True)
combined_news.to_csv("data/processed/combined_news.csv", index=False)

# ------------------------ 6. SENTIMENT + RISK KEYWORDS ------------------------
def analyze_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return 0
    return TextBlob(text).sentiment.polarity

combined_news["title_sentiment"] = combined_news["title"].apply(analyze_sentiment)
combined_news["content_sentiment"] = combined_news["text"].apply(analyze_sentiment)

# Define critical keywords
risk_keywords = ['strike', 'war', 'hurricane', 'delay', 'shortage', 'sanction']
for kw in risk_keywords:
    combined_news[f"has_{kw}"] = combined_news["text"].apply(
        lambda x: int(kw in str(x).lower())
    )

combined_news.to_csv("data/processed/combined_news_enriched.csv", index=False)

print("✅ All raw files processed and saved.")
print("✅ Sentiment + risk keyword features saved to 'combined_news_enriched.csv'.")
