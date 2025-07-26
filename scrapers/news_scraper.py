import requests
import pandas as pd
from config import NEWS_API_KEY
from datetime import datetime, timedelta

def get_supply_chain_news(keywords=["supply chain", "disruption", "shortage", "logistics"]):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": " OR ".join(keywords),
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    }

    response = requests.get(url, params=params)
    articles = response.json().get('articles', [])

    news_data = []
    for article in articles:
        news_data.append({
            "published_at": article["publishedAt"],
            "title": article["title"],
            "description": article["description"],
            "source": article["source"]["name"],
            "url": article["url"],
            "content": article["content"]
        })

    return pd.DataFrame(news_data)
