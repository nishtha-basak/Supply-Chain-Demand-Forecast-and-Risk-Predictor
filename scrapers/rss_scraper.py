# scrapers/rss_scraper.py

import feedparser
import pandas as pd
from datetime import datetime

def fetch_google_rss():
    """Fetch recent supply chain disruption headlines from Google News RSS"""
    url = "https://news.google.com/rss/search?q=supply+chain+disruption+OR+shortage+OR+delay&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries:
        articles.append({
            "published": datetime(*entry.published_parsed[:6]),
            "title": entry.title,
            "summary": entry.summary,
            "link": entry.link,
            "source": entry.source.title if "source" in entry else "Google News"
        })

    return pd.DataFrame(articles)
