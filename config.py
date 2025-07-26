# config.py

from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

# API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Data source URLs
SUPPLIER_DATA_URL = "https://raw.githubusercontent.com/nogibjj/supply-chain-risk/main/data/suppliers.csv"
