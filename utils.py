import os
import pandas as pd
import sqlite3

def init_database():
    """Initialize SQLite database"""
    os.makedirs("data/database", exist_ok=True)
    conn = sqlite3.connect("data/database/supply_chain.db")
    return conn

def save_data(df, name, data_type="processed"):
    """Save data to appropriate location"""
    os.makedirs(f"data/{data_type}", exist_ok=True)
    df.to_csv(f"data/{data_type}/{name}.csv", index=False)
    
    # Also save to database
    conn = init_database()
    df.to_sql(name, conn, if_exists="replace", index=False)
    conn.close()