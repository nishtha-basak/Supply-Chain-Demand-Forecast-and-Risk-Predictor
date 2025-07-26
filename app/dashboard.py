# app/dashboard.py

import streamlit as st
import plotly.express as px
import pandas as pd

def show_dashboard(enriched_suppliers_df, forecast_df, combined_news):

    st.set_page_config(layout="wide")
    st.title("📦 Supply Chain Risk Dashboard")

    # 🌍 Supplier Map (uses enriched supplier data)
    st.subheader("🌐 Supplier Locations & Disruption Probability")
    map_df = enriched_suppliers_df.copy()
    fig = px.scatter_geo(
        map_df,
        lat='latitude',
        lon='longitude',
        hover_name='name',
        hover_data={
            'location': True,
            'lead_time': True,
            'defect_rate': True,
            'on_time_rate': True,
            'disruption_prob': True
        },
        size='disruption_prob',
        color='disruption_prob',
        color_continuous_scale='Reds',
        title='Supplier Risk Map'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig)

    # 📈 Demand Forecast
    st.subheader("📊 Demand Forecast for Product A")
    st.line_chart(forecast_df[["ds", "yhat"]].set_index("ds"))

    # 🚨 High Risk Suppliers Table
    st.subheader("🚨 Top Risky Suppliers")
    top_risk = enriched_suppliers_df.sort_values("disruption_prob", ascending=False).head(10)
    st.dataframe(top_risk[[
        'name', 'location', 'lead_time', 'defect_rate',
        'on_time_rate', 'disruption_prob'
    ]])
