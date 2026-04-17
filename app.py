import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# --- 1. Path Configuration (Dynamic Version) ---
# 这样无论文件夹放在哪，Python 都会自动找到当前目录
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
def gaussian_weight(d, d0=20):
    """Gaussian decay function for 2SFCA step."""
    if d <= d0:
        num = np.exp(-0.5 * (d / d0)**2) - np.exp(-0.5)
        den = 1 - np.exp(-0.5)
        return num / den
    return 0

def calculate_2sfca(demand_df, supply_df, lines_df, d0=20):
    """Calculates Spatial Accessibility (Ai) using the Gaussian 2SFCA method."""
    # Step 1: Calculate Hospital Supply-to-Demand Ratio (Rj)
    step1 = lines_df.merge(demand_df[['community_id', 'Pop_Demand']], on='community_id')
    step1['weighted_pk'] = step1.apply(
        lambda x: x['Pop_Demand'] * gaussian_weight(x['travel_time'], d0), axis=1
    )
    hospital_demand = step1.groupby('hospital_id')['weighted_pk'].sum().reset_index()
    rj_df = hospital_demand.merge(supply_df[['hospital_id', 'Sj']], on='hospital_id')
    rj_df['Rj'] = rj_df['Sj'] / rj_df['weighted_pk']
    
    # Step 2: Calculate Community Accessibility Score (Ai)
    step2 = lines_df.merge(rj_df[['hospital_id', 'Rj']], on='hospital_id')
    step2['weighted_rj'] = step2.apply(
        lambda x: x['Rj'] * gaussian_weight(x['travel_time'], d0), axis=1
    )
    ai_results = step2.groupby('community_id')['weighted_rj'].sum().reset_index()
    ai_results.columns = ['community_id', 'Ai']
    return ai_results

@st.cache_data
def load_resources():
    def read_csv_safe(file_name, dtypes=None):
        path = os.path.join(BASE_PATH, file_name)
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                df = pd.read_csv(path, dtype=dtypes, encoding=enc)
                df.columns = df.columns.str.strip() # Clean column names
                return df
            except: continue
        return None

    demand = read_csv_safe("Demand_Dataset.csv", {'community_id': str})
    supply = read_csv_safe("Supply_Dataset.csv", {'hospital_id': str})
    lines = read_csv_safe("Lines.csv", {'community_id': str, 'hospital_id': str})
    # 加载带有全属性和坐标的 Candidates.csv
    candidates = read_csv_safe("Candidates.csv")
    
    return demand, supply, lines, candidates

# --- 2. Interface UI ---
st.set_page_config(page_title="HK Healthcare DSS", layout="wide")
st.title("Hong Kong Healthcare Site Selection Decision Support System")

demand_df, supply_df, lines_df, candidates_df = load_resources()

# Session state to persist calculation results
if 'final_results' not in st.session_state:
    st.session_state.final_results = None

if demand_df is not None:
    # Sidebar: Weight Controls
    st.sidebar.header("MCE Weight Configuration")
    w_ai = st.sidebar.slider("Medical Urgency (Ai)", 0.0, 1.0, 0.45)
    w_den = st.sidebar.slider("Population Density", 0.0, 1.0, 0.20)
    w_str = st.sidebar.slider("Vulnerable Groups", 0.0, 1.0, 0.20)
    w_con = st.sidebar.slider("Transport Convenience", 0.0, 1.0, 0.15)

    if st.button("🚀 Execute Spatial Analysis"):
        with st.spinner('Synchronizing spatial layers and calculating...'):
            ai_df = calculate_2sfca(demand_df, supply_df, lines_df)
            st.session_state.final_results = demand_df.merge(ai_df, on='community_id', how='inner')

    # Display Results
    if st.session_state.final_results is not None:
        df = st.session_state.final_results.copy()
        scaler = MinMaxScaler()
        
        # Normalize Score: Lower accessibility (Ai) = Higher Priority (Score)
        df['Ai_Score'] = 1 - scaler.fit_transform(df[['Ai']])
        
        # Total Suitability Calculation (MCE)
        tw = w_ai + w_den + w_str + w_con
        if tw > 0:
            df['Total_Score'] = (
                df['Ai_Score'] * (w_ai/tw) +
                df['Pop_Density'] * (w_den/tw) +
                df['Pop_Structure'] * (w_str/tw) +
                df['Convenience'] * (w_con/tw)
            )

        # Main Layout: Table and Map
        col1, col2 = st.columns([1, 1.8])
        
        with col1:
            st.subheader("Top 10 Priority Communities")
            top_10 = df.sort_values(by='Total_Score', ascending=False).head(10)
            st.dataframe(
                top_10[['community_id', 'Total_Score', 'Pop_Real']], 
                hide_index=True, 
                use_container_width=True
            )

        with col2:
            st.subheader("Spatial Suitability Analysis")
            
            # Layer 1: Suitability Heatmap (TPU Bubbles)
            fig = px.scatter_mapbox(
                df, lat="lat", lon="lon", 
                color="Total_Score", size="Pop_Real",
                color_continuous_scale=px.colors.sequential.Reds,
                zoom=10, height=650,
                mapbox_style="open-street-map", 
                hover_name="community_id",
                labels={'Total_Score': 'Suitability Score'} 
            )

            # Layer 2: Candidate Sites (Blue Markers from your new CSV)
            if candidates_df is not None:
                # 只显示 ID，如果有其他有意义的列（比如面积、地类）再加进去
                hover_text = [f"Proposed Site ID: {row.get('CID', 'N/A')}" for _, row in candidates_df.iterrows()]

                fig.add_trace(go.Scattermapbox(
                    lat=candidates_df['lat'],
                    lon=candidates_df['lon'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=14, color='#007BFF', opacity=0.9
                    ),
                    name='Candidate Locations (MCE Targets)',
                    hoverinfo='text',
                    text=hover_text
                ))

            # UI Final Touches
            fig.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                legend=dict(
                    yanchor="top", y=0.99, 
                    xanchor="left", x=0.01, 
                    bgcolor="rgba(255,255,255,0.7)"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **Visualization Guide:** The **Red Bubbles** indicate TPU-level demand urgency (calculated via 2SFCA). The **Blue Markers** represent specific candidate site locations from the architectural analysis.")