import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import plotly.io as pio
from datetime import datetime
import os

# ==================== Page Config ====================
st.set_page_config(page_title="Mining Dashboard", page_icon="⛏️", layout="wide")

# ==================== Load Data ====================
@st.cache_data
def load_data():
    sheet_id = "1U7nKh9PGuNsNxJVx73-HgnrvWlTmpsayI1GfaNe2CgI"
    gid = "1457068487"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
    df = pd.read_csv(url, engine='python')
    df['Date'] = pd.to_datetime(df['Date'])
    
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Date'])
    return df

# ==================== Calculate Statistics ====================
def calculate_statistics(data, mine_columns):
    stats_dict = {}
    
    for mine in mine_columns:
        if mine in data.columns:
            stats_dict[mine] = {
                'Mean': data[mine].mean(),
                'Std Dev': data[mine].std(),
                'Median': data[mine].median(),
                'IQR': data[mine].quantile(0.75) - data[mine].quantile(0.25)
            }
    
    total_output = data[mine_columns].sum(axis=1)
    stats_dict['Total'] = {
        'Mean': total_output.mean(),
        'Std Dev': total_output.std(),
        'Median': total_output.median(),
        'IQR': total_output.quantile(0.75) - total_output.quantile(0.25)
    }
    
    stats_df = pd.DataFrame(stats_dict).T
    return stats_df

# ==================== Anomaly Detection Methods ====================
def detect_anomalies_iqr(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)

def detect_anomalies_zscore(series, threshold=3.0):
    mean = series.mean()
    std = series.std()
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold

def detect_anomalies_moving_avg(series, window=7, threshold=20.0):
    ma = series.rolling(window=window, center=True).mean()
    ma = ma.fillna(series)
    percent_diff = np.abs((series - ma) / ma * 100)
    return percent_diff > threshold

def detect_anomalies_grubbs(series, alpha=0.05):
    anomalies = pd.Series([False] * len(series), index=series.index)
    data_copy = series.copy()
    
    while len(data_copy) > 2:
        mean = data_copy.mean()
        std = data_copy.std()
        
        if std == 0:
            break
        
        abs_diff = np.abs(data_copy - mean)
        max_idx = abs_diff.idxmax()
        G = abs_diff.max() / std
        
        n = len(data_copy)
        t_dist = scipy_stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
        
        if G > G_critical:
            anomalies[max_idx] = True
            data_copy = data_copy.drop(max_idx)
        else:
            break
    
    return anomalies

def detect_all_anomalies(data, mine_columns, params):
    results = {}
    
    for mine in mine_columns:
        series = data[mine]
        
        results[mine] = {
            'IQR': detect_anomalies_iqr(series, params['iqr_multiplier']),
            'Z-score': detect_anomalies_zscore(series, params['zscore_threshold']),
            'Moving Avg': detect_anomalies_moving_avg(series, params['ma_window'], params['ma_threshold']),
            'Grubbs': detect_anomalies_grubbs(series, params['grubbs_alpha'])
        }
    
    return results

# ==================== Trendline Calculation ====================
def calculate_trendline(x, y, degree):
    x_numeric = np.arange(len(x))
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x_numeric.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return y_pred

# ==================== Create Chart ====================
def create_chart(data, mine_columns, chart_type, anomaly_method, anomaly_results, trendline_degree):
    fig = go.Figure()
    
    if chart_type == "Line":
        for mine in mine_columns:
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data[mine],
                mode='lines',
                name=mine,
                line=dict(width=2)
            ))
            
            anomalies = anomaly_results[mine][anomaly_method]
            if anomalies.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=data.loc[anomalies, 'Date'],
                    y=data.loc[anomalies, mine],
                    mode='markers',
                    name=f'{mine} (Anomalies)',
                    marker=dict(size=10, color='red', symbol='x', line=dict(width=2))
                ))
            
            if trendline_degree > 0:
                y_trend = calculate_trendline(data['Date'], data[mine], trendline_degree)
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=y_trend,
                    mode='lines',
                    name=f'{mine} (Trend)',
                    line=dict(dash='dash', width=2),
                    opacity=0.6
                ))
    
    elif chart_type == "Bar":
        for mine in mine_columns:
            normal_mask = ~anomaly_results[mine][anomaly_method]
            fig.add_trace(go.Bar(
                x=data.loc[normal_mask, 'Date'],
                y=data.loc[normal_mask, mine],
                name=mine,
                marker=dict(opacity=0.7)
            ))
            
            anomalies = anomaly_results[mine][anomaly_method]
            if anomalies.sum() > 0:
                fig.add_trace(go.Bar(
                    x=data.loc[anomalies, 'Date'],
                    y=data.loc[anomalies, mine],
                    name=f'{mine} (Anomalies)',
                    marker=dict(color='red', opacity=0.8)
                ))
            
            if trendline_degree > 0:
                y_trend = calculate_trendline(data['Date'], data[mine], trendline_degree)
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=y_trend,
                    mode='lines',
                    name=f'{mine} (Trend)',
                    line=dict(dash='dash', width=3, color='black')
                ))
    
    elif chart_type == "Stacked":
        for mine in mine_columns:
            fig.add_trace(go.Bar(
                x=data['Date'],
                y=data[mine],
                name=mine
            ))
        
        fig.update_layout(barmode='stack')
    
    fig.update_layout(
        title=f"Mining Operations - {chart_type} Chart",
        xaxis_title="Date",
        yaxis_title="Daily Output",
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig

# ==================== Extract Anomaly Details ====================
def extract_anomaly_details(data, mine_columns, anomaly_results, anomaly_method):
    """Extract detailed information about each anomaly"""
    anomalies_list = []
    
    for mine in mine_columns:
        anomalies_mask = anomaly_results[mine][anomaly_method]
        anomaly_indices = data[anomalies_mask].index
        
        for idx in anomaly_indices:
            date = data.loc[idx, 'Date']
            value = data.loc[idx, mine]
            baseline = data[mine].mean()
            
            # Determine type and magnitude
            if value > baseline:
                anomaly_type = "Spike"
                magnitude = f"+{((value - baseline) / baseline * 100):.1f}%"
            else:
                anomaly_type = "Drop"
                magnitude = f"{((value - baseline) / baseline * 100):.1f}%"
            
            anomalies_list.append({
                'date': date,
                'mine': mine,
                'type': anomaly_type,
                'value': value,
                'baseline': baseline,
                'magnitude': magnitude,
                'method': anomaly_method
            })
    
    # Sort by date
    anomalies_list.sort(key=lambda x: x['date'])
    
    return anomalies_list

# ==================== Generate PDF Report ====================
def generate_pdf_report(data, mine_columns, stats_df, anomaly_results, params, chart_type, anomaly_method, trendline_degree, fig):
    """Generate comprehensive PDF report"""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ========== Cover Page ==========
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.ln(60)
    pdf.cell(0, 10, "Weyland-Yutani Corporation", align='C', ln=True)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "Mining Operations Analysis Report", align='C', ln=True)
    pdf.ln(20)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Period: {data['Date'].min().date()} to {data['Date'].max().date()}", align='C', ln=True)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align='C', ln=True)
    
    # ========== Statistical Summary ==========
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "1. Statistical Summary", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    for entity in list(mine_columns) + ['Total']:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, entity, ln=True)
        
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 6, "Mean Daily Output:", border=1)
        pdf.cell(40, 6, f"{stats_df.loc[entity, 'Mean']:.2f}", border=1)
        pdf.ln()
        
        pdf.cell(50, 6, "Standard Deviation:", border=1)
        pdf.cell(40, 6, f"{stats_df.loc[entity, 'Std Dev']:.2f}", border=1)
        pdf.ln()
        
        pdf.cell(50, 6, "Median:", border=1)
        pdf.cell(40, 6, f"{stats_df.loc[entity, 'Median']:.2f}", border=1)
        pdf.ln()
        
        pdf.cell(50, 6, "IQR:", border=1)
        pdf.cell(40, 6, f"{stats_df.loc[entity, 'IQR']:.2f}", border=1)
        pdf.ln(10)
    
    # ========== Anomaly Detection Summary ==========
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. Anomaly Detection Summary", ln=True)
    pdf.ln(5)
    
    # Detection parameters
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detection Parameters:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"  IQR Multiplier: {params['iqr_multiplier']}", ln=True)
    pdf.cell(0, 6, f"  Z-score Threshold: {params['zscore_threshold']}", ln=True)
    pdf.cell(0, 6, f"  Moving Average Window: {params['ma_window']} days", ln=True)
    pdf.cell(0, 6, f"  MA Distance Threshold: {params['ma_threshold']}%", ln=True)
    pdf.cell(0, 6, f"  Grubbs' Test Alpha: {params['grubbs_alpha']}", ln=True)
    pdf.ln(10)
    
    # Anomaly counts
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Anomalies Detected:", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(40, 6, "Mine", border=1)
    pdf.cell(25, 6, "IQR", border=1, align='C')
    pdf.cell(25, 6, "Z-score", border=1, align='C')
    pdf.cell(30, 6, "Moving Avg", border=1, align='C')
    pdf.cell(25, 6, "Grubbs", border=1, align='C')
    pdf.ln()
    
    pdf.set_font("Arial", "", 10)
    for mine in mine_columns:
        pdf.cell(40, 6, mine, border=1)
        pdf.cell(25, 6, str(anomaly_results[mine]['IQR'].sum()), border=1, align='C')
        pdf.cell(25, 6, str(anomaly_results[mine]['Z-score'].sum()), border=1, align='C')
        pdf.cell(30, 6, str(anomaly_results[mine]['Moving Avg'].sum()), border=1, align='C')
        pdf.cell(25, 6, str(anomaly_results[mine]['Grubbs'].sum()), border=1, align='C')
        pdf.ln()
    
    # ========== Chart ==========
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "3. Data Visualization", ln=True)
    pdf.ln(5)
    
    # Save chart as image
    chart_filename = "temp_chart.png"
    try:
        pio.write_image(fig, chart_filename, width=800, height=600)
        pdf.image(chart_filename, x=10, y=40, w=190)
        os.remove(chart_filename)  # Clean up
    except Exception as e:
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"[Chart could not be embedded: {str(e)}]", ln=True)
    
    # ========== Detailed Anomaly Analysis ==========
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "4. Detailed Anomaly Analysis", ln=True)
    pdf.ln(5)
    
    # Extract anomalies
    anomalies_list = extract_anomaly_details(data, mine_columns, anomaly_results, anomaly_method)
    
    if len(anomalies_list) == 0:
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "No anomalies detected with current parameters.", ln=True)
    else:
        for idx, anomaly in enumerate(anomalies_list, 1):
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"Anomaly #{idx}: {anomaly['type']}", ln=True)
            pdf.ln(2)
            
            pdf.set_font("Arial", "", 10)
            pdf.cell(50, 6, "Date:", border=1)
            pdf.cell(60, 6, str(anomaly['date'].date()), border=1)
            pdf.ln()
            
            pdf.cell(50, 6, "Mine:", border=1)
            pdf.cell(60, 6, anomaly['mine'], border=1)
            pdf.ln()
            
            pdf.cell(50, 6, "Type:", border=1)
            pdf.cell(60, 6, anomaly['type'], border=1)
            pdf.ln()
            
            pdf.cell(50, 6, "Actual Value:", border=1)
            pdf.cell(60, 6, f"{anomaly['value']:.2f}", border=1)
            pdf.ln()
            
            pdf.cell(50, 6, "Baseline (Mean):", border=1)
            pdf.cell(60, 6, f"{anomaly['baseline']:.2f}", border=1)
            pdf.ln()
            
            pdf.cell(50, 6, "Magnitude:", border=1)
            pdf.cell(60, 6, anomaly['magnitude'], border=1)
            pdf.ln()
            
            pdf.cell(50, 6, "Detection Method:", border=1)
            pdf.cell(60, 6, anomaly['method'], border=1)
            pdf.ln(10)
            
            # Page break if needed
            if idx < len(anomalies_list) and pdf.get_y() > 250:
                pdf.add_page()
    
    # Save PDF
    pdf_filename = f"Mining_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_filename)
    
    return pdf_filename

# ==================== Main App ====================
data = load_data()

# Auto-detect mine columns
all_columns = data.columns.tolist()
exclude_keywords = ['Date', 'Event', 'Factor', 'Total']
mine_columns = [col for col in all_columns 
                if col != 'Date' 
                and not any(keyword.lower() in col.lower() for keyword in exclude_keywords)]

if not mine_columns:
    st.error("❌ No mine columns found!")
    st.stop()

# ==================== Title ====================
st.title("⛏️ Mining Operations Dashboard")
st.write(f"**Period:** {data['Date'].min().date()} to {data['Date'].max().date()} | **Total Days:** {len(data)}")

st.markdown("---")

# ==================== Statistics ====================
st.subheader("📈 Statistical Summary")

stats_df = calculate_statistics(data, mine_columns)
all_entities = mine_columns + ['Total']

for entity in all_entities:
    st.write(f"### {entity}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Daily Output", f"{stats_df.loc[entity, 'Mean']:.2f}")
    with col2:
        st.metric("Standard Deviation", f"{stats_df.loc[entity, 'Std Dev']:.2f}")
    with col3:
        st.metric("Median", f"{stats_df.loc[entity, 'Median']:.2f}")
    with col4:
        st.metric("IQR", f"{stats_df.loc[entity, 'IQR']:.2f}")
    
    st.markdown("---")

# ==================== Anomaly Detection ====================
st.subheader("🚨 Anomaly Detection")

st.sidebar.header("🔧 Anomaly Detection Parameters")

params = {
    'iqr_multiplier': st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1),
    'zscore_threshold': st.sidebar.slider("Z-score Threshold", 1.0, 4.0, 3.0, 0.1),
    'ma_window': st.sidebar.slider("Moving Average Window (days)", 3, 14, 7, 1),
    'ma_threshold': st.sidebar.slider("MA Distance Threshold (%)", 5.0, 50.0, 20.0, 1.0),
    'grubbs_alpha': st.sidebar.slider("Grubbs' Test Alpha", 0.01, 0.10, 0.05, 0.01)
}

anomaly_results = detect_all_anomalies(data, mine_columns, params)

st.write("### Anomaly Counts by Method")

for mine in mine_columns:
    st.write(f"**{mine}**")
    
    cols = st.columns(4)
    methods = ['IQR', 'Z-score', 'Moving Avg', 'Grubbs']
    
    for i, method in enumerate(methods):
        with cols[i]:
            count = anomaly_results[mine][method].sum()
            st.metric(method, count)
    
    st.markdown("---")

# ==================== Charts ====================
st.subheader("📊 Interactive Charts")

col1, col2, col3 = st.columns(3)

with col1:
    chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Stacked"])

with col2:
    anomaly_method = st.selectbox("Anomaly Detection Method", ["IQR", "Z-score", "Moving Avg", "Grubbs"])

with col3:
    trendline_degree = st.selectbox("Trendline Degree", [0, 1, 2, 3, 4], format_func=lambda x: "None" if x == 0 else f"Polynomial {x}")

fig = create_chart(data, mine_columns, chart_type, anomaly_method, anomaly_results, trendline_degree)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== PDF Report Generation ====================
st.subheader("📄 Generate PDF Report")

st.write("Generate a comprehensive PDF report including all statistics, charts, and detailed anomaly analysis.")

if st.button("📥 Generate PDF Report", type="primary"):
    with st.spinner("Generating PDF report..."):
        try:
            pdf_filename = generate_pdf_report(
                data, 
                mine_columns, 
                stats_df, 
                anomaly_results, 
                params, 
                chart_type, 
                anomaly_method, 
                trendline_degree, 
                fig
            )
            
            # Read PDF file
            with open(pdf_filename, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Provide download button
            st.success("✅ PDF report generated successfully!")
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf"
            )
            
            # Clean up
            os.remove(pdf_filename)
            
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            st.write("Please make sure kaleido is installed: `pip install kaleido`")