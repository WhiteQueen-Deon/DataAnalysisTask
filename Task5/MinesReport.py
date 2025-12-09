import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
        'Mean': np.mean([stats_dict[m]['Mean'] for m in mine_columns]),
        'Std Dev': np.mean([stats_dict[m]['Std Dev'] for m in mine_columns]),
        'Median': np.mean([stats_dict[m]['Median'] for m in mine_columns]),
        'IQR': np.mean([stats_dict[m]['IQR'] for m in mine_columns])
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

def t_ppf(p, df):
    a = 1.0 / (df - 0.5)
    b = 48.0 / (a * a)
    c = (((20700.0 * a) / b - 98.0) * a) - 16.0
    d = (((94.5 / (b + c)) * a) - 3.0) * a + 1.0
    x = d * np.tan(np.pi * (p - 0.5))

    for _ in range(2):
        x = x - (cdf_t(x, df) - p) / pdf_t(x, df)

    return x

def pdf_t(x, df):
    return (1 + x**2/df)**(-(df+1)/2)

def cdf_t(x, df):
    from math import atan, pi
    if x == 0:
        return 0.5
    t = atan(x / np.sqrt(df))
    return 0.5 + t / pi


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
        t_dist = t_ppf(1 - alpha / (2 * n), n - 2)
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
    """Create interactive chart with anomalies and trendline"""
    
    fig = go.Figure()
    
    # Define colors for mines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    if chart_type == "Line":
        # First pass: Add all normal lines
        for i, mine in enumerate(mine_columns):
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data[mine],
                mode='lines',
                name=mine,
                line=dict(width=2, color=colors[i % len(colors)]),
                legendgroup=mine
            ))
        
        # Second pass: Add all anomalies (single legend entry)
        all_anomaly_dates = []
        all_anomaly_values = []
        all_anomaly_mines = []
        
        for mine in mine_columns:
            anomalies = anomaly_results[mine][anomaly_method]
            if anomalies.sum() > 0:
                all_anomaly_dates.extend(data.loc[anomalies, 'Date'].tolist())
                all_anomaly_values.extend(data.loc[anomalies, mine].tolist())
                all_anomaly_mines.extend([mine] * anomalies.sum())
        
        # Add single anomaly trace
        if len(all_anomaly_dates) > 0:
            fig.add_trace(go.Scatter(
                x=all_anomaly_dates,
                y=all_anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(size=10, color='red', symbol='x', line=dict(width=2)),
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>',
                text=all_anomaly_mines,
                showlegend=True
            ))
        
        # Add trendlines
        if trendline_degree > 0:
            for i, mine in enumerate(mine_columns):
                y_trend = calculate_trendline(data['Date'], data[mine], trendline_degree)
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=y_trend,
                    mode='lines',
                    name=f'{mine} Trend',
                    line=dict(dash='dash', width=2, color=colors[i % len(colors)]),
                    opacity=0.6,
                    legendgroup=mine,
                    showlegend=False 
                ))
    
    elif chart_type == "Bar":
        # Bar chart: all bars grouped, markers manually offset
        
        # Add all bars with offset groups
        for i, mine in enumerate(mine_columns):
            fig.add_trace(go.Bar(
                x=data['Date'],
                y=data[mine],
                name=mine,
                marker=dict(color=colors[i % len(colors)], opacity=0.7),
                legendgroup=mine
            ))
        
        # Calculate bar positions for markers
        num_mines = len(mine_columns)
        
        # Plotly bar width calculation (approximation)
        dates = data['Date'].unique()
        if len(dates) > 1:
            # Calculate average time gap between dates
            time_diffs = np.diff(dates.astype('int64'))
            avg_gap_days = np.mean(time_diffs) / (10**9 * 86400)  # Convert nanoseconds to days
            
            # Bar width as fraction of gap
            total_bar_width = avg_gap_days * 0.8  # 80% of gap
            single_bar_width = total_bar_width / num_mines
            
            # Calculate offsets for each mine
            offsets = []
            for i in range(num_mines):
                offset = (i - (num_mines - 1) / 2) * single_bar_width
                offsets.append(pd.Timedelta(days=offset))
        else:
            offsets = [pd.Timedelta(days=0)] * num_mines
        
        # Add anomaly markers with offsets
        anomaly_shown = False
        
        for idx, mine in enumerate(mine_columns):
            anomalies = anomaly_results[mine][anomaly_method]
            if anomalies.sum() > 0:
                anomaly_dates = data.loc[anomalies, 'Date']
                anomaly_values = data.loc[anomalies, mine]
                
                # Apply horizontal offset
                anomaly_dates_offset = anomaly_dates + offsets[idx]
                
                # Position markers above bars
                y_positions = anomaly_values
                
                fig.add_trace(go.Scatter(
                    x=anomaly_dates_offset,
                    y=y_positions,
                    mode='markers',
                    name='Anomalies' if not anomaly_shown else '',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='x',
                        line=dict(width=3, color='darkred')
                    ),
                    showlegend=(not anomaly_shown),
                    customdata=anomaly_values,
                    hovertemplate=f'<b>{mine} - Anomaly</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{customdata:.2f}}<extra></extra>',
                    legendgroup='anomalies'
                ))
                
                anomaly_shown = True
        
        # Set barmode to group
        fig.update_layout(
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1
        )
        
        # Add trendlines
        if trendline_degree > 0:
            for i, mine in enumerate(mine_columns):
                y_trend = calculate_trendline(data['Date'], data[mine], trendline_degree)
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=y_trend,
                    mode='lines',
                    name=f'{mine} Trend',
                    line=dict(dash='dash', width=3, color=colors[i % len(colors)]),
                    showlegend=False
                ))
                
    elif chart_type == "Stacked":
        for i, mine in enumerate(mine_columns):
            fig.add_trace(go.Bar(
                x=data['Date'],
                y=data[mine],
                name=mine,
                marker=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(barmode='stack')
    
    # Update layout
    fig.update_layout(
        title=f"Mining Operations - {chart_type} Chart ({anomaly_method} Detection)",
        xaxis_title="",
        yaxis_title="Daily Output",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(t=20, b=40, l=50, r=10) 
    )
    
    return fig

def generate_pdf_report(data, mine_columns, stats_df, anomaly_results, params, trendline_degree):
    """Generate simplified PDF report with statistics, anomaly detection, and charts"""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ========== Page 1: Cover + Statistics ==========
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Mining Operations Report", align='C', ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Period: {data['Date'].min().date()} to {data['Date'].max().date()}", align='C', ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align='C', ln=True)
    pdf.ln(10)
    
    # Section 1: Statistics
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Statistical Summary", ln=True)
    pdf.ln(3)
    
    # Table header
    pdf.set_font("Arial", "B", 10)
    pdf.cell(45, 8, "Mine", border=1, align='C')
    pdf.cell(35, 8, "Mean", border=1, align='C')
    pdf.cell(35, 8, "Std Dev", border=1, align='C')
    pdf.cell(35, 8, "Median", border=1, align='C')
    pdf.cell(35, 8, "IQR", border=1, align='C')
    pdf.ln()
    
    # Table data
    pdf.set_font("Arial", "", 10)
    for entity in list(mine_columns) + ['Total']:
        pdf.cell(45, 8, entity, border=1)
        pdf.cell(35, 8, f"{stats_df.loc[entity, 'Mean']:.2f}", border=1, align='C')
        pdf.cell(35, 8, f"{stats_df.loc[entity, 'Std Dev']:.2f}", border=1, align='C')
        pdf.cell(35, 8, f"{stats_df.loc[entity, 'Median']:.2f}", border=1, align='C')
        pdf.cell(35, 8, f"{stats_df.loc[entity, 'IQR']:.2f}", border=1, align='C')
        pdf.ln()
    
    pdf.ln(10)
    
    # ========== Section 2: Anomaly Detection ==========
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "2. Anomaly Detection", ln=True)
    pdf.ln(3)
    
    # Parameters
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Detection Parameters:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"  IQR Multiplier: {params['iqr_multiplier']}", ln=True)
    pdf.cell(0, 6, f"  Z-score Threshold: {params['zscore_threshold']}", ln=True)
    pdf.cell(0, 6, f"  Moving Average Window: {params['ma_window']} days", ln=True)
    pdf.cell(0, 6, f"  MA Distance Threshold: {params['ma_threshold']}%", ln=True)
    pdf.cell(0, 6, f"  Grubbs' Alpha: {params['grubbs_alpha']}", ln=True)
    pdf.ln(5)
    
    # Anomaly counts table
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Anomalies Detected:", ln=True)
    pdf.ln(2)
    
    # Table header
    pdf.set_font("Arial", "B", 9)
    pdf.cell(45, 7, "Mine", border=1, align='C')
    pdf.cell(30, 7, "IQR", border=1, align='C')
    pdf.cell(30, 7, "Z-score", border=1, align='C')
    pdf.cell(35, 7, "Moving Avg", border=1, align='C')
    pdf.cell(30, 7, "Grubbs", border=1, align='C')
    pdf.ln()
    
    # Table data
    pdf.set_font("Arial", "", 9)
    for mine in mine_columns:
        pdf.cell(45, 7, mine, border=1)
        pdf.cell(30, 7, str(anomaly_results[mine]['IQR'].sum()), border=1, align='C')
        pdf.cell(30, 7, str(anomaly_results[mine]['Z-score'].sum()), border=1, align='C')
        pdf.cell(35, 7, str(anomaly_results[mine]['Moving Avg'].sum()), border=1, align='C')
        pdf.cell(30, 7, str(anomaly_results[mine]['Grubbs'].sum()), border=1, align='C')
        pdf.ln()
    
# ========== Section 3: Charts ==========
    # For each chart type, show all 4 detection methods vertically (1x4)
    
    chart_types = ['Line', 'Bar', 'Stacked']
    detection_methods = ['IQR', 'Z-score', 'Moving Avg', 'Grubbs']
    
    for chart_type in chart_types:
        pdf.add_page()
        
        # Page title
        pdf.set_font("Arial", "B", 14)
        if trendline_degree == 0:
            pdf.cell(0, 10, f"3. {chart_type} Charts (No Trendline)", ln=True)
        else:
            pdf.cell(0, 10, f"3. {chart_type} Charts (Polynomial Degree {trendline_degree})", ln=True)
        pdf.ln(5)
        
        # Generate 4 charts vertically
        for idx, method in enumerate(detection_methods):
            # Create chart
            fig = create_chart(data, mine_columns, chart_type, method, anomaly_results, trendline_degree)
            
            # Remove chart title
            fig.update_layout(title="")  
            
            # Save as image
            chart_filename = f"temp_{chart_type}_{method}.png"
            
            try:
                img_bytes = fig.to_image(format="png", width=1000, height=320)
                with open(chart_filename, "wb") as f:
                    f.write(img_bytes)
                
                # Method label (closer to chart)
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 4, method, ln=True, align='C')  
                
                # Add chart image (full width)
                pdf.image(chart_filename, x=10, y=pdf.get_y(), w=190)
                
                # Clean up
                os.remove(chart_filename)
                
                # Space before next chart
                pdf.ln(63)  
                
            except Exception as e:
                pdf.set_font("Arial", "", 9)
                pdf.cell(0, 5, f"[Chart error: {str(e)}]", ln=True)
                pdf.ln(5)
    
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

 #==================== PDF Report Generation ====================
st.subheader("📄 Generate PDF Report")

st.write("Generate a PDF report with statistics, anomaly detection results, and charts.")

if st.button("📥 Generate PDF Report", type="primary"):
    with st.spinner("Generating PDF report..."):
        try:
            pdf_filename = generate_pdf_report(
                data, 
                mine_columns, 
                stats_df, 
                anomaly_results, 
                params,
                trendline_degree
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
            st.write("Error details:", str(e))