import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analysis Dashboard")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📁 DATA1", "📁 DATA2", "📁 DATA3"])

# ==================== DATA1 Tab ====================
with tab1:
    st.header("DATA1 Analysis Results")
    
    # Task 1: Top 5 Days 
    st.subheader("📅 Task 1: Top 5 Revenue Days")
    st.markdown("""
    | Rank | Date | Revenue ($) |
    |------|------|-------------|
    | 1 | 2024-12-17 | $57011.458 |
    | 2 | 2024-11-03 | $46258.646 |
    | 3 | 2025-03-23 | $39120.974 |
    | 4 | 2024-09-06 | $32795.310 |
    | 5 | 2025-01-25 | $31732.458 |
    """)
    
    st.markdown("---")
    
    # Task 2-5: 
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Task 2: Unique Users",
            value="3066",
            delta="Real unique users count"
        )
    
    with col2:
        st.metric(
            label="✍️ Task 3: Author Sets",
            value="325",
            delta="unique sets of authors"
        )
    
    with col3:
        st.metric(
            label="⭐ Task 4: Top Author",
            value="Arlinda Huel",
            delta="most popular author set"
        )
    
    with col4:
        st.metric(
            label="💰 Task 5: Top Customer",
            value="$37609.70",
            delta="Total spending"
        )
    
    st.markdown("---")
    
    # Task 4 
    st.subheader("⭐ Task 4: Most Popular Author")
    st.info("**Author:** Arlinda Huel  \n**Books Sold:** 201")
    
    # Task 5 
    st.subheader("💰 Task 5: Top Customer Details")
    st.success("**User IDs:** [45800]  \n**Total Spending:** $37609.70")
    
    st.markdown("---")
    
    # Task 6: Chart
    st.subheader("📈 Task 6: Daily Revenue Over Time")
    try:
        st.image(os.path.join(BASE_DIR, "daily_revenue_DATA1.png"))
    except:
        st.error("⚠️ Chart not found. Please ensure 'daily_revenue_DATA1.png' is in the same directory.")

# ==================== DATA2 Tab ====================
with tab2:
    st.header("DATA2 Analysis Results")
    
    st.subheader("📅 Task 1: Top 5 Revenue Days")
    st.markdown("""
    | Rank | Date | Revenue ($) |
    |------|------|-------------|
    | 1 | 2024-12-24 | $42137.010 |
    | 2 | 2024-08-29 | $40556.078 |
    | 3 | 2024-12-29 | $39297.212 |
    | 4 | 2025-01-30 | $39021.688 |
    | 5 | 2024-11-29 | $35207.050 |
    """)
    
    st.markdown("---")
    
    # Task 2-5: 
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Task 2: Unique Users",
            value="2633",
            delta="Real unique users count"
        )
    
    with col2:
        st.metric(
            label="✍️ Task 3: Author Sets",
            value="293",
            delta="unique sets of authors"
        )
    
    with col3:
        st.metric(
            label="⭐ Task 4: Top Author",
            value="Hershel Treutel & Miss Modesto Denesik & Sen. Trula Bosco",
            delta="most popular author set"
        )
    
    with col4:
        st.metric(
            label="💰 Task 5: Top Customer",
            value="$37051.25",
            delta="Total spending"
        )
    
    st.markdown("---")
    
    # Task 4 
    st.subheader("⭐ Task 4: Most Popular Author")
    st.info("**Author:** Hershel Treutel & Miss Modesto Denesik & Sen. Trula Bosco  \n**Books Sold:** 163")
    
    # Task 5 
    st.subheader("💰 Task 5: Top Customer Details")
    st.success("**User IDs:** [53256]  \n**Total Spending:** $37051.25")
    
    st.markdown("---")
    
    # Task 6: Chart
    st.subheader("📈 Task 6: Daily Revenue Over Time")
    try:
        st.image(os.path.join(BASE_DIR, "daily_revenue_DATA2.png"))
    except:
        st.error("⚠️ Chart not found. Please ensure 'daily_revenue_DATA1.png' is in the same directory.")


# ==================== DATA3 Tab ====================
with tab3:
    st.header("DATA3 Analysis Results")
    
    st.subheader("📅 Task 1: Top 5 Revenue Days")
    st.markdown("""
    | Rank | Date | Revenue ($) |
    |------|------|-------------|
    | 1 | 2025-02-03 | $63761.338 |
    | 2 | 2024-06-09 | $42015.360 |
    | 3 | 2024-07-26 | $38903.900 |
    | 4 | 2024-11-03 | $32517.200 |
    | 5 | 2024-09-20 | $30660.500 |
    """)
    
    st.markdown("---")
    
    # Task 2-5: 
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Task 2: Unique Users",
            value="3232",
            delta="Real unique users count"
        )
    
    with col2:
        st.metric(
            label="✍️ Task 3: Author Sets",
            value="268",
            delta="unique sets of authors"
        )
    
    with col3:
        st.metric(
            label="⭐ Task 4: Top Author",
            value="Coy Streich & Keeley Hand & Lela Emard",
            delta="most popular author set"
        )
    
    with col4:
        st.metric(
            label="💰 Task 5: Top Customer",
            value="$44582.89",
            delta="Total spending"
        )
    
    st.markdown("---")
    
    # Task 4 
    st.subheader("⭐ Task 4: Most Popular Author")
    st.info("**Author:** Coy Streich & Keeley Hand & Lela Emard  \n**Books Sold:** 159")
    
    # Task 5 
    st.subheader("💰 Task 5: Top Customer Details")
    st.success("**User IDs:** [49002, 49414]  \n**Total Spending:** $44582.89")
    
    st.markdown("---")
    
    # Task 6: Chart
    st.subheader("📈 Task 6: Daily Revenue Over Time")
    try:
        st.image(os.path.join(BASE_DIR, "daily_revenue_DATA3.png"))
    except:
        st.error("⚠️ Chart not found. Please ensure 'daily_revenue_DATA1.png' is in the same directory.")


st.markdown("---")
st.caption("📊 Data Analysis Dashboard | Generated for Contemporary Data Processing Systems Course")