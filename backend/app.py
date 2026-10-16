"""
app.py - ProfitGuard AI
An Intelligent Revenue Leakage Detection & Business Decision-Support Platform
Datathon 2026 - Track 4 (Business / The Money Leak)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ml import (
    KMeansFromScratch,
    StandardScalerFromScratch,
    compute_silhouette_score_sample,
    extract_rfm_features
)

st.set_page_config(
    page_title="ProfitGuard AI - Revenue Leakage & Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .leakage-badge {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .safe-badge {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_data():
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(base_dir):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    products_path = os.path.join(base_dir, "products.csv")
    transactions_path = os.path.join(base_dir, "transactions.csv")
    customers_path = os.path.join(base_dir, "customers.csv")
    inventory_path = os.path.join(base_dir, "inventory.csv")
    campaigns_path = os.path.join(base_dir, "campaigns.csv")

    products = pd.read_csv(products_path)
    transactions = pd.read_csv(transactions_path)
    customers = pd.read_csv(customers_path) if os.path.exists(customers_path) else pd.DataFrame()
    inventory = pd.read_csv(inventory_path) if os.path.exists(inventory_path) else pd.DataFrame()
    campaigns = pd.read_csv(campaigns_path) if os.path.exists(campaigns_path) else pd.DataFrame()

    # Preprocessing
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    # Merge transactions with products to get unit_cost and category
    df = transactions.merge(
        products[['product_id', 'category', 'unit_cost', 'list_price', 'supplier_lead_days']],
        on='product_id',
        how='left'
    )

    # Unit economics calculations:
    # Effective selling price per unit = unit_price * (1 - discount_pct)
    # Unit margin = effective_selling_price - unit_cost
    # Total margin = quantity * unit_margin
    # Total gross revenue = quantity * effective_selling_price
    df['effective_unit_price'] = df['unit_price'] * (1.0 - df['discount_pct'])
    df['unit_margin'] = df['effective_unit_price'] - df['unit_cost']
    df['total_margin'] = df['quantity'] * df['unit_margin']
    df['gross_revenue'] = df['quantity'] * df['effective_unit_price']

    # Flag leakage for completed transactions
    df['is_leakage'] = (df['status'] == 'Completed') & (df['total_margin'] < 0)

    # Root cause classification
    def diagnose_leakage(row):
        if not row['is_leakage']:
            return "Profitable / Healthy"
        if row['unit_price'] < row['unit_cost']:
            return "Price Below Cost (Base Price < Cost)"
        else:
            return "Discount Review (Discount takes estimated price below cost)"

    df['leakage_reason'] = df.apply(diagnose_leakage, axis=1)

    return df, products, customers, inventory, campaigns


# Load data
df, products_df, customers_df, inventory_df, campaigns_df = load_and_preprocess_data()

# Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("ProfitGuard AI")
st.sidebar.caption("Intelligent Revenue Leakage Detection & Decision Support")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Executive Overview",
        "2. Revenue Leakage Detector",
        "3. Customer Intelligence (ML)",
        "4. Inventory & Marketing Diagnostics",
        "5. Data Quality & Assumptions"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "**Datathon 2026**\n\n"
    "Track 4: Business (The Money Leak)\n\n"
    "**Model:** NumPy K-Means from scratch\n"
    "**Method:** Unit Economics Auditing"
)

# -------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if page == "1. Executive Overview":
    st.title("📊 Executive Business Overview")
    st.write("Dataset-level review of transaction economics, completed sales, and transactions with estimated negative gross margin.")

    completed_df = df[df['status'] == 'Completed']
    total_tx = len(df)
    completed_tx = len(completed_df)
    total_rev = completed_df['gross_revenue'].sum()
    leakage_tx_count = completed_df['is_leakage'].sum()
    leakage_pct = (leakage_tx_count / completed_tx * 100) if completed_tx else 0
    total_leakage_loss = abs(completed_df[completed_df['is_leakage']]['total_margin'].sum())
    returned_tx = len(df[df['status'] == 'Returned'])

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transactions", f"{total_tx:,}")
    with col2:
        st.metric("Completed Orders", f"{completed_tx:,}")
    with col3:
        st.metric("Completed Revenue", f"₹{total_rev/1e5:,.2f} Lakhs", help=f"₹{total_rev:,.2f}")
    with col4:
        st.metric("Negative-Margin Orders", f"{leakage_tx_count:,} ({leakage_pct:.1f}%)", help="Calculated from supplied synthetic prices, discounts, quantities, and product costs.")
    with col5:
        st.metric("Estimated Negative Gross Margin", f"₹{total_leakage_loss/1e5:,.2f} Lakhs", help="An estimate for investigation; not proof of recoverable revenue.")

    st.divider()

    # Visual Insights Row
    g1, g2 = st.columns([3, 2])

    with g1:
        st.subheader("Revenue vs. Profit Margin by Category")
        cat_agg = completed_df.groupby('category').agg(
            revenue=('gross_revenue', 'sum'),
            margin=('total_margin', 'sum')
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat_agg['category'], y=cat_agg['revenue'],
            name='Completed Revenue (₹)', marker_color='#3b82f6'
        ))
        fig.add_trace(go.Bar(
            x=cat_agg['category'], y=cat_agg['margin'],
            name='Net Estimated Margin (₹)', marker_color='#10b981'
        ))
        fig.update_layout(
            barmode='group',
            xaxis_title="Product Category",
            yaxis_title="Amount (₹)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.subheader("Transaction Status Breakdown")
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_donut = px.pie(
            status_counts, values='Count', names='Status',
            hole=0.45,
            color='Status',
            color_discrete_map={
                'Completed': '#10b981',
                'Returned': '#f59e0b',
                'Cancelled': '#ef4444',
                'Pending': '#64748b'
            }
        )
        fig_donut.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("🚨 Priority Review: 10 Largest Negative-Margin Transactions")
    st.write("Completed transactions with the lowest estimated margins:")
    leakage_sorted = completed_df[completed_df['is_leakage']].sort_values(by='total_margin').head(10)
    display_cols = [
        'transaction_id', 'product_id', 'category', 'quantity',
        'unit_price', 'discount_pct', 'unit_cost', 'total_margin', 'leakage_reason'
    ]
    st.dataframe(
        leakage_sorted[display_cols].style.format({
            'unit_price': '₹{:.2f}',
            'unit_cost': '₹{:.2f}',
            'discount_pct': '{:.1%}',
            'total_margin': '₹{:.2f}'
        }),
        use_container_width=True
    )

# -------------------------------------------------------------
# PAGE 2: REVENUE LEAKAGE DETECTOR
# -------------------------------------------------------------
elif page == "2. Revenue Leakage Detector":
    st.title("🔍 Transaction Margin Review")
    st.write("Inspect individual transactions, understand unit economics, and see transparent mathematical explanations.")

    completed_df = df[df['status'] == 'Completed']
    leakage_df = completed_df[completed_df['is_leakage']]

    c1, c2 = st.columns([1, 2])
    with c1:
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(completed_df['category'].dropna().unique().tolist())
        )
        leakage_reason_filter = st.selectbox(
            "Filter by Review Signal",
            ["All", "Price Below Cost (Base Price < Cost)", "Discount Review (Discount takes estimated price below cost)"]
        )

    filtered = leakage_df
    if category_filter != "All":
        filtered = filtered[filtered['category'] == category_filter]
    if leakage_reason_filter != "All":
        filtered = filtered[filtered['leakage_reason'] == leakage_reason_filter]

    st.write(f"Showing **{len(filtered):,}** flagged transactions matching criteria:")

    # Transaction selection
    selected_tx_id = st.selectbox(
        "Select a Flagged Transaction ID to Audit",
        filtered['transaction_id'].tolist() if len(filtered) > 0 else []
    )

    if selected_tx_id:
        tx = df[df['transaction_id'] == selected_tx_id].iloc[0]

        st.markdown(f"### 📋 Audit Report: `{tx['transaction_id']}` (Product `{tx['product_id']}`)")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Quantity Sold (Q)", f"{int(tx['quantity'])} units")
        k2.metric("Base Unit Price (P)", f"₹{tx['unit_price']:.2f}")
        k3.metric("Discount Applied (D)", f"{tx['discount_pct']*100:.1f}%")
        k4.metric("Product Unit Cost (C)", f"₹{tx['unit_cost']:.2f}")

        st.divider()

        calc_col, alert_col = st.columns([3, 2])

        with calc_col:
            st.markdown("#### 📐 Mathematical Step-by-Step Breakdown")
            st.latex(r"\text{Estimated Margin} = Q \times [P(1 - D) - C]")
            eff_p = tx['effective_unit_price']
            u_cost = tx['unit_cost']
            u_margin = tx['unit_margin']
            tot_margin = tx['total_margin']

            st.markdown(f"""
            1. **Effective Selling Price per Unit:**  
               $P(1 - D) = {tx['unit_price']:.2f} \times (1 - {tx['discount_pct']:.2f}) =$ **₹{eff_p:.2f}**
            2. **Unit Margin:**  
               $P(1 - D) - C = {eff_p:.2f} - {u_cost:.2f} =$ **₹{u_margin:.2f} per unit**
            3. **Total Transaction Margin:**  
               $Q \times [P(1 - D) - C] = {int(tx['quantity'])} \times {u_margin:.2f} =$ **₹{tot_margin:.2f}**
            """)

        with alert_col:
            st.markdown("#### 🛡️ ProfitGuard Diagnostic")
            if tx['unit_price'] < tx['unit_cost']:
                st.error(
                    f"**FLAG: Price Below Cost**\n\n"
                    f"The recorded unit price (₹{tx['unit_price']:.2f}) was ALREADY lower than unit cost (₹{tx['unit_cost']:.2f}) before applying discount!\n\n"
                    f"This is a review signal, not proof of an error. Check the cost basis, approved promotion, clearance plan, and basket context before acting."
                )
            else:
                st.warning(
                    f"**FLAG: Discount reduces estimated price below cost**\n\n"
                    f"The recorded unit price (₹{tx['unit_price']:.2f}) was above cost (₹{tx['unit_cost']:.2f}), but the {tx['discount_pct']*100:.1f}% discount brings the estimated selling price below cost.\n\n"
                    f"Review the discount, cost basis, promotion approval, and basket context before taking action. This prototype does not enforce checkout rules."
                )

# -------------------------------------------------------------
# PAGE 3: CUSTOMER INTELLIGENCE (ML)
# -------------------------------------------------------------
elif page == "3. Customer Intelligence (ML)":
    st.title("🧠 Customer Profile & RFM Exploration")
    st.write("Exploratory customer grouping with a from-scratch **NumPy K-Means** model; profile fields provide context but are not model inputs.")

    with st.expander("ℹ️ ML Model Architecture & Technical Details", expanded=False):
        st.markdown("""
        - **Algorithm:** K-Means from Scratch using vectorized NumPy (`KMeansFromScratch`)
        - **Features (RFM):**
          - **Recency ($R$):** Days since customer's last completed purchase
          - **Frequency ($F$):** Total count of completed orders
        - **Monetary ($M$):** Total completed-order sales value (₹), before product cost
        - **Optimization:** Vectorized Euclidean distance, convergence checking ($\\Delta < 10^{-4}$), $N=5$ random restarts to reduce local-minimum sensitivity.
        - **Validation:** Sample-based Silhouette Score calculation ($N=1,000$).
        """)

    with st.spinner("Extracting RFM features from transactions..."):
        rfm_df = extract_rfm_features(df)
        profile_columns = [c for c in ["customer_id", "region", "customer_segment", "preferred_channel"] if c in customers_df.columns]
        if len(profile_columns) > 1:
            rfm_df = rfm_df.merge(customers_df[profile_columns].drop_duplicates("customer_id"), on="customer_id", how="left")

    col_ctrl, col_stats = st.columns([1, 2])
    with col_ctrl:
        k_val = st.slider("Select Number of Clusters ($K$)", min_value=2, max_value=6, value=4)
        n_restarts = st.slider("Random Restarts (n_init)", min_value=1, max_value=10, value=5)
        run_btn = st.button("🚀 Train K-Means Model", type="primary")

    # Fit Model
    scaler = StandardScalerFromScratch()
    X_raw = rfm_df[['recency', 'frequency', 'monetary']].values
    X_scaled = scaler.fit_transform(X_raw)

    kmeans = KMeansFromScratch(n_clusters=k_val, n_init=n_restarts, random_state=42)
    kmeans.fit(X_scaled)
    rfm_df['Cluster'] = [f"Cluster {c}" for c in kmeans.labels]

    sil_score = compute_silhouette_score_sample(X_scaled, kmeans.labels, sample_size=1000)

    with col_stats:
        m1, m2, m3 = st.columns(3)
        m1.metric("Analyzed Customers", f"{len(rfm_df):,}")
        m2.metric("Model Inertia (WCSS)", f"{kmeans.inertia_:,.1f}")
        m3.metric("Sample Silhouette Score", f"{sil_score:.3f}", help="Range: -1 to +1. Higher is better separated.")

    st.divider()

    # Visualizations
    v1, v2 = st.columns([3, 2])
    with v1:
        st.subheader("Customer Clusters: Recency vs. Monetary Spend")
        fig_scatter = px.scatter(
            rfm_df,
            x='recency',
            y='monetary',
            color='Cluster',
            size='frequency',
            hover_data=[c for c in ['customer_id', 'region', 'customer_segment', 'preferred_channel', 'frequency', 'monetary', 'recency'] if c in rfm_df.columns],
            labels={'recency': 'Recency (Days Ago)', 'monetary': 'Total Monetary Spend (₹)'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with v2:
        st.subheader("Cluster Profile & Interpretation")
        cluster_summary = rfm_df.groupby('Cluster').agg(
            Customers=('customer_id', 'count'),
            Avg_Recency=('recency', 'mean'),
            Avg_Frequency=('frequency', 'mean'),
            Avg_Spend=('monetary', 'mean')
        ).reset_index()

        st.dataframe(
            cluster_summary.style.format({
                'Avg_Recency': '{:.0f} days',
                'Avg_Frequency': '{:.1f} orders',
                'Avg_Spend': '₹{:.2f}'
            }),
            use_container_width=True
        )

        st.info(
            "**Strategic Business Takeaway:**\n"
            "- Low Recency + High Spend $\\to$ **High-Value Champions** (Protect with VIP retention).\n"
            "- High Recency + High Spend $\\to$ **At-Risk Valuable Customers** (Re-engage with winback campaigns).\n"
            "- Low Recency + Low Spend $\\to$ **New / Developing Customers** (Nurture with cross-sells)."
        )

# -------------------------------------------------------------
# PAGE 4: INVENTORY & MARKETING DIAGNOSTICS
# -------------------------------------------------------------
elif page == "4. Inventory & Marketing Diagnostics":
    st.title("📦 Inventory & Marketing Diagnostics")
    st.write("Cross-system visibility into supply chain delays and advertising performance.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Inventory Stockout Risk")
        if not inventory_df.empty:
            stockouts = inventory_df[inventory_df['stockout_flag'] == 1]
            st.metric("Total Stockout Events Flagged", f"{len(stockouts):,}")
            st.metric("Avg Supplier Lead Delay", f"{inventory_df['supplier_delay_days'].mean():.1f} days")
            st.dataframe(inventory_df.head(10), use_container_width=True)
        else:
            st.info("Inventory dataset not available.")

    with c2:
        st.subheader("Marketing Campaign Performance")
        if not campaigns_df.empty:
            camp_agg = campaigns_df.groupby('channel').agg(
                Spend=('spend', 'sum'),
                Conversions=('conversions', 'sum')
            ).reset_index()
            camp_agg['Cost_Per_Conversion'] = camp_agg['Spend'] / camp_agg['Conversions']

            st.dataframe(camp_agg.style.format({
                'Spend': '₹{:,.2f}',
                'Conversions': '{:,.0f}',
                'Cost_Per_Conversion': '₹{:.2f}'
            }), use_container_width=True)
        else:
            st.info("Campaigns dataset not available.")

# -------------------------------------------------------------
# PAGE 5: DATA QUALITY & ASSUMPTIONS
# -------------------------------------------------------------
elif page == "5. Data Quality & Assumptions":
    st.title("📑 Data Quality, Limitations & Assumptions")
    st.markdown("""
    ### 🔬 Dataset Profile and Assumptions
    These results use the supplied synthetic educational CSVs. They describe this dataset only and should not be treated as evidence of real-world performance.

    1. **Observed Negative Margins:**
       **5,315 completed transactions** have a negative estimated gross margin under the formula below. This count does not establish why the values were generated or whether the transactions were errors.
    2. **Discount Semantics Assumption:**  
       We assume `unit_price` in `transactions.csv` represents the list/pre-discount price, and `discount_pct` is applied at purchase time:
       $$\\text{Effective Selling Price} = P \\times (1 - D)$$
    3. **Data Coverage:**
       Transactions are joined to product costs; customer region, supplied segment, and preferred channel enrich the RFM view. Inventory records are summarized for stockout and supplier-delay review. Campaign records are summarized by channel using spend and conversions. These are descriptive joins, not causal attribution.
    4. **Prototype Scope:**
       The app provides review flags and descriptive analytics. It does not predict churn, recommend a validated optimal price, connect to a checkout system, or send external alerts.
    """)
