import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from login import check_login, logout_button

st.set_page_config(page_title="E-commerce Analytics Dashboard", page_icon="🧸", layout="wide")

if not check_login():
    st.stop()
logout_button()

MODEL_PATH = "conversion_model.pkl"
DATA_DIR = "data"


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def q(text):
    st.markdown(f"**Q: {text}**")


def conclude(points):
    st.markdown("**Conclusion:**")
    for p in points:
        st.markdown(f"- {p}")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# Data load + cleaning (same logic as the notebook, cached so it only
# hits SQL Server once per session)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    sessions = pd.read_parquet(f"{DATA_DIR}/sessions.parquet")
    orders = pd.read_parquet(f"{DATA_DIR}/orders.parquet")
    order_items = pd.read_parquet(f"{DATA_DIR}/order_items.parquet")
    products = pd.read_parquet(f"{DATA_DIR}/products.parquet")
    refunds = pd.read_parquet(f"{DATA_DIR}/refunds.parquet")
    website_pageviews = pd.read_parquet(f"{DATA_DIR}/website_pageviews.parquet")

    # --- Outlier handling (IQR capping) ---
    for col in ['price_usd', 'cogs_usd', 'items_purchased']:
        q1, q3 = orders[col].quantile(0.25), orders[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
        orders[col] = orders[col].clip(lower=lower, upper=upper)

    for col in ['price_usd', 'cogs_usd']:
        q1, q3 = order_items[col].quantile(0.25), order_items[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
        order_items[col] = order_items[col].clip(lower=lower, upper=upper)

    q1, q3 = refunds['refund_amount_usd'].quantile(0.25), refunds['refund_amount_usd'].quantile(0.75)
    iqr = q3 - q1
    lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
    refunds['refund_amount_usd'] = refunds['refund_amount_usd'].clip(lower=lower, upper=upper)
    refunds['created_at'] = pd.to_datetime(refunds['created_at'])

    return sessions, orders, order_items, products, refunds, website_pageviews


st.title("🧸 E-commerce Analytics Dashboard")
st.caption("Digital analytics case study — descriptive, diagnostic, and predictive views.")

try:
    sessions, orders, order_items, products, refunds, website_pageviews = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(
        "Couldn't load the data files. Descriptive/Diagnostic tabs need the parquet files "
        "in the 'data/' folder next to app.py (sessions, orders, order_items, products, "
        "refunds, website_pageviews).\n\n"
        f"Error: {e}"
    )

tab_desc, tab_diag, tab_pred = st.tabs(["📊 Descriptive Analysis", "🔍 Diagnostic Analysis", "🎯 Predictor"])

# ============================================================
# DESCRIPTIVE ANALYSIS
# ============================================================
if data_loaded:
    with tab_desc:
        d1, d2, d3, d4, d5, d6 = st.tabs(
            ["Sales & Revenue", "Orders", "Products", "Traffic & Channel", "Customers", "Refunds"]
        )

        # ---------------- 1.1 Sales & Revenue ----------------
        with d1:
            st.header("Sales & Revenue Analysis")

            q("What is our overall revenue and profitability position?")
            total_revenue = order_items['price_usd'].sum()
            total_refund = refunds['refund_amount_usd'].sum()
            net_revenue = total_revenue - total_refund
            total_cogs = order_items['cogs_usd'].sum()
            gross_margin = total_revenue - total_cogs
            gross_margin_perc = gross_margin / total_revenue * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Revenue", f"${total_revenue:,.0f}")
            c2.metric("Net Revenue", f"${net_revenue:,.0f}")
            c3.metric("Total Refunds", f"${total_refund:,.0f}")
            c4.metric("Gross Margin", f"{gross_margin_perc:.1f}%")
            conclude([f"Net revenue stands at ${net_revenue:,.2f} after ${total_refund:,.2f} in refunds, "
                      f"with a gross margin of {gross_margin_perc:.1f}%."])

            st.divider()
            q("How has revenue trended month over month?")
            order_items['months'] = order_items['created_at'].dt.to_period('M').astype(str)
            monthly_revenue = order_items.groupby('months')['price_usd'].sum().reset_index().sort_values('months')
            monthly_revenue['sales_growth'] = monthly_revenue['price_usd'].pct_change()
            st.line_chart(monthly_revenue.set_index('months')['price_usd'])
            st.dataframe(monthly_revenue, use_container_width=True, hide_index=True)
            conclude([f"Revenue moved from ${monthly_revenue['price_usd'].iloc[0]:,.0f} to "
                      f"${monthly_revenue['price_usd'].iloc[-1]:,.0f} over the period — "
                      f"{'net growth' if monthly_revenue['price_usd'].iloc[-1] > monthly_revenue['price_usd'].iloc[0] else 'net decline'} overall."])

            st.divider()
            q("Which quarter generates the most revenue?")
            orders['quarter'] = orders['created_at'].dt.to_period('Q').astype(str)
            revenue_by_quarter = orders.groupby('quarter')['price_usd'].sum().reset_index().sort_values('quarter')
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(revenue_by_quarter['quarter'], revenue_by_quarter['price_usd'], color='red')
            ax.set_title('Revenue by Quarter'); ax.set_xlabel('Quarter'); ax.set_ylabel('Revenue (USD)')
            plt.xticks(rotation=90); fig.tight_layout()
            show(fig)
            top_quarter = revenue_by_quarter.loc[revenue_by_quarter['price_usd'].idxmax()]
            conclude([f"{top_quarter['quarter']} is the strongest quarter (${top_quarter['price_usd']:,.2f})."])

            st.divider()
            q("What's the typical price point and cost spread of items sold?")
            price_dist = order_items['price_usd'].describe()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(order_items['price_usd'], bins=20, color='pink', edgecolor='black')
            ax.set_title('Price Distribution'); ax.set_xlabel('Price'); ax.set_ylabel('Number of order items')
            fig.tight_layout()
            show(fig)
            conclude([f"Typical item price sits around ${price_dist['50%']:.2f} (median), "
                      f"ranging from ${price_dist['min']:.2f} to ${price_dist['max']:.2f}."])

            st.divider()
            q("Is revenue consistent across years and quarters, or seasonal?")
            orders['years'] = orders['created_at'].dt.year
            orders['month_num'] = orders['created_at'].dt.month
            heatmap_data = orders.pivot_table(index='years', columns='month_num', values='price_usd', aggfunc='sum')
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.heatmap(heatmap_data, cmap='Blues', ax=ax)
            ax.set_title('Revenue Heatmap (Year x Month)')
            show(fig)
            conclude(["The heatmap shows whether growth is broad-based across the year or concentrated in specific months."])

            st.divider()
            q("Which months had the sharpest revenue swings month-over-month?")
            monthly_revenue['pct_change'] = monthly_revenue['price_usd'].pct_change().round(2)
            swing_threshold = monthly_revenue['pct_change'].std()
            biggest_jump = monthly_revenue.loc[monthly_revenue['pct_change'].idxmax()]
            biggest_drop = monthly_revenue.loc[monthly_revenue['pct_change'].idxmin()]
            st.dataframe(monthly_revenue[['months', 'price_usd', 'pct_change']], use_container_width=True, hide_index=True)
            conclude([f"Sharpest jump was {biggest_jump['months']} ({biggest_jump['pct_change']:+.2f}%), "
                      f"sharpest drop was {biggest_drop['months']} ({biggest_drop['pct_change']:+.2f}%). "
                      f"A swing bigger than {swing_threshold:.2f} is outside the usual month-to-month range."])

        # ---------------- 1.2 Orders ----------------
        with d2:
            st.header("Order Analysis")

            q("How many orders are we getting, and what's the AOV and conversion rate?")
            total_orders = orders['order_id'].nunique()
            total_session = sessions['website_session_id'].nunique()
            conversion_rate = total_orders / total_session * 100
            avg_value_order = orders['price_usd'].sum() / total_orders
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Orders", f"{total_orders:,}")
            c2.metric("Conversion Rate", f"{conversion_rate:.2f}%")
            c3.metric("Avg Order Value", f"${avg_value_order:.2f}")
            conclude([f"{total_orders:,} orders from {total_session:,} sessions "
                      f"({conversion_rate:.2f}% conversion), averaging ${avg_value_order:.2f} per order."])

            st.divider()
            q("Does order count trend track with revenue?")
            order_trend = orders.groupby(orders['created_at'].dt.to_period('M')).agg(
                order_count=('order_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index()
            order_trend['created_at'] = order_trend['created_at'].astype(str)
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(order_trend['created_at'], order_trend['order_count'], color='steelblue', marker='o')
            ax1.set_ylabel('Order Count', color='steelblue'); ax1.tick_params(axis='y', labelcolor='steelblue')
            plt.xticks(rotation=45)
            ax2 = ax1.twinx()
            ax2.plot(order_trend['created_at'], order_trend['revenue'], color='darkorange', marker='o')
            ax2.set_ylabel('Revenue (USD)', color='darkorange'); ax2.tick_params(axis='y', labelcolor='darkorange')
            ax1.set_title('Monthly Order Count vs Revenue Trend'); fig.tight_layout()
            show(fig)
            conclude(["Order count and revenue trends are plotted together for a visual read on whether they move in lockstep."])

            st.divider()
            q("Is average order value (AOV) rising or declining over time?")
            aov_trend = orders.groupby(orders['created_at'].dt.to_period('M')).agg(
                order_count=('order_id', 'nunique'), sales=('price_usd', 'sum')).reset_index()
            aov_trend['aov'] = aov_trend['sales'] / aov_trend['order_count']
            aov_trend['months'] = aov_trend['created_at'].astype(str)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=aov_trend, x='months', y='aov', color='teal', ax=ax)
            plt.xticks(rotation=90); ax.set_title('Monthly AOV Trend'); fig.tight_layout()
            show(fig)
            conclude([f"AOV moved from ${aov_trend['aov'].iloc[0]:.2f} to ${aov_trend['aov'].iloc[-1]:.2f} over the period."])

            st.divider()
            q("How many items do customers typically buy per order?")
            order_size_distribution = orders['items_purchased'].value_counts().sort_index().reset_index()
            order_size_distribution.columns = ['items_purchased', 'num_orders']
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(order_size_distribution['items_purchased'].astype(str), order_size_distribution['num_orders'], color='mediumseagreen')
            ax.set_title('Order Size Distribution'); ax.set_xlabel('Items per order'); ax.set_ylabel('Number of orders')
            fig.tight_layout()
            show(fig)
            most_common_size = order_size_distribution.loc[order_size_distribution['num_orders'].idxmax()]
            conclude([f"Most orders contain {most_common_size['items_purchased']} item(s) ({most_common_size['num_orders']} orders)."])

            st.divider()
            q("Are certain days of the week consistently stronger for order volume?")
            orders['weekdays'] = orders['created_at'].dt.day_name()
            volume_distribution = orders.groupby('weekdays')['price_usd'].sum().reset_index(name='days_sales')
            volume_distribution = volume_distribution.sort_values(by='days_sales', ascending=False)
            volume_distribution['days_contribution'] = volume_distribution['days_sales'] / volume_distribution['days_sales'].sum() * 100
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(volume_distribution['days_contribution'], labels=volume_distribution['weekdays'], autopct='%1.1f%%')
            ax.set_title('Sales Contribution by Weekday')
            show(fig)
            top_day = volume_distribution.iloc[0]
            conclude([f"{top_day['weekdays']} contributes the most sales ({top_day['days_contribution']:.1f}% of total)."])

        # ---------------- 1.3 Products ----------------
        with d3:
            st.header("Product Analysis")
            q("Which products drive the most revenue?")
            merged3 = order_items.merge(products, on='product_id')
            top_products = merged3.groupby('product_name').agg(
                units_sold=('order_item_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index().sort_values('revenue', ascending=False)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(top_products["product_name"], top_products["revenue"], color="darkorange")
            ax.set_title("Top Products by Revenue"); ax.set_xlabel("Product"); ax.set_ylabel("Revenue (USD)")
            plt.xticks(rotation=45); fig.tight_layout()
            show(fig)
            st.dataframe(top_products, use_container_width=True, hide_index=True)
            best_product = top_products.iloc[0]
            conclude([f"{best_product['product_name']} is the top seller (${best_product['revenue']:,.2f} from {best_product['units_sold']} units)."])

        # ---------------- 1.4 Traffic & Channel ----------------
        with d4:
            st.header("Traffic & Channel Analysis")

            q("Does traffic track with revenue month to month?")
            sessions['months'] = sessions['created_at'].dt.to_period('M').astype(str)
            monthly_sessions = sessions.groupby('months').size().reset_index(name='sessions').sort_values('months')
            st.line_chart(monthly_sessions.set_index('months')['sessions'])
            busy_month = monthly_sessions.loc[monthly_sessions['sessions'].idxmax()]
            quiet_month = monthly_sessions.loc[monthly_sessions['sessions'].idxmin()]
            conclude([f"{busy_month['months']} had the most traffic ({busy_month['sessions']} sessions), "
                      f"{quiet_month['months']} had the least ({quiet_month['sessions']} sessions)."])

            st.divider()
            q("Which acquisition channel (utm_source) brings in the most revenue?")
            merged = sessions[['website_session_id', 'utm_source']].merge(orders[['order_id', 'website_session_id']], on='website_session_id')
            merged = merged.merge(order_items[['order_id', 'price_usd']], on='order_id')
            revenue_by_source = merged.groupby('utm_source').agg(
                orders=('order_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index().sort_values('revenue', ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(revenue_by_source['utm_source'], revenue_by_source['revenue'], color='orange')
            ax.set_title('Revenue by Source'); fig.tight_layout()
            show(fig)
            top_source = revenue_by_source.iloc[0]
            conclude([f"{top_source['utm_source']} brings in the most revenue (${top_source['revenue']:,.2f} from {top_source['orders']} orders)."])

            st.divider()
            q("Which device type generates the most revenue?")
            merged2 = sessions[['website_session_id', 'device_type']].merge(orders[['order_id', 'website_session_id']], on='website_session_id')
            merged2 = merged2.merge(order_items[['order_id', 'price_usd']], on='order_id')
            revenue_by_device = merged2.groupby('device_type').agg(
                orders=('order_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index().sort_values('revenue', ascending=False)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(revenue_by_device['device_type'], revenue_by_device['revenue'], color='pink')
            ax.set_title('Revenue by Device'); fig.tight_layout()
            show(fig)
            top_device = revenue_by_device.iloc[0]
            conclude([f"{top_device['device_type']} generates the most revenue (${top_device['revenue']:,.2f})."])

            st.divider()
            q("Where does our traffic actually come from — channel and device mix?")
            source_mix = sessions['utm_source'].value_counts(normalize=True).round(2).reset_index()
            source_mix.columns = ['utm_source', 'pct_of_sessions']
            device_mix = sessions['device_type'].value_counts(normalize=True).round(2).reset_index()
            device_mix.columns = ['device_type', 'pct_of_sessions']
            colA, colB = st.columns(2)
            with colA:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(source_mix['utm_source'], source_mix['pct_of_sessions'], color='darkgreen')
                ax.set_title('Session Mix by Source'); fig.tight_layout()
                show(fig)
            with colB:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(device_mix['device_type'], device_mix['pct_of_sessions'], color='lightgreen')
                ax.set_title('Session Mix by Device'); fig.tight_layout()
                show(fig)
            top_traffic_source = source_mix.iloc[0]
            top_traffic_device = device_mix.iloc[0]
            conclude([f"{top_traffic_source['utm_source']} sends the most traffic ({top_traffic_source['pct_of_sessions'] * 100:.0f}% of sessions); "
                      f"{top_traffic_device['device_type']} sends the most traffic ({top_traffic_device['pct_of_sessions'] * 100:.0f}% of sessions)."])

        # ---------------- 1.5 Customers ----------------
        with d5:
            st.header("Customer / Visitor Analysis")
            q("What share of our sessions are new vs. repeat visitors?")
            session_mix = sessions['is_repeat_session'].value_counts().reset_index()
            session_mix.columns = ['is_repeat_session', 'num_sessions']
            session_mix['label'] = session_mix['is_repeat_session'].map({0: "new", 1: "repeat"})
            session_mix['pct_of_sessions'] = (session_mix['num_sessions'] / session_mix['num_sessions'].sum() * 100).round(2)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(session_mix['num_sessions'], labels=session_mix['label'], autopct='%1.1f%%', colors=['pink', 'purple'])
            ax.set_title('New vs Repeat Visitor Mix')
            show(fig)
            repeat_pct = session_mix.loc[session_mix['label'] == 'repeat', 'pct_of_sessions'].values[0]
            conclude([f"Repeat visitors make up {repeat_pct}% of all sessions."])

        # ---------------- 1.6 Refunds ----------------
        with d6:
            st.header("Refund Analysis")
            q("How have refunds trended over time — any spikes to flag?")
            refunds['month'] = refunds['created_at'].dt.to_period('M').astype(str)
            monthly_refunds = refunds.groupby('month')['refund_amount_usd'].sum().reset_index().sort_values('month')
            st.line_chart(monthly_refunds.set_index('month')['refund_amount_usd'])
            worst_refund_month = monthly_refunds.loc[monthly_refunds['refund_amount_usd'].idxmax()]
            conclude([f"{worst_refund_month['month']} had the highest refund total (${worst_refund_month['refund_amount_usd']:.2f})."])

        st.divider()
        st.subheader("Section Summary — Descriptive Analysis")
        conclude([
            f"Net revenue: ${net_revenue:,.2f} at a {gross_margin_perc:.1f}% gross margin, with {total_orders:,} orders "
            f"from {total_session:,} sessions ({conversion_rate:.2f}% conversion).",
            f"Strongest quarter: {top_quarter['quarter']}. Best-selling product: {best_product['product_name']}.",
            f"Top traffic/revenue channel: {top_source['utm_source']}. Top device: {top_device['device_type']}.",
            f"Repeat visitors are {repeat_pct}% of sessions. Highest refund month: {worst_refund_month['month']}.",
        ])

    # ============================================================
    # DIAGNOSTIC ANALYSIS
    # ============================================================
    with tab_diag:
        g1, g2, g3, g4, g5, g6 = st.tabs(
            ["Sales & Revenue", "Orders", "Products", "Traffic & Channel", "Customers", "Refunds"]
        )
        merged_grp = sessions.merge(orders[['order_id', 'website_session_id']], on='website_session_id', how='left')

        # ---------------- 2.1 Sales & Revenue Diagnostics ----------------
        with g1:
            st.header("Sales & Revenue Diagnostics")

            q("Does session traffic explain the month-to-month revenue trend, or is revenue driven by something else?")
            rev_vs_sessions = monthly_revenue[['months', 'price_usd']].merge(monthly_sessions, on='months', how='inner')
            traffic_revenue_corr = rev_vs_sessions['price_usd'].corr(rev_vs_sessions['sessions'])
            st.dataframe(rev_vs_sessions, use_container_width=True, hide_index=True)
            conclude([
                f"Sessions and revenue have a {traffic_revenue_corr:.2f} correlation month-to-month.",
                "A strong positive value (close to 1) means revenue is mostly a volume story — more traffic, more revenue.",
                "A weaker value means revenue swings are being driven by something else too, like AOV or conversion rate.",
            ])

            st.divider()
            q("Is revenue growth coming from more orders, or from customers spending more per order (AOV)?")
            order_growth_pct = (order_trend['order_count'].iloc[-1] - order_trend['order_count'].iloc[0]) / order_trend['order_count'].iloc[0] * 100
            aov_growth_pct = (aov_trend['aov'].iloc[-1] - aov_trend['aov'].iloc[0]) / aov_trend['aov'].iloc[0] * 100
            c1, c2 = st.columns(2)
            c1.metric("Order Count Growth", f"{order_growth_pct:+.2f}%")
            c2.metric("AOV Growth", f"{aov_growth_pct:+.2f}%")
            conclude([
                f"Order count changed {order_growth_pct:+.2f}% over the period.",
                f"AOV changed {aov_growth_pct:+.2f}% over the period.",
                f"{'Order volume is the bigger driver of revenue change' if abs(order_growth_pct) > abs(aov_growth_pct) else 'AOV is the bigger driver of revenue change'} — "
                f"{'focus on acquisition/conversion to keep growing' if abs(order_growth_pct) > abs(aov_growth_pct) else 'focus on upsell/bundling to keep growing'}.",
            ])

        # ---------------- 2.2 Order Diagnostics ----------------
        with g2:
            st.header("Order Diagnostics")
            q("Are order size (items purchased), price, and cost related to one another?")
            numeric_cols = ['items_purchased', 'price_usd', 'cogs_usd']
            correlation = orders[numeric_cols].corr().round(3)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label='correlation')
            ax.set_xticks(range(len(numeric_cols))); ax.set_xticklabels(numeric_cols, rotation=45)
            ax.set_yticks(range(len(numeric_cols))); ax.set_yticklabels(numeric_cols)
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax.text(j, i, correlation.iloc[i, j], ha='center', va='center', color='black')
            ax.set_title('Correlation Heatmap (Order-level)'); fig.tight_layout()
            show(fig)
            strongest_pair = correlation.where(~np.eye(len(numeric_cols), dtype=bool)).abs().stack().idxmax()
            conclude([
                f"{strongest_pair[0]} and {strongest_pair[1]} are the most correlated pair ({correlation.loc[strongest_pair]:.2f}).",
                "The other variable pairs move fairly independently of each other.",
                "This tells us whether bigger orders are simply higher-priced items, or a genuinely different customer behavior.",
            ])

        # ---------------- 2.3 Product Diagnostics ----------------
        with g3:
            st.header("Product Diagnostics")
            q("Which products carry the highest refund risk?")
            items_with_product = order_items.merge(products, on='product_id')
            items_with_refund = items_with_product.merge(refunds[['order_item_id', 'refund_amount_usd']], on='order_item_id', how='left')
            items_with_refund['was_refunded'] = items_with_refund['refund_amount_usd'].notna()
            refund_rate_by_product = items_with_refund.groupby('product_name').agg(
                units_sold=('order_item_id', 'nunique'), units_refunded=('was_refunded', 'sum'),
                refund_amount=('refund_amount_usd', 'sum')).reset_index()
            refund_rate_by_product['refund_rate_pct'] = (refund_rate_by_product['units_refunded'] / refund_rate_by_product['units_sold'] * 100).round(2)
            refund_rate_by_product = refund_rate_by_product.sort_values('refund_rate_pct', ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(refund_rate_by_product['product_name'], refund_rate_by_product['refund_rate_pct'], color='blue')
            ax.set_title('Refund Rate by Product'); ax.set_ylabel('Refund rate (%)')
            plt.xticks(rotation=45); fig.tight_layout()
            show(fig)
            worst_refund = refund_rate_by_product.iloc[0]
            best_refund = refund_rate_by_product.iloc[-1]
            conclude([
                f"{worst_refund['product_name']} has the highest refund rate ({worst_refund['refund_rate_pct']}% of units sold).",
                f"{best_refund['product_name']} has the lowest refund rate ({best_refund['refund_rate_pct']}%).",
                "Products with high revenue AND high refund rate deserve priority review.",
            ])

        # ---------------- 2.4 Traffic & Channel Diagnostics ----------------
        with g4:
            st.header("Traffic & Channel Diagnostics")

            q("Why do some acquisition channels convert better than others?")
            conversion_by_source = merged_grp.groupby('utm_source').agg(
                sessions=('website_session_id', 'nunique'), orders=('order_id', 'nunique')).reset_index()
            conversion_by_source['conversion_rate_pct'] = (conversion_by_source['orders'] / conversion_by_source['sessions'] * 100).round(2)
            conversion_by_source = conversion_by_source.sort_values('conversion_rate_pct', ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(conversion_by_source['utm_source'], conversion_by_source['conversion_rate_pct'], color='steelblue')
            ax.set_title('Conversion Rate by Source'); ax.set_ylabel('Conversion rate (%)'); fig.tight_layout()
            show(fig)
            best_conv_source = conversion_by_source.iloc[0]
            worst_conv_source = conversion_by_source.iloc[-1]
            conclude([
                f"{best_conv_source['utm_source']} converts best ({best_conv_source['conversion_rate_pct']}%).",
                f"{worst_conv_source['utm_source']} converts worst ({worst_conv_source['conversion_rate_pct']}%).",
                "Traffic quality/intent differs meaningfully by channel — weight budget toward higher-converting sources.",
            ])

            st.divider()
            q("Why does one device type convert better than another?")
            conversion_by_device = merged_grp.groupby('device_type').agg(
                sessions=('website_session_id', 'nunique'), orders=('order_id', 'nunique')).reset_index()
            conversion_by_device['conversion_rate_pct'] = (conversion_by_device['orders'] / conversion_by_device['sessions'] * 100).round(2)
            conversion_by_device = conversion_by_device.sort_values('conversion_rate_pct', ascending=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(conversion_by_device['device_type'], conversion_by_device['conversion_rate_pct'], color='steelblue')
            ax.set_title('Conversion Rate by Device'); ax.set_ylabel('Conversion rate (%)'); fig.tight_layout()
            show(fig)
            best_conv_device = conversion_by_device.iloc[0]
            worst_conv_device = conversion_by_device.iloc[-1]
            conclude([
                f"{best_conv_device['device_type']} converts best ({best_conv_device['conversion_rate_pct']}%).",
                f"{worst_conv_device['device_type']} converts worst ({worst_conv_device['conversion_rate_pct']}%).",
                "Points to a UX or checkout friction gap between devices.",
            ])

            st.divider()
            q("Where in the user journey (funnel) are we losing the most sessions?")
            total_session_f = sessions['website_session_id'].nunique()
            sessions_with_pageview = website_pageviews['website_session_id'].nunique()
            sessions_with_order = orders['website_session_id'].nunique()
            funnel = pd.DataFrame({
                'stage': ["Landed", "Viewed a page", "Placed an order"],
                'sessions': [total_session_f, sessions_with_pageview, sessions_with_order],
            })
            funnel['pct_of_landed'] = (funnel['sessions'] / total_session_f * 100).round(2)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(funnel['stage'], funnel['sessions'], color='teal')
            ax.set_title('Funnel Drop-off'); fig.tight_layout()
            show(fig)
            drop_pageview = 100 - funnel.loc[1, 'pct_of_landed']
            drop_order = funnel.loc[1, 'pct_of_landed'] - funnel.loc[2, 'pct_of_landed']
            biggest_drop_stage = 'landed -> viewed a page' if drop_pageview > drop_order else 'viewed a page -> placed an order'
            conclude([
                f"Biggest drop-off happens between '{biggest_drop_stage}'.",
                f"Landed -> Viewed a page loses {drop_pageview:.1f}% of sessions.",
                f"Viewed a page -> Placed an order loses {drop_order:.1f}% of sessions.",
            ])

        # ---------------- 2.5 Customer Diagnostics ----------------
        with g5:
            st.header("Customer Diagnostics")
            q("Why do repeat visitors behave differently from new visitors?")
            repeat_vs_new = merged_grp.groupby('is_repeat_session').agg(
                sessions=('website_session_id', 'nunique'), orders=('order_id', 'nunique')).reset_index()
            repeat_vs_new['conversion_rate_pct'] = (repeat_vs_new['orders'] / repeat_vs_new['sessions'] * 100).round(2)
            repeat_vs_new = repeat_vs_new.sort_values('conversion_rate_pct', ascending=False)
            labels = ['repeat' if v == 1 else 'new' for v in repeat_vs_new['is_repeat_session']]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.bar(labels, repeat_vs_new['conversion_rate_pct'], color=['purple', 'grey'])
            ax.set_title('Repeat vs New Visitor Conversion'); ax.set_ylabel('Conversion rate (%)')
            fig.tight_layout()
            show(fig)
            repeat_row = repeat_vs_new[repeat_vs_new['is_repeat_session'] == 1].iloc[0]
            new_row = repeat_vs_new[repeat_vs_new['is_repeat_session'] == 0].iloc[0]
            gap = repeat_row['conversion_rate_pct'] - new_row['conversion_rate_pct']
            conclude([
                f"Repeat visitors convert at {repeat_row['conversion_rate_pct']}% vs {new_row['conversion_rate_pct']}% for new visitors.",
                f"That's a {gap:.2f} percentage-point gap.",
                f"{'Familiarity/trust is clearly driving conversion' if gap > 0 else 'Repeat visits are not translating into a conversion advantage yet'}.",
            ])

        # ---------------- 2.6 Refund Diagnostics ----------------
        with g6:
            st.header("Refund Diagnostics")
            q("Is the refund trend proportional to order volume, or spiking independently?")
            refund_vs_orders = monthly_refunds.merge(order_trend.rename(columns={'created_at': 'month'}), on='month', how='inner')
            refund_vs_orders['refund_rate_pct'] = (refund_vs_orders['refund_amount_usd'] / refund_vs_orders['revenue'] * 100).round(2)
            st.line_chart(refund_vs_orders.set_index('month')['refund_rate_pct'])
            worst_refund_rate_month = refund_vs_orders.loc[refund_vs_orders['refund_rate_pct'].idxmax()]
            avg_refund_rate = refund_vs_orders['refund_rate_pct'].mean()
            conclude([
                f"Average monthly refund rate is {avg_refund_rate:.2f}% of revenue.",
                f"{worst_refund_rate_month['month']} spiked to {worst_refund_rate_month['refund_rate_pct']}% — "
                f"{'well above the average, an anomaly worth investigating' if worst_refund_rate_month['refund_rate_pct'] > avg_refund_rate * 1.5 else 'roughly in line with the normal range'}.",
            ])

        st.divider()
        st.subheader("Section Summary — Diagnostic Analysis")
        conclude([
            f"Revenue is {'largely a traffic/volume story' if traffic_revenue_corr > 0.7 else 'not fully explained by traffic alone'} "
            f"(sessions-vs-revenue correlation: {traffic_revenue_corr:.2f}).",
            f"{best_conv_source['utm_source']} is both a top revenue channel and the best converter — {worst_conv_source['utm_source']} converts worst.",
            f"{best_conv_device['device_type']} converts better than {worst_conv_device['device_type']} — a device-specific UX gap.",
            f"Biggest funnel drop-off: '{biggest_drop_stage}'.",
            f"{worst_refund['product_name']} carries the highest refund risk.",
        ])

else:
    with tab_desc:
        st.info("Descriptive charts need a live database connection (see error above).")
    with tab_diag:
        st.info("Diagnostic charts need a live database connection (see error above).")

# ============================================================
# PREDICTOR
# ============================================================
with tab_pred:
    st.header("🎯 Website Session Conversion Predictor")
    st.caption(
        "Predicts the probability that a website session converts into an order, "
        "based on the logistic regression model trained on the e-commerce analytics capstone data."
    )

    import os
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"'{MODEL_PATH}' not found. Put conversion_model.pkl in the same folder as app.py "
            "(it's created at the end of the notebook's Section 3)."
        )
    else:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        encoder = bundle["encoder"]
        categorical_features = bundle["categorical_features"]
        binary_features = bundle["binary_features"]
        cat_options = dict(zip(categorical_features, encoder.categories_))

        st.subheader("Session details")
        col1, col2 = st.columns(2)
        with col1:
            utm_source = st.selectbox("UTM Source", list(cat_options["utm_source"]))
            utm_campaign = st.selectbox("UTM Campaign", list(cat_options["utm_campaign"]))
        with col2:
            device_type = st.selectbox("Device Type", list(cat_options["device_type"]))
            is_repeat_session = st.selectbox("Repeat Session?", ["No", "Yes"])

        if st.button("Predict Conversion", type="primary"):
            input_df = pd.DataFrame([{
                "utm_source": utm_source, "utm_campaign": utm_campaign, "device_type": device_type,
            }])
            X_cat = encoder.transform(input_df[categorical_features]).toarray()
            X_bin = np.array([[1 if is_repeat_session == "Yes" else 0]])
            X = np.hstack([X_cat, X_bin])

            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][1]

            st.divider()
            st.metric("Conversion Probability", f"{proba * 100:.2f}%")
            if pred == 1:
                st.success("Predicted: Session is LIKELY to convert")
            else:
                st.warning("Predicted: Session is UNLIKELY to convert")

        st.divider()
        st.caption("Model: Logistic Regression (class_weight='balanced') · Features: utm_source, utm_campaign, device_type (OneHotEncoder) + is_repeat_session")