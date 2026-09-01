#!/usr/bin/env python
# coding: utf-8

# # E-commerce Analytics Pipeline
# 
# ```
# E-commerce Analytics Pipeline
# ==============================
# A digital analytics case study for an e-commerce toy startup, structured the
# way it would be presented internally:
# 
#   SECTION 0 - Setup, data load, cleaning
#   SECTION 1 - DESCRIPTIVE ANALYSIS   (what happened)
#   SECTION 2 - DIAGNOSTIC ANALYSIS    (why it happened)   -- mirrors Section 1's
#                                                               headings 1:1
#   SECTION 3 - PREDICTIVE ANALYSIS    (what's likely to happen) 
# 
# 

# In[4]:


import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns





# ##  DATA LOAD & CLEANING

# In[6]:


conn = pyodbc.connect(
    'Driver={SQL Server};'
    'Server=KOMAL\\SQLEXPRESS;'
    'Database=Ecommerce_Analytics_project;'
    'Trusted_connection=yes;'
)
cursor = conn.cursor()

sessions = pd.read_sql("SELECT * FROM website_sessions", conn)
orders = pd.read_sql("SELECT * FROM orders", conn)
order_items = pd.read_sql("SELECT * FROM orders_items", conn)
products = pd.read_sql("SELECT * FROM products", conn)
refunds = pd.read_sql("SELECT * FROM order_item_refunds", conn)
website_pageviews = pd.read_sql("SELECT * FROM website_pageviews", conn)

print(sessions.head())
print(orders.head())
print(order_items.head())
print(products.head())
print(refunds.head())
print(website_pageviews.head())



# In[7]:


# --- Outlier handling (IQR capping, flagged not deleted) ---
for col in ['price_usd', 'cogs_usd', 'items_purchased']:
    q1, q3 = orders[col].quantile(0.25), orders[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
    flag_col = col + "_was_outlier"
    orders[flag_col] = (orders[col] < lower) | (orders[col] > upper)
    n_flagged = orders[flag_col].sum()
    orders[col] = orders[col].clip(lower=lower, upper=upper)
    print(f'orders.{col}: bounds=({lower:.2f},{upper:.2f}), {n_flagged} rows capped')

for col in ['price_usd', 'cogs_usd']:
    q1, q3 = order_items[col].quantile(0.25), order_items[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
    flag_col = col + "_was_outlier"
    order_items[flag_col] = (order_items[col] < lower) | (order_items[col] > upper)
    n_flagged = order_items[flag_col].sum()
    order_items[col] = order_items[col].clip(lower=lower, upper=upper)
    print(f'order_items.{col}: bounds=({lower:.2f},{upper:.2f}), {n_flagged} rows capped')

col = 'refund_amount_usd'
q1, q3 = refunds[col].quantile(0.25), refunds[col].quantile(0.75)
iqr = q3 - q1
lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
refunds[col + "_was_outlier"] = (refunds[col] < lower) | (refunds[col] > upper)
n_flagged = refunds[col + "_was_outlier"].sum()
refunds[col] = refunds[col].clip(lower=lower, upper=upper)
print(f'refunds.{col}: bounds=({lower:.2f},{upper:.2f}), {n_flagged} rows capped')


# ============================================================
# SECTION 1 - DESCRIPTIVE ANALYSIS (what happened)
# ============================================================


# ## SECTION 1: DESCRIPTIVE ANALYSIS

# ### 1.1 Sales & Revenue Analysis

# In[64]:


'''Q: What is our overall revenue and profitability position?'''
total_revenue = order_items['price_usd'].sum()
total_refund = refunds['refund_amount_usd'].sum()
net_revenue = total_revenue - total_refund
total_cogs = order_items['cogs_usd'].sum()
gross_margin = total_revenue - total_cogs
gross_margin_perc = gross_margin / total_revenue * 100

print('total_revenue:', round(total_revenue, 2))
print('total_refund:', round(total_refund, 2))
print('net_revenue:', round(net_revenue, 2))
print('total_cogs:', round(total_cogs, 2))
print('gross_margin:', round(gross_margin, 2))
print('gross_margin_%:', round(gross_margin_perc, 2))
print(f"insights: Net revenue stands at ${net_revenue:,.2f} after ${total_refund:,.2f} in refunds, "
      f"with a gross margin of {gross_margin_perc:.1f}%.")


# In[63]:


'''Q: How has revenue trended month over month?'''
order_items['months'] = order_items['created_at'].dt.to_period('M').astype(str)
monthly_revenue = order_items.groupby('months')['price_usd'].sum().reset_index().sort_values('months')
monthly_revenue['sales_growth'] = monthly_revenue['price_usd'].pct_change()
print(monthly_revenue.to_string(index=False))
print(f"insights: Revenue moved from ${monthly_revenue['price_usd'].iloc[0]:,.0f} to "
      f"${monthly_revenue['price_usd'].iloc[-1]:,.0f} over the period — "
      f"{'net growth' if monthly_revenue['price_usd'].iloc[-1] > monthly_revenue['price_usd'].iloc[0] else 'net decline'} overall.")


# In[62]:


'''Q: Which quarter generates the most revenue?'''
orders['quarter'] = orders['created_at'].dt.to_period('Q').astype(str)
revenue_by_quarter = orders.groupby('quarter')['price_usd'].sum().reset_index().sort_values('quarter')
print(revenue_by_quarter.to_string(index=False))
plt.figure(figsize=(7, 4))
plt.bar(revenue_by_quarter['quarter'], revenue_by_quarter['price_usd'], color='red')
plt.title('Revenue by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Revenue (USD)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
top_quarter = revenue_by_quarter.loc[revenue_by_quarter['price_usd'].idxmax()]
print(f"insights: {top_quarter['quarter']} is the strongest quarter (${top_quarter['price_usd']:,.2f}).")


# In[61]:


'''Q: What's the typical price point and cost spread of items sold?'''
price_dist = order_items['price_usd'].describe()
cogs_dist = order_items['cogs_usd'].describe()
print('\n-- price distribution --')
print(price_dist.to_string())
print('\n-- cost distribution --')
print(cogs_dist.to_string())
plt.figure(figsize=(8, 5))
plt.hist(order_items['price_usd'], bins=20, color='pink', edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Number of order items')
plt.tight_layout()
plt.show()
print(f"insights: Typical item price sits around ${price_dist['50%']:.2f} (median), "
      f"ranging from ${price_dist['min']:.2f} to ${price_dist['max']:.2f}.")


# In[60]:


'''Q: Is revenue consistent across years and quarters, or seasonal?'''
orders['years'] = orders['created_at'].dt.year
orders['month_num'] = orders['created_at'].dt.month
heatmap_data = orders.pivot_table(index='years', columns='month_num', values='price_usd', aggfunc='sum')
sns.heatmap(heatmap_data, cmap='Blues')
plt.title('Revenue Heatmap (Year x Month)')
plt.show()
quarter_data = orders.groupby(['years', orders['created_at'].dt.quarter])['price_usd'].sum().reset_index()
quarter_data.columns = ['years', 'quarter_num', 'price_usd']
quarter_pivot = quarter_data.pivot(index='years', columns='quarter_num', values='price_usd')
quarter_pivot.plot(kind='bar', stacked=True)
plt.title('Revenue by Year, Stacked by Quarter')
plt.tight_layout()
plt.show()
print("insights: The heatmap/stacked chart shows whether growth is broad-based across the year "
      "or concentrated in specific months.")


# In[59]:


'''Q: Which months had the sharpest revenue swings month-over-month?'''
monthly_revenue['pct_change'] = monthly_revenue['price_usd'].pct_change().round(2)
swing_threshold = monthly_revenue['pct_change'].std()
biggest_jump = monthly_revenue.loc[monthly_revenue['pct_change'].idxmax()]
biggest_drop = monthly_revenue.loc[monthly_revenue['pct_change'].idxmin()]
print(monthly_revenue[['months', 'price_usd', 'pct_change']].to_string(index=False))
print(f"insights: Sharpest jump was {biggest_jump['months']} ({biggest_jump['pct_change']:+.2f}%), "
      f"sharpest drop was {biggest_drop['months']} ({biggest_drop['pct_change']:+.2f}%). "
      f"A swing bigger than {swing_threshold:.2f} is outside the usual month-to-month range.")


# In[ ]:





# ### 1.2 Order Analysis

# In[65]:


'''Q: How many orders are we getting, and what's the average order value and conversion rate?'''
total_orders = orders['order_id'].nunique()
total_session = sessions['website_session_id'].nunique()
conversion_rate = total_orders / total_session * 100
avg_value_order = orders['price_usd'].sum() / total_orders
print('total_orders:', total_orders)
print('total_sessions:', total_session)
print('conversion_rate_%:', round(conversion_rate, 2))
print('avg_order_value:', round(avg_value_order, 2))
print(f"insights: {total_orders:,} orders from {total_session:,} sessions "
      f"({conversion_rate:.2f}% conversion), averaging ${avg_value_order:.2f} per order.")


# In[58]:


'''Q: Does order count trend track with revenue?'''
order_trend = orders.groupby(orders['created_at'].dt.to_period('M')).agg(
    order_count=('order_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index()
order_trend['created_at'] = order_trend['created_at'].astype(str)
print(order_trend.to_string(index=False))
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(order_trend['created_at'], order_trend['order_count'], color='steelblue', marker='o', label='Order Count')
ax1.set_xlabel('Month')
ax1.set_ylabel('Order Count', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
plt.xticks(rotation=45)
ax2 = ax1.twinx()
ax2.plot(order_trend['created_at'], order_trend['revenue'], color='darkorange', marker='o', label='Revenue')
ax2.set_ylabel('Revenue (USD)', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
plt.title('Monthly Order Count vs Revenue Trend')
fig.tight_layout()
plt.show()
print(f"insights: Order count and revenue trends are plotted together above for a visual read "
      f"on whether they move in lockstep.")


# In[57]:


print("\nQ: Is average order value (AOV) rising or declining over time?")
aov_trend = orders.groupby(orders['created_at'].dt.to_period('M')).agg(
    order_count=('order_id', 'nunique'), sales=('price_usd', 'sum')).reset_index()
aov_trend['aov'] = aov_trend['sales'] / aov_trend['order_count']
aov_trend['months'] = aov_trend['created_at'].astype(str)
plt.figure(figsize=(14, 6))
sns.barplot(data=aov_trend, x='months', y='aov', color='teal')
plt.xticks(rotation=90)
plt.title('Monthly AOV Trend', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()
print(f"insights: AOV moved from ${aov_trend['aov'].iloc[0]:.2f} to ${aov_trend['aov'].iloc[-1]:.2f} "
      f"over the period.")


# In[56]:


'''Q: How many items do customers typically buy per order?'''
order_size_distribution = orders['items_purchased'].value_counts().sort_index().reset_index()
order_size_distribution.columns = ['items_purchased', 'num_orders']
print(order_size_distribution.to_string(index=False))
plt.figure(figsize=(4, 4))
plt.bar(order_size_distribution['items_purchased'].astype(str), order_size_distribution['num_orders'], color='mediumseagreen')
plt.title('Order Size Distribution')
plt.xlabel('Items per order')
plt.ylabel('Number of orders')
plt.tight_layout()
plt.show()
most_common_size = order_size_distribution.loc[order_size_distribution['num_orders'].idxmax()]
print(f"insights: Most orders contain {most_common_size['items_purchased']} item(s) "
      f"({most_common_size['num_orders']} orders).")


# In[55]:


'''Q: Are certain days of the week consistently stronger for order volume?'''
orders['weekdays'] = orders['created_at'].dt.day_name()
volume_distribution = orders.groupby('weekdays')['price_usd'].sum().reset_index(name='days_sales')
volume_distribution = volume_distribution.sort_values(by='days_sales', ascending=False)
volume_distribution['days_contribution'] = volume_distribution['days_sales'] / volume_distribution['days_sales'].sum() * 100
print(volume_distribution.to_string(index=False))
plt.figure(figsize=(7, 3))
plt.pie(x=volume_distribution['days_contribution'], labels=volume_distribution['weekdays'], autopct='%1.1f%%')
plt.suptitle('Sales Contribution by Weekday', fontweight='bold')
plt.show()
top_day = volume_distribution.iloc[0]
print(f"insights: {top_day['weekdays']} contributes the most sales ({top_day['days_contribution']:.1f}% of total).")


# ### 1.3 Product Analysis

# In[54]:


'''Q: Which products drive the most revenue?")'''
merged3 = order_items.merge(products, on='product_id')
top_products = merged3.groupby('product_name').agg(
    units_sold=('order_item_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index().sort_values('revenue', ascending=False)
print(top_products.to_string(index=False))
plt.figure(figsize=(10, 4))
plt.bar(top_products["product_name"], top_products["revenue"], color="darkorange")
plt.title("Top Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue (USD)")
plt.tight_layout()
plt.show()
best_product = top_products.iloc[0]
print(f"insights: {best_product['product_name']} is the top seller "
      f"(${best_product['revenue']:,.2f} from {best_product['units_sold']} units).")


# ### 1.4 Traffic & Channel Analysis

# In[53]:


'''Q: Does traffic track with revenue month to month, or do they move independently?'''
sessions['months'] = sessions['created_at'].dt.to_period('M').astype(str)
monthly_sessions = sessions.groupby('months').size().reset_index(name='sessions').sort_values('months')
print(monthly_sessions.to_string(index=False))
busy_month = monthly_sessions.loc[monthly_sessions['sessions'].idxmax()]
quiet_month = monthly_sessions.loc[monthly_sessions['sessions'].idxmin()]
print(f"insights: {busy_month['months']} had the most traffic ({busy_month['sessions']} sessions), "
      f"{quiet_month['months']} had the least ({quiet_month['sessions']} sessions).")


# In[52]:


'''Q: Which acquisition channel (utm_source) brings in the most revenue?'''
merged = sessions[['website_session_id', 'utm_source']].merge(orders[['order_id', 'website_session_id']], on='website_session_id')
merged = merged.merge(order_items[['order_id', 'price_usd']], on='order_id')
revenue_by_source = merged.groupby('utm_source').agg(
    orders=('order_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index().sort_values('revenue', ascending=False)
print(revenue_by_source.to_string(index=False))
plt.figure(figsize=(8, 4))
plt.bar(revenue_by_source['utm_source'], revenue_by_source['revenue'], color='orange')
plt.title('Revenue by Source')
plt.xlabel('Source')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()
top_source = revenue_by_source.iloc[0]
print(f"insights: {top_source['utm_source']} brings in the most revenue "
      f"(${top_source['revenue']:,.2f} from {top_source['orders']} orders).")


# In[51]:


'''Q: Which device type generates the most revenue?'''
merged2 = sessions[['website_session_id', 'device_type']].merge(orders[['order_id', 'website_session_id']], on='website_session_id')
merged2 = merged2.merge(order_items[['order_id', 'price_usd']], on='order_id')
revenue_by_device = merged2.groupby('device_type').agg(
    orders=('order_id', 'nunique'), revenue=('price_usd', 'sum')).reset_index().sort_values('revenue', ascending=False)
print(revenue_by_device.to_string(index=False))
plt.figure(figsize=(6, 3))
plt.bar(revenue_by_device['device_type'], revenue_by_device['revenue'], color='pink')
plt.title('Revenue by Device')
plt.xlabel('Device')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()
top_device = revenue_by_device.iloc[0]
print(f"insights: {top_device['device_type']} generates the most revenue (${top_device['revenue']:,.2f}).")


# In[50]:


'''Q: Where does our traffic actually come from — channel and device mix?'''
source_mix = sessions['utm_source'].value_counts(normalize=True).round(2).reset_index()
source_mix.columns = ['utm_source', 'pct_of_sessions']
device_mix = sessions['device_type'].value_counts(normalize=True).round(2).reset_index()
device_mix.columns = ['device_type', 'pct_of_sessions']
print('\n-- session mix by source --')
print(source_mix.to_string(index=False))
print('\n-- session mix by device --')
print(device_mix.to_string(index=False))
plt.figure(figsize=(6, 4))
plt.bar(source_mix['utm_source'], source_mix['pct_of_sessions'], color='darkgreen')
plt.title('Session Mix by Source')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 4))
plt.bar(device_mix['device_type'], device_mix['pct_of_sessions'], color='lightgreen')
plt.title('Session Mix by Device')
plt.tight_layout()
plt.show()
top_traffic_source = source_mix.iloc[0]
top_traffic_device = device_mix.iloc[0]
print(f"insights: {top_traffic_source['utm_source']} sends the most traffic "
      f"({top_traffic_source['pct_of_sessions'] * 100:.0f}% of sessions); "
      f"{top_traffic_device['device_type']} sends the most traffic "
      f"({top_traffic_device['pct_of_sessions'] * 100:.0f}% of sessions).")


# ### 1.5 Customer / Visitor Analysis

# In[49]:


'''Q: What share of our sessions are new vs. repeat visitors?'''
session_mix = sessions['is_repeat_session'].value_counts().reset_index()
session_mix.columns = ['is_repeat_session', 'num_sessions']
session_mix['label'] = session_mix['is_repeat_session'].map({0: "new", 1: "repeat"})
session_mix['pct_of_sessions'] = (session_mix['num_sessions'] / session_mix['num_sessions'].sum() * 100).round(2)
print(session_mix[['label', 'pct_of_sessions', 'num_sessions']].to_string(index=False))
plt.figure(figsize=(5, 3))
plt.pie(session_mix['num_sessions'], labels=session_mix['label'], autopct='%1.1f%%', colors=['pink', 'purple'])
plt.title('New vs Repeat Visitor Mix')
plt.legend()
plt.tight_layout()
plt.show()
repeat_pct = session_mix.loc[session_mix['label'] == 'repeat', 'pct_of_sessions'].values[0]
print(f"insights: Repeat visitors make up {repeat_pct}% of all sessions.")


# ### 1.6 Refund Analysis

# In[48]:


'''Q: How have refunds trended over time — any spikes to flag?'''
refunds['created_at'] = pd.to_datetime(refunds['created_at'])
refunds['month'] = refunds['created_at'].dt.to_period('M').astype(str)
monthly_refunds = refunds.groupby('month')['refund_amount_usd'].sum().reset_index().sort_values('month')
print(monthly_refunds.to_string(index=False))
plt.figure(figsize=(8, 4))
plt.plot(monthly_refunds['month'], monthly_refunds['refund_amount_usd'], marker='o', color='red')
plt.title('Monthly Refund Trend')
plt.xlabel('Month')
plt.ylabel('Refunds ($)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
worst_refund_month = monthly_refunds.loc[monthly_refunds['refund_amount_usd'].idxmax()]
print(f"insights: {worst_refund_month['month']} had the highest refund total "
      f"(${worst_refund_month['refund_amount_usd']:.2f}).")


# ### SECTION 1 SUMMARY — Descriptive Analysis

# In[43]:


def insights(points):
    """Print a bullet-point conclusion. `points` is a list of strings."""
    print("insights:")
    for p in points:
        print(f"  - {p}")


# In[44]:


print("Q: Overall, what does the descriptive data tell us happened in the business?")
insights([
    f"Net revenue: ${net_revenue:,.2f} at a {gross_margin_perc:.1f}% gross margin, "
    f"with {total_orders:,} orders from {total_session:,} sessions ({conversion_rate:.2f}% conversion).",
    f"Strongest quarter: {top_quarter['quarter']}. Best-selling product: {best_product['product_name']}.",
    f"Top traffic/revenue channel: {top_source['utm_source']}. Top device: {top_device['device_type']}.",
    f"Repeat visitors are {repeat_pct}% of sessions. Highest refund month: {worst_refund_month['month']}.",
    "Overall the business shows a clear revenue leader by channel/product/quarter — the Diagnostic section "
    "below investigates the 'why' behind these numbers.",
])




# ## SECTION 2: DIAGNOSTIC ANALYSIS

# ### 2.1 Sales & Revenue Diagnostics

# In[45]:


'''Q: Does session traffic explain the month-to-month revenue trend, or is revenue driven by something else?'''
rev_vs_sessions = monthly_revenue[['months', 'price_usd']].merge(
    monthly_sessions, on='months', how='inner')
traffic_revenue_corr = rev_vs_sessions['price_usd'].corr(rev_vs_sessions['sessions'])
print(rev_vs_sessions.to_string(index=False))
print(f"Correlation (monthly sessions vs monthly revenue): {traffic_revenue_corr:.2f}")
insights([
    f"Sessions and revenue have a {traffic_revenue_corr:.2f} correlation month-to-month.",
    "A strong positive value (close to 1) means revenue is mostly a volume story — more traffic, more revenue.",
    "A weaker value means revenue swings are being driven by something else too, like AOV or conversion rate, not just traffic.",
])



# In[47]:


'''Q: Is revenue growth coming from more orders, or from customers spending more per order (AOV)?'''
order_growth_pct = (order_trend['order_count'].iloc[-1] - order_trend['order_count'].iloc[0]) / order_trend['order_count'].iloc[0] * 100
aov_growth_pct = (aov_trend['aov'].iloc[-1] - aov_trend['aov'].iloc[0]) / aov_trend['aov'].iloc[0] * 100
print(f"order_count growth (first to last month): {order_growth_pct:+.2f}%")
print(f"AOV growth (first to last month): {aov_growth_pct:+.2f}%")
insights([
    f"Order count changed {order_growth_pct:+.2f}% over the period.",
    f"AOV changed {aov_growth_pct:+.2f}% over the period.",
    f"{'Order volume is the bigger driver of revenue change' if abs(order_growth_pct) > abs(aov_growth_pct) else 'AOV is the bigger driver of revenue change'} — "
    f"{'focus on acquisition/conversion to keep growing' if abs(order_growth_pct) > abs(aov_growth_pct) else 'focus on upsell/bundling to keep growing'}.",
])


# ### 2.2 Order Diagnostics

# In[46]:


print("Q: Are order size (items purchased), price, and cost related to one another?")
numeric_cols = ['items_purchased', 'price_usd', 'cogs_usd']
correlation = orders[numeric_cols].corr().round(3)
print(correlation.to_string())
plt.figure(figsize=(5, 4))
plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='correlation')
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
plt.yticks(range(len(numeric_cols)), numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, correlation.iloc[i, j], ha='center', va='center', color='black')
plt.title('Correlation Heatmap (Order-level)')
plt.tight_layout()
plt.show()
strongest_pair = correlation.where(~np.eye(len(numeric_cols), dtype=bool)).abs().stack().idxmax()
insights([
    f"{strongest_pair[0]} and {strongest_pair[1]} are the most correlated pair ({correlation.loc[strongest_pair]:.2f}).",
    "The other variable pairs move fairly independently of each other.",
    "This tells us whether bigger orders are simply higher-priced items, or a genuinely different customer behavior (more items, not pricier ones).",
])


# ### 2.3 Product Diagnostics

# In[66]:


print("Q: Which products carry the highest refund risk?")
items_with_product = order_items.merge(products, on='product_id')
items_with_refund = items_with_product.merge(refunds[['order_item_id', 'refund_amount_usd']], on='order_item_id', how='left')
items_with_refund['was_refunded'] = items_with_refund['refund_amount_usd'].notna()
refund_rate_by_product = items_with_refund.groupby('product_name').agg(
    units_sold=('order_item_id', 'nunique'), units_refunded=('was_refunded', 'sum'),
    refund_amount=('refund_amount_usd', 'sum')).reset_index()
refund_rate_by_product['refund_rate_pct'] = (refund_rate_by_product['units_refunded'] / refund_rate_by_product['units_sold'] * 100).round(2)
refund_rate_by_product = refund_rate_by_product.sort_values('refund_rate_pct', ascending=False)
print(refund_rate_by_product.to_string(index=False))
plt.figure(figsize=(6, 6))
plt.bar(refund_rate_by_product['product_name'], refund_rate_by_product['refund_rate_pct'], color='blue')
plt.title('Refund Rate by Product')
plt.xlabel('Product')
plt.ylabel('Refund rate (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
worst_refund = refund_rate_by_product.iloc[0]
best_refund = refund_rate_by_product.iloc[-1]
insights([
    f"{worst_refund['product_name']} has the highest refund rate ({worst_refund['refund_rate_pct']}% of units sold) — likely a quality, sizing, or expectation-setting issue.",
    f"{best_refund['product_name']} has the lowest refund rate ({best_refund['refund_rate_pct']}%) — worth reviewing what it does right (listing, packaging, QC).",
    "Products with high revenue AND high refund rate deserve priority review — they carry both upside and risk.",
])


# ### 2.4 Traffic & Channel Diagnostics

# In[38]:


merged_grp = sessions.merge(orders[['order_id', 'website_session_id']], on='website_session_id', how='left')


# In[67]:


'''Q: Why do some acquisition channels convert better than others?'''
conversion_by_source = merged_grp.groupby('utm_source').agg(
    sessions=('website_session_id', 'nunique'), orders=('order_id', 'nunique')).reset_index()
conversion_by_source['conversion_rate_pct'] = (conversion_by_source['orders'] / conversion_by_source['sessions'] * 100).round(2)
conversion_by_source = conversion_by_source.sort_values('conversion_rate_pct', ascending=False)
print(conversion_by_source.to_string(index=False))
plt.figure(figsize=(7, 4))
plt.bar(conversion_by_source['utm_source'], conversion_by_source['conversion_rate_pct'], color='steelblue')
plt.title('Conversion Rate by Source')
plt.xlabel('Source')
plt.ylabel('Conversion rate (%)')
plt.tight_layout()
plt.show()
best_conv_source = conversion_by_source.iloc[0]
worst_conv_source = conversion_by_source.iloc[-1]
insights([
    f"{best_conv_source['utm_source']} converts best ({best_conv_source['conversion_rate_pct']}%).",
    f"{worst_conv_source['utm_source']} converts worst ({worst_conv_source['conversion_rate_pct']}%).",
    "Traffic quality/intent differs meaningfully by channel — budget should weight toward the higher-converting sources, not just the highest-volume ones.",
])


# In[68]:


'''Q: Why does one device type convert better than another?'''
conversion_by_device = merged_grp.groupby('device_type').agg(
    sessions=('website_session_id', 'nunique'), orders=('order_id', 'nunique')).reset_index()
conversion_by_device['conversion_rate_pct'] = (conversion_by_device['orders'] / conversion_by_device['sessions'] * 100).round(2)
conversion_by_device = conversion_by_device.sort_values('conversion_rate_pct', ascending=False)
print(conversion_by_device.to_string(index=False))
plt.figure(figsize=(7, 4))
plt.bar(conversion_by_device['device_type'], conversion_by_device['conversion_rate_pct'], color='steelblue')
plt.title('Conversion Rate by Device')
plt.xlabel('Device')
plt.ylabel('Conversion rate (%)')
plt.tight_layout()
plt.show()
best_conv_device = conversion_by_device.iloc[0]
worst_conv_device = conversion_by_device.iloc[-1]
insights([
    f"{best_conv_device['device_type']} converts best ({best_conv_device['conversion_rate_pct']}%).",
    f"{worst_conv_device['device_type']} converts worst ({worst_conv_device['conversion_rate_pct']}%).",
    "Points to a UX or checkout friction gap between devices — worth a UX audit on the weaker-converting device.",
])


# In[41]:


'''Q: Where in the user journey (funnel) are we losing the most sessions?'''
total_session_f = sessions['website_session_id'].nunique()
sessions_with_pageview = website_pageviews['website_session_id'].nunique()
sessions_with_order = orders['website_session_id'].nunique()
funnel = pd.DataFrame({
    'stage': ["Landed (session started)", "Viewed a page", "Placed an order"],
    'sessions': [total_session_f, sessions_with_pageview, sessions_with_order],
})
funnel['pct_of_landed'] = (funnel['sessions'] / total_session_f * 100).round(2)
print(funnel.to_string(index=False))
plt.figure(figsize=(6, 4))
plt.bar(funnel['stage'], funnel['sessions'], color='teal')
plt.title('Funnel Drop-off')
plt.xlabel('Stage')
plt.ylabel('Sessions')
plt.tight_layout()
plt.show()
drop_pageview = 100 - funnel.loc[1, 'pct_of_landed']
drop_order = funnel.loc[1, 'pct_of_landed'] - funnel.loc[2, 'pct_of_landed']
biggest_drop_stage = 'landed -> viewed a page' if drop_pageview > drop_order else 'viewed a page -> placed an order'
conclusion([
    f"Biggest drop-off happens between '{biggest_drop_stage}'.",
    f"Landed -> Viewed a page loses {drop_pageview:.1f}% of sessions.",
    f"Viewed a page -> Placed an order loses {drop_order:.1f}% of sessions.",
    "That's the stage to prioritize for fixes (e.g. page load speed for the first drop, checkout friction for the second).",
])


# ### 2.5 Customer Diagnostics

# In[69]:


'''Q: Why do repeat visitors behave differently from new visitors?'''
repeat_vs_new = merged_grp.groupby('is_repeat_session').agg(
    sessions=('website_session_id', 'nunique'), orders=('order_id', 'nunique')).reset_index()
repeat_vs_new['conversion_rate_pct'] = (repeat_vs_new['orders'] / repeat_vs_new['sessions'] * 100).round(2)
repeat_vs_new = repeat_vs_new.sort_values('conversion_rate_pct', ascending=False)
print(repeat_vs_new.to_string(index=False))
labels = ['repeat' if v == 1 else 'new' for v in repeat_vs_new['is_repeat_session']]
plt.figure(figsize=(5, 4))
plt.bar(labels, repeat_vs_new['conversion_rate_pct'], color=['purple', 'grey'])
plt.title('Repeat vs New Visitor Conversion')
plt.xlabel('Visitor type')
plt.ylabel('Conversion rate (%)')
plt.tight_layout()
plt.show()
repeat_row = repeat_vs_new[repeat_vs_new['is_repeat_session'] == 1].iloc[0]
new_row = repeat_vs_new[repeat_vs_new['is_repeat_session'] == 0].iloc[0]
gap = repeat_row['conversion_rate_pct'] - new_row['conversion_rate_pct']
insights([
    f"Repeat visitors convert at {repeat_row['conversion_rate_pct']}% vs {new_row['conversion_rate_pct']}% for new visitors.",
    f"That's a {gap:.2f} percentage-point gap.",
    f"{'Familiarity/trust is clearly driving conversion — retention efforts (email, loyalty) are worth the investment' if gap > 0 else 'Repeat visits are not translating into a conversion advantage yet — worth investigating why'}.",
])


# ### 2.6 Refund Diagnostics

# In[70]:


print("Q: Is the refund trend proportional to order volume, or spiking independently?")
refund_vs_orders = monthly_refunds.merge(
    order_trend.rename(columns={'created_at': 'month'}), on='month', how='inner')
refund_vs_orders['refund_rate_pct'] = (refund_vs_orders['refund_amount_usd'] / refund_vs_orders['revenue'] * 100).round(2)
print(refund_vs_orders.to_string(index=False))
plt.figure(figsize=(8, 4))
plt.plot(refund_vs_orders['month'], refund_vs_orders['refund_rate_pct'], marker='o', color='crimson')
plt.title('Monthly Refund Rate (% of Revenue)')
plt.xlabel('Month')
plt.ylabel('Refund rate (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
worst_refund_rate_month = refund_vs_orders.loc[refund_vs_orders['refund_rate_pct'].idxmax()]
avg_refund_rate = refund_vs_orders['refund_rate_pct'].mean()
insights([
    f"Average monthly refund rate is {avg_refund_rate:.2f}% of revenue.",
    f"{worst_refund_rate_month['month']} spiked to {worst_refund_rate_month['refund_rate_pct']}% — "
    f"{'well above the average, an anomaly worth investigating' if worst_refund_rate_month['refund_rate_pct'] > avg_refund_rate * 1.5 else 'roughly in line with the normal range'}.",
    "Cross-check this against the Product Diagnostics section — a spike often traces back to one specific product batch.",
])


# ### SECTION 2 SUMMARY — Diagnostic Analysis

# In[71]:


print("Q: Overall, what's driving the patterns we saw in the descriptive numbers?")
conclusion([
    f"Revenue is {'largely a traffic/volume story' if traffic_revenue_corr > 0.7 else 'not fully explained by traffic alone'} "
    f"(sessions-vs-revenue correlation: {traffic_revenue_corr:.2f}).",
    f"{best_conv_source['utm_source']} is both a top revenue channel and the best converter — {worst_conv_source['utm_source']} "
    f"converts worst and deserves review.",
    f"{best_conv_device['device_type']} converts better than {worst_conv_device['device_type']} — a device-specific UX gap.",
    f"Biggest funnel drop-off: '{biggest_drop_stage}'.",
    f"{worst_refund['product_name']} carries the highest refund risk — pair this with its revenue rank to prioritize action.",
    "Together these point to three levers worth acting on: channel budget reallocation, device UX parity, and "
    "quality review on the highest-refund product.",
])


# ============================================================
# SECTION 3 - PREDICTIVE ANALYSIS (what's likely to happen)
#   - OneHotEncoder only for categorical encoding
# The encoder is called directly and its output is combined with the plain
# numeric column by hand — every step is visible, nothing is hidden inside a
# wrapper object.
# ============================================================


# ## SECTION 3: PREDICTIVE ANALYSIS

# In[72]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, roc_auc_score,
                              recall_score, f1_score, confusion_matrix, classification_report)


# '''Q: Can we predict, from a session's channel/campaign/device/visitor-type, whether it will convert?'''

# In[74]:


converted_ids = set(orders['website_session_id'].dropna())
sessions['converted'] = sessions['website_session_id'].isin(converted_ids).astype(int)

categorical_features = ['utm_source', 'utm_campaign', 'device_type']
binary_features = ['is_repeat_session']
target = 'converted'

keep_cols = ['website_session_id'] + categorical_features + binary_features + [target]
dataset = sessions[keep_cols].dropna(subset=categorical_features + binary_features)

print('session dataset shape:', dataset.shape)
print('baseline conversion rate: {:.2f}%'.format(dataset[target].mean() * 100))


# In[75]:


# --- Encode the categorical columns with OneHotEncoder directly ---
encoder = OneHotEncoder(handle_unknown='ignore')
X_cat = encoder.fit_transform(dataset[categorical_features]).toarray()
X_bin = dataset[binary_features].to_numpy()
X = np.hstack([X_cat, X_bin])
y = dataset[target].to_numpy()

print('encoded categories:', encoder.categories_)
print('final feature matrix shape:', X.shape)


# In[76]:


# --- Train/test split (stratified — conversion is a minority class) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# In[77]:


# --- Fit logistic regression directly — no wrapping in a Pipeline ---
# class_weight="balanced" matters — most sessions do NOT convert, and an
# unweighted model would just predict "no" for every session.
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)


# In[78]:


# --- Evaluate ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print('\n====== model evaluation ======')
print('train rows:', len(X_train), '| test rows:', len(X_test))
print('Accuracy:', round(accuracy, 4))
print('Precision:', round(precision, 4))
print('ROC-AUC:', round(roc_auc, 4))
print('Recall:', round(recall, 4))
print('F1:', round(f1, 4))
print('\nConfusion matrix [[TN, FP], [FN, TP]]:')
print(cm)
print('\n' + classification_report(y_test, y_pred, zero_division=0))


# In[81]:


conclusion([
    f"The model ranks conversion likelihood with an ROC-AUC of {roc_auc:.2f}",
    f"Precision is {precision:.2%} and recall is {recall:.2%}",
    "Channel, campaign, device, and repeat-visitor status alone only weakly predict an individual session's conversion.",
    "Best used for relative scoring/targeting (e.g. rank sessions by predicted probability) rather than confident yes/no calls.",
])


# In[82]:


# --- Save model + encoder so the Streamlit app can transform new input the
# exact same way -
import pickle

model_bundle = {
    'model': model,
    'encoder': encoder,
    'categorical_features': categorical_features,
    'binary_features': binary_features,
}
with open("conversion_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("\nModel bundle saved to conversion_model.pkl")


# ### SECTION 3 SUMMARY — Predictive Analysis

# In[85]:


print("Q: Overall, how reliable and useful is this model for the business?")
conclusion([
    f" conversion rate in the data is {dataset[target].mean() * 100:.2f}%.",
    f"Logistic Regression on utm_source, utm_campaign, device_type (one-hot encoded) + is_repeat_session ",
    f"achieves ROC-AUC {roc_auc:.2f}, precision {precision:.2%}, recall {recall:.2%}.",
])

print("\nPipeline complete.")


# In[ ]:




