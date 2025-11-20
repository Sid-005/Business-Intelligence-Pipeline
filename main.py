# all import statements
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
#------------------------------------------------------------------------------
# I. DATA LOADING AND CLEANING
base = Path(__file__).parent

def read_csv_safe(path: Path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}\nWorking directory: {base}")
        raise

sales = read_csv_safe(base / 'group_24' / 'sales_24.csv')
products = read_csv_safe(base / 'group_24' / 'products_24.csv')
customers = read_csv_safe(base / 'group_24' / 'customers_24.csv')

# checking basic info
for df, name in [(sales, "sales"), (products, "products"), (customers, "customers")]:
    print(name)
    print(df.head(), "\n")
    print(df.info(), "\n")
    print(df.isna().sum(), "\n")

# convert the date object in data frame to a Dtype of datetime
sales["Invoice date"] = pd.to_datetime(sales["Invoice date"], format="%d/%m/%Y")

sales_exp = (
    sales.assign(Product_ID=sales["Product id list"].str.split(","))
    .explode("Product_ID")
    .drop(columns=["Product id list"])
)
sales_exp["Product id"] = sales_exp["Product_ID"].str.strip()
sales_exp = sales_exp.drop(columns=["Product_ID"]) #removing the temporary column
sales_prods = sales_exp.merge(products, on="Product id", how="left")
combined_data = sales_prods.merge(customers, on="Customer id", how="left")

print(combined_data.head())

# write combined_data to CSV
out_path = base / 'outputs' / 'Cleaned-Data' / 'combined_data.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
combined_data.to_csv(out_path, index=False)
print(f"Wrote combined data to {out_path}") #confirmation message

# adding calculated columns for descriptive analysis
combined_data["Quantity"] = 1
combined_data["LineAmount"] = combined_data["Price"] * combined_data["Quantity"]
combined_data["Year"]  = combined_data["Invoice date"].dt.year
combined_data["Month"] = combined_data["Invoice date"].dt.to_period("M")
combined_data["Weekday"] = combined_data["Invoice date"].dt.day_name()

# II. DESCRIPTIVE ANALYSIS

#1. overall kpis
total_revenue = combined_data["LineAmount"].sum()
total_orders = combined_data["Invoice no"].nunique() #aka the number of total invoices
total_customers = combined_data["Customer id"].nunique()
basket_size = combined_data.groupby("Invoice no")["Quantity"].sum().mean()
basket_value = combined_data.groupby("Invoice no")["LineAmount"].sum().mean()

#2. sales by mall
mall_revenue = combined_data.groupby("Shopping mall")["LineAmount"].sum().reset_index().sort_values(by="LineAmount", ascending=False)
mall_txn = combined_data.groupby("Shopping mall")["Invoice no"].nunique().reset_index().sort_values(by="Invoice no", ascending=False)

#3. Sales by product category
category_revenue = combined_data.groupby("Category")["LineAmount"].sum().reset_index().sort_values(by="LineAmount", ascending=False)
top_products = (
    combined_data.groupby(["Product id", "Category"])["LineAmount"]
    .sum()
    .reset_index()
    .sort_values(by="LineAmount", ascending=False)
    .groupby("Category")
    .head(10)
    .reset_index(drop=True)
)

#4. sales by customer demographics
combined_data["Age Group"] = pd.cut(combined_data["Age"], bins=[0,18,30,45,60,100], labels=["<=18","19-30","31-45","46-60","60+"])
gender_category = pd.crosstab(combined_data["Gender"], combined_data["Category"])
age_mall_category = pd.crosstab(combined_data["Age Group"], combined_data["Shopping mall"])
payment_share = combined_data.groupby("Payment method")["LineAmount"].sum().reset_index()

#5. Time patterns
monthly_revenue = combined_data.groupby("Month")["LineAmount"].sum().reset_index()
weekday_revenue = combined_data.groupby("Weekday")["LineAmount"].sum().reset_index()


# III. DESCRIPTIVE ANAYSIS - VISUALIZATION SECTION
#1. Overview of KPIs
print ("\n\n KPIs (OVERVIEW):")
print(f"Total Revenue: {total_revenue}")
print(f"Total Orders: {total_orders}")
print(f"Total Customers: {total_customers}")
print(f"Average Basket Size: {basket_size}")
print(f"Average Basket Value: {basket_value}")
print ()

# 2. Bar charts for Mall Revenue and Mall Transactions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2.1 Bar chart for Revenue by Mall
axes[0].bar(mall_revenue["Shopping mall"], mall_revenue["LineAmount"], color='steelblue', edgecolor='black')
axes[0].set_title("Revenue by Shopping Mall", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Shopping Mall", fontsize=12)
axes[0].set_ylabel("Revenue (in Million $)", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# 2.2 Bar chart for Transactions by Mall
axes[1].bar(mall_txn["Shopping mall"], mall_txn["Invoice no"], color='coral', edgecolor='black')
axes[1].set_title("Transactions by Shopping Mall", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Shopping Mall", fontsize=12)
axes[1].set_ylabel("Number of Transactions", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '2.mall-analysis-barCharts.png', dpi=300, bbox_inches='tight')
print(f"Saved chart to {base / 'outputs' / 'Visualizations' / '2.mall-analysis-barCharts.png'}")
plt.show()

# 3. Categorical Analysis
# 3.1 Pie chart for Revenue Share by Category
fig_pie, ax_pie = plt.subplots(figsize=(12, 9))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#F39C12']
wedges, texts, autotexts = ax_pie.pie(
    category_revenue["LineAmount"], 
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 12, 'weight': 'bold'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 3}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(13)
legend_labels = [f"{cat}: ${rev:,.0f}" for cat, rev in zip(category_revenue["Category"], category_revenue["LineAmount"])]
ax_pie.legend(
    legend_labels, 
    loc='center left', 
    bbox_to_anchor=(1, 0, 0.5, 1), 
    fontsize=12, 
    title='Category Details',
    title_fontsize=13,
    frameon=True, 
    fancybox=True, 
    shadow=True,
    framealpha=0.95
)
ax_pie.set_title("Revenue Share by Category", fontsize=16, fontweight='bold', pad=20)
fig_pie.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '3.1-revenue-per-category-pieChart.png', dpi=300, bbox_inches='tight')
print(f"Saved pie chart to {base / 'outputs' / 'Visualizations' / '3.1-revenue-per-category-pieChart.png'}")
plt.show()

# 3.2 Horizontal stacked bar chart for top 10 products per category
fig_products, ax_products = plt.subplots(figsize=(20, 7))
stacked_data = top_products.pivot_table(
    index='Category', 
    columns='Product id', 
    values='LineAmount', 
    fill_value=0
)
stacked_data = stacked_data.loc[
    stacked_data.sum(axis=1).sort_values(ascending=False).index
]
contrasting_colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C7ECEE',
    '#D5F4E6', '#FADBD8', '#FCF3CF', '#D5F5E3', '#EBDEF0',
    '#E8DAEF', '#D7BDE2', '#C39BD3', '#AF7AC5', '#9B59B6',
    '#7D3C98', '#6C3483', '#5B2C6F', '#52BE80', '#45B649',
    '#27AE60', '#229954', '#1E8449', '#1ABC9C', '#16A085',
    '#138D75', '#0E6251', '#117864', '#145A32', '#0B5345',
    '#F39C12', '#E67E22', '#D68910', '#CA6F1E', '#BA4A00',
    '#EC7063', '#E74C3C', '#C0392B', '#A93226', '#922B21',
    '#E59866', '#DC7633', '#CD6155', '#A569BD', '#7D3C98'
]
stacked_data.plot(
    kind='barh',
    stacked=True,
    ax=ax_products,
    color=contrasting_colors[:len(stacked_data.columns)],
    edgecolor='black',
    linewidth=1.2
)
ax_products.set_xlabel("Revenue ($)", fontsize=14, fontweight='bold')
ax_products.set_ylabel("Category", fontsize=14, fontweight='bold')
ax_products.set_title("Top 10 Products Revenue Distribution by Category", fontsize=16, fontweight='bold', pad=20)
ax_products.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
ax_products.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax_products.tick_params(axis='both', which='major', labelsize=12)

for container in ax_products.containers:
    product_id = container.get_label()
    if product_id and product_id.startswith('P'):
        labels = [product_id] * len(container)
        ax_products.bar_label(container, labels=labels, label_type='center', fontsize=8, fontweight='bold', color='white')

handles, labels = ax_products.get_legend_handles_labels()
ax_products.legend(
    handles, 
    labels,
    title='Product ID', 
    bbox_to_anchor=(1.01, 1), 
    loc='upper left', 
    fontsize=9, 
    title_fontsize=12,
    ncol=2,
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95
)
ax_products.spines['top'].set_visible(False)
ax_products.spines['right'].set_visible(False)
ax_products.spines['left'].set_linewidth(1.5)
ax_products.spines['bottom'].set_linewidth(1.5)
ax_products.set_facecolor('#f8f9fa')
fig_products.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '3.2-top-10-per-category-stackedBarChart.png', dpi=300, bbox_inches='tight')
print(f"Saved stacked top products chart to {base / 'outputs' / 'Visualizations' / '3.2-top-10-per-category-stackedBarChart.png'}")
plt.show()

# # 3.2.2 Printing Top 10 products per category 
# print("\nTop 10 Products per Category:")
# for category in top_products["Category"].unique():
#     print(f"\n{category}:")
#     cat_products = top_products[top_products["Category"] == category][["Product id", "LineAmount"]].reset_index(drop=True)
#     cat_products.index = cat_products.index + 1
#     print(cat_products.to_string())

# 4. CUSTOMER DEMOGRAPHICS VISUALIZATIONS

# 4.1 Histogram for Age Distribution by Shopping Mall
fig_demo, ax_demo = plt.subplots(figsize=(14, 7))
malls = combined_data["Shopping mall"].unique()
colors_mall = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# Prepare data for each mall
mall_data_list = []
mall_labels = []
for mall in malls:
    mall_data_list.append(combined_data[combined_data["Shopping mall"] == mall]["Age"])
    mall_labels.append(mall)

# Create stacked histogram
ax_demo.hist(mall_data_list, bins=15, label=mall_labels, color=colors_mall[:len(malls)], 
             edgecolor='black', linewidth=0.5, stacked=True)

ax_demo.set_xlabel("Age", fontsize=13, fontweight='bold')
ax_demo.set_ylabel("Frequency (Number of Customers)", fontsize=13, fontweight='bold')
ax_demo.set_title("Age Distribution of Customers by Shopping Mall", fontsize=15, fontweight='bold', pad=20)
ax_demo.legend(fontsize=11, title='Shopping Mall', title_fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
ax_demo.grid(axis='y', alpha=0.3, linestyle='--')
ax_demo.tick_params(axis='both', which='major', labelsize=11)
ax_demo.set_facecolor('#f8f9fa')
fig_demo.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '4.1-age-distribution-histogram.png', dpi=300, bbox_inches='tight')
print(f"Saved age distribution chart to {base / 'outputs' / 'Visualizations' / '4.1-age-distribution-histogram.png'}")
plt.show()

# 4.2 Bar chart for Gender vs Category
fig_gender, ax_gender = plt.subplots(figsize=(12, 6))

gender_category.T.plot(
    kind='bar',
    ax=ax_gender,
    color=['#FF6B6B', '#4ECDC4'],
    edgecolor='black',
    linewidth=1.2,
    width=0.7
)
ax_gender.set_xlabel("Product Category", fontsize=13, fontweight='bold')
ax_gender.set_ylabel("Number of Customers", fontsize=13, fontweight='bold')
ax_gender.set_title("Customer Distribution by Gender across Product Categories", fontsize=15, fontweight='bold', pad=20)
ax_gender.legend(['Male', 'Female'], fontsize=11, title='Gender', title_fontsize=12, frameon=True, fancybox=True, shadow=True)
ax_gender.grid(axis='y', alpha=0.3, linestyle='--')
ax_gender.tick_params(axis='x', rotation=45, labelsize=11)
ax_gender.tick_params(axis='y', labelsize=11)
ax_gender.set_facecolor('#f8f9fa')
fig_gender.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '4.2-gender-category-barChart.png', dpi=300, bbox_inches='tight')
print(f"Saved gender category chart to {base / 'outputs' / 'Visualizations' / '4.2-gender-category-barChart.png'}")
plt.show()

# 4.3 Bar chart for Payment Method Share
fig_payment, ax_payment = plt.subplots(figsize=(10, 6))

payment_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax_payment.bar(payment_share["Payment method"], payment_share["LineAmount"], 
                       color=payment_colors, edgecolor='black', linewidth=1.5)

ax_payment.set_xlabel("Payment Method", fontsize=13, fontweight='bold')
ax_payment.set_ylabel("Revenue ($)", fontsize=13, fontweight='bold')
ax_payment.set_title("Revenue by Payment Method", fontsize=15, fontweight='bold', pad=20)
ax_payment.grid(axis='y', alpha=0.3, linestyle='--')
ax_payment.tick_params(axis='both', which='major', labelsize=11)
ax_payment.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
for bar in bars:
    height = bar.get_height()
    ax_payment.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height/1e6:.2f}M',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

ax_payment.set_facecolor('#f8f9fa')
fig_payment.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '4.3-payment-methods-barChart.png', dpi=300, bbox_inches='tight')
print(f"Saved payment method chart to {base / 'outputs' / 'Visualizations' / '4.3-payment-methods-barChart.png'}")
plt.show()

# 5. TIME PATTERNS VISUALIZATIONS

# 5.1 Line chart for Monthly Revenue Trends
fig_time, ax_time = plt.subplots(figsize=(14, 7))

monthly_revenue['Month_str'] = monthly_revenue['Month'].astype(str)

ax_time.plot(range(len(monthly_revenue)), monthly_revenue['LineAmount'], 
             marker='o', linewidth=2.5, markersize=8, color='#FF6B6B', label='Monthly Revenue')
ax_time.fill_between(range(len(monthly_revenue)), monthly_revenue['LineAmount'], alpha=0.2, color='#FF6B6B')

ax_time.set_xlabel("Month", fontsize=13, fontweight='bold')
ax_time.set_ylabel("Revenue ($)", fontsize=13, fontweight='bold')
ax_time.set_title("Monthly Revenue Trends", fontsize=15, fontweight='bold', pad=20)
ax_time.grid(True, alpha=0.3, linestyle='--')
ax_time.tick_params(axis='both', which='major', labelsize=11)
ax_time.set_xticks(range(len(monthly_revenue)))
ax_time.set_xticklabels(monthly_revenue['Month_str'], rotation=45, ha='right')
ax_time.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

for i, (idx, row) in enumerate(monthly_revenue.iterrows()):
    ax_time.text(i, row['LineAmount'], f'${row["LineAmount"]/1e6:.2f}M', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_time.set_facecolor('#f8f9fa')
fig_time.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '5.1-monthly-revenue-lineChart.png', dpi=300, bbox_inches='tight')
print(f"Saved monthly revenue chart to {base / 'outputs' / 'Visualizations' / '5.1-monthly-revenue-lineChart.png'}")
plt.show()

# 5.2 Line chart for Weekday Revenue Patterns
fig_weekday, ax_weekday = plt.subplots(figsize=(12, 7))

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_revenue['Weekday'] = pd.Categorical(weekday_revenue['Weekday'], categories=day_order, ordered=True)
weekday_revenue_sorted = weekday_revenue.sort_values('Weekday')

ax_weekday.plot(range(len(weekday_revenue_sorted)), weekday_revenue_sorted['LineAmount'], 
                marker='s', linewidth=2.5, markersize=10, color='#4ECDC4', label='Daily Revenue')
ax_weekday.fill_between(range(len(weekday_revenue_sorted)), weekday_revenue_sorted['LineAmount'], alpha=0.2, color='#4ECDC4')

ax_weekday.set_xlabel("Day of Week", fontsize=13, fontweight='bold')
ax_weekday.set_ylabel("Revenue ($)", fontsize=13, fontweight='bold')
ax_weekday.set_title("Weekday Revenue Patterns", fontsize=15, fontweight='bold', pad=20)
ax_weekday.grid(True, alpha=0.3, linestyle='--')
ax_weekday.tick_params(axis='both', which='major', labelsize=11)
ax_weekday.set_xticks(range(len(weekday_revenue_sorted)))
ax_weekday.set_xticklabels(weekday_revenue_sorted['Weekday'], rotation=45, ha='right')
ax_weekday.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))

for i, (idx, row) in enumerate(weekday_revenue_sorted.iterrows()):
    ax_weekday.text(i, row['LineAmount'], f'${row["LineAmount"]/1e6:.2f}M', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_weekday.set_facecolor('#f8f9fa')
fig_weekday.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '5.2-weekday-revenue-lineChart.png', dpi=300, bbox_inches='tight')
print(f"Saved weekday revenue chart to {base / 'outputs' / 'Visualizations' / '5.2-weekday-revenue-lineChart.png'}")
plt.show()

# IV. FEATURE ENGINEERING FOR CUSTOMER LEVEL ANALYSIS
# 1. Recency, Frequency, Monetary (RFM) values for each customer
snapshot_date = combined_data["Invoice date"].max() + pd.Timedelta(days=1)
customers = combined_data.groupby("Customer id").agg(
    last_purchase_date = ("Invoice date", "max"),
    frequency = ("Invoice no", "nunique"),
    monetary = ("LineAmount", "sum"),
    age = ("Age", "first"),
    gender = ("Gender", "first")
)
customers["recency"] = (snapshot_date - customers["last_purchase_date"]).dt.days
customers.drop(columns=["last_purchase_date"], inplace=True)

# 2. Adding composition features
category_pivot = (combined_data.pivot_table(index="Customer id",
                                            columns="Category",
                                            values="LineAmount",
                                            aggfunc="sum",
                                            fill_value=0))
category_pivot = category_pivot.div(category_pivot.sum(axis=1), axis=0) # Note: Percentage by customer
mall_preference = combined_data.groupby("Customer id")["Shopping mall"].agg(lambda x: x.value_counts().idxmax())
payment_preference = combined_data.groupby("Customer id")["Payment method"].agg(lambda x: x.value_counts().idxmax())
customers = customers.join(category_pivot, how="left")
customers["mall_preference"] = mall_preference
customers["payment_preference"] = payment_preference
out_path = base / 'outputs' / 'Cleaned-Data' / 'RFM-Features.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
customers.to_csv(out_path, index=False)
print("\nCustomer-level RFM and Composition Features:")
print(customers.head())
print ("wrote to csv file at:", out_path)

# V. DATA MINING PHASE 1: CUSTOMER SEGMENTATION VIA CLUSTERING
# Note: Clustering to identiy key customer groups and their purchasing patterns
features = ["recency", "frequency", "monetary", "age"] + list(category_pivot.columns)
mining = customers[features].fillna(0)
scaler = StandardScaler()
mining_scaled = scaler.fit_transform(mining)

scores = {}
for k in range(3,8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(mining_scaled)
    silhouette_avg = silhouette_score(mining_scaled, cluster_labels)
    scores[k] = silhouette_avg
    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")
print (scores)
optimal_k = max(scores, key=scores.get)
print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
customers['Cluster'] = kmeans_final.fit_predict(mining_scaled)

out_path = base / 'outputs' / 'Cleaned-Data' / 'Customer-Clusters.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
customers.to_csv(out_path, index=False)
print("\nCustomer Clusters Assigned:")
print(customers[['Cluster']].head())
print ("wrote to csv file at:", out_path)

# Note: Profiling each cluster for business insights
cluster_profiles = customers.groupby('Cluster')[features].mean()
print ("\nCluster Profiles:\n", cluster_profiles)

cluster_category_preference = combined_data.merge(customers[['Cluster']], left_on='Customer id', right_index=True).groupby(['Cluster', 'Category'])['LineAmount'].sum().reset_index()
cluster_category_preference['LineAmount'] = cluster_category_preference.groupby('Cluster')['LineAmount'].transform(lambda x: x / x.sum())
print ("\nCluster Category Preferences:\n", cluster_category_preference)

cluster_mall_preference = combined_data.merge(customers[['Cluster']], left_on='Customer id', right_index=True).groupby(['Cluster', 'Shopping mall'])['LineAmount'].sum().reset_index()
cluster_mall_preference['LineAmount'] = cluster_mall_preference.groupby('Cluster')['LineAmount'].transform(lambda x: x / x.sum())
print ("\nCluster Mall Preferences:\n", cluster_mall_preference)

# VI. CLUSTER VISUALIZATIONS

# 6.1 2D PCA Scatter Plot
print("\n=== Generating Cluster Visualizations ===")
pca_2d = PCA(n_components=2, random_state=42)
principal_components_2d = pca_2d.fit_transform(mining_scaled)
pca_df_2d = pd.DataFrame(data=principal_components_2d, columns=['PC1', 'PC2'])
pca_df_2d['Cluster'] = customers['Cluster'].values

fig_pca2d, ax_pca2d = plt.subplots(figsize=(12, 8))
cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#F39C12', '#98D8C8', '#BB8FCE']
for cluster in sorted(pca_df_2d['Cluster'].unique()):
    cluster_data = pca_df_2d[pca_df_2d['Cluster'] == cluster]
    ax_pca2d.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                     c=[cluster_colors[cluster]], 
                     label=f'Cluster {cluster}',
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

ax_pca2d.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=13, fontweight='bold')
ax_pca2d.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=13, fontweight='bold')
ax_pca2d.set_title('Customer Clusters - 2D PCA Visualization', fontsize=16, fontweight='bold', pad=20)
ax_pca2d.legend(fontsize=11, title='Cluster', title_fontsize=12, frameon=True, fancybox=True, shadow=True)
ax_pca2d.grid(True, alpha=0.3, linestyle='--')
ax_pca2d.set_facecolor('#f8f9fa')
fig_pca2d.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '6.1-clusters-2d-pca.png', dpi=300, bbox_inches='tight')
print(f"Saved 2D PCA cluster visualization to {base / 'outputs' / 'Visualizations' / '6.1-clusters-2d-pca.png'}")
plt.show()

# 6.2 3D PCA Scatter Plot
pca_3d = PCA(n_components=3, random_state=42)
principal_components_3d = pca_3d.fit_transform(mining_scaled)
pca_df_3d = pd.DataFrame(data=principal_components_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['Cluster'] = customers['Cluster'].values

fig_pca3d = plt.figure(figsize=(14, 10))
ax_pca3d = fig_pca3d.add_subplot(111, projection='3d')

for cluster in sorted(pca_df_3d['Cluster'].unique()):
    cluster_data = pca_df_3d[pca_df_3d['Cluster'] == cluster]
    ax_pca3d.scatter(cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'],
                     c=[cluster_colors[cluster]], 
                     label=f'Cluster {cluster}',
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

ax_pca3d.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
ax_pca3d.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=12, fontweight='bold')
ax_pca3d.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=12, fontweight='bold')
ax_pca3d.set_title('Customer Clusters - 3D PCA Visualization', fontsize=16, fontweight='bold', pad=20)
ax_pca3d.legend(fontsize=10, title='Cluster', title_fontsize=11, loc='upper left')
ax_pca3d.view_init(elev=20, azim=45)
fig_pca3d.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '6.2-clusters-3d-pca.png', dpi=300, bbox_inches='tight')
print(f"Saved 3D PCA cluster visualization to {base / 'outputs' / 'Visualizations' / '6.2-clusters-3d-pca.png'}")
plt.show()

# 6.3 RFM Pair Plot (Recency, Frequency, Monetary)
rfm_plot_data = customers[['recency', 'frequency', 'monetary', 'Cluster']].copy()
rfm_plot_data['Cluster'] = rfm_plot_data['Cluster'].astype(str)

fig_rfm, axes_rfm = plt.subplots(3, 3, figsize=(16, 14))
fig_rfm.suptitle('RFM Analysis - Pair Plot by Cluster', fontsize=18, fontweight='bold', y=0.995)

rfm_features = ['recency', 'frequency', 'monetary']
for i, feat1 in enumerate(rfm_features):
    for j, feat2 in enumerate(rfm_features):
        ax = axes_rfm[i, j]
        
        if i == j:
            # Diagonal: histograms
            for cluster in sorted(rfm_plot_data['Cluster'].unique()):
                cluster_data = rfm_plot_data[rfm_plot_data['Cluster'] == cluster]
                ax.hist(cluster_data[feat1], bins=20, alpha=0.6, 
                       color=cluster_colors[int(cluster)], label=f'Cluster {cluster}', edgecolor='black')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            if i == 0:
                ax.legend(fontsize=8, loc='upper right')
        else:
            # Off-diagonal: scatter plots
            for cluster in sorted(rfm_plot_data['Cluster'].unique()):
                cluster_data = rfm_plot_data[rfm_plot_data['Cluster'] == cluster]
                ax.scatter(cluster_data[feat2], cluster_data[feat1], 
                          c=[cluster_colors[int(cluster)]], s=30, alpha=0.6, 
                          edgecolors='black', linewidth=0.3)
        
        if i == 2:
            ax.set_xlabel(feat2.capitalize(), fontsize=11, fontweight='bold')
        if j == 0:
            ax.set_ylabel(feat1.capitalize(), fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '6.3-clusters-rfm-pairplot.png', dpi=300, bbox_inches='tight')
print(f"Saved RFM pair plot to {base / 'outputs' / 'Visualizations' / '6.3-clusters-rfm-pairplot.png'}")
plt.show()

# 6.4 Radar Chart for Cluster Profiles
cluster_profiles_normalized = cluster_profiles.copy()
for col in cluster_profiles_normalized.columns:
    min_val = cluster_profiles_normalized[col].min()
    max_val = cluster_profiles_normalized[col].max()
    cluster_profiles_normalized[col] = (cluster_profiles_normalized[col] - min_val) / (max_val - min_val) if max_val != min_val else 0

num_clusters = len(cluster_profiles_normalized)
fig_radar, axes_radar = plt.subplots(1, num_clusters, figsize=(5*num_clusters, 5), subplot_kw=dict(projection='polar'))
if num_clusters == 1:
    axes_radar = [axes_radar]

feature_labels = cluster_profiles_normalized.columns.tolist()
num_vars = len(feature_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for idx, (cluster_id, ax_radar) in enumerate(zip(cluster_profiles_normalized.index, axes_radar)):
    values = cluster_profiles_normalized.loc[cluster_id].tolist()
    values += values[:1]
    
    ax_radar.plot(angles, values, 'o-', linewidth=2, color=cluster_colors[cluster_id], label=f'Cluster {cluster_id}')
    ax_radar.fill(angles, values, alpha=0.25, color=cluster_colors[cluster_id])
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(feature_labels, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title(f'Cluster {cluster_id} Profile', fontsize=13, fontweight='bold', pad=20)
    ax_radar.grid(True, alpha=0.3)
    ax_radar.set_facecolor('#f8f9fa')

fig_radar.suptitle('Normalized Cluster Profiles - Radar Charts', fontsize=16, fontweight='bold', y=1.02)
fig_radar.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(base / 'outputs' / 'Visualizations' / '6.4-clusters-radar-profiles.png', dpi=300, bbox_inches='tight')
print(f"Saved radar chart profiles to {base / 'outputs' / 'Visualizations' / '6.4-clusters-radar-profiles.png'}")
plt.show()

print("\n=== All cluster visualizations completed successfully ===")
print(f"Total variance explained by first 2 PCs: {pca_2d.explained_variance_ratio_.sum():.1%}")
print(f"Total variance explained by first 3 PCs: {pca_3d.explained_variance_ratio_.sum():.1%}")

# VII. ASSOCIATION RULES FOR PRODUCT BUNDLING
# Note: Using Apriori algorithm to identify frequently bought together products
print("\n=== Generating Association Rules ===")

# Create basket matrix (transactions x products)
basket = (combined_data.groupby(['Invoice no', 'Product id'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Invoice no'))

# Convert to binary (1 if product purchased, 0 otherwise)
def encode_units(x):
    if x <= 0:
        return 0
    return 1

basket_sets = basket.applymap(encode_units)

print(f"Basket matrix shape: {basket_sets.shape}")
print(f"Total transactions: {len(basket_sets)}")
print(f"Total unique products: {len(basket_sets.columns)}")

# Apply Apriori algorithm to find frequent itemsets
# min_support=0.01 means item(s) must appear in at least 1% of transactions
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print(f"\nFound {len(frequent_itemsets)} frequent itemsets")
print("\nTop 10 most frequent itemsets:")
print(frequent_itemsets.nlargest(10, 'support')[['support', 'itemsets', 'length']])

# Generate association rules
# metric='lift' measures how much more likely items are bought together vs independently
# min_threshold=1.0 means we want rules where lift > 1 (positive association)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

print(f"\n{len(rules)} association rules generated")

# Filter for practical rules (support > 0.01, confidence > 0.3, lift > 1.2)
strong_rules = rules[(rules['support'] >= 0.01) & 
                     (rules['confidence'] >= 0.3) & 
                     (rules['lift'] >= 1.2)]

print(f"\n{len(strong_rules)} strong association rules found")
print("\nTop 20 Association Rules:")
print(strong_rules.head(20)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Export rules to CSV
rules_output = strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']].copy()
rules_output['antecedents'] = rules_output['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_output['consequents'] = rules_output['consequents'].apply(lambda x: ', '.join(list(x)))

out_path = base / 'outputs' / 'Cleaned-Data' / 'Association-Rules.csv'
rules_output.to_csv(out_path, index=False)
print(f"\nWrote association rules to: {out_path}")

# Visualization: Top 15 rules by lift
if len(strong_rules) > 0:
    fig_rules, ax_rules = plt.subplots(figsize=(14, 10))
    
    top_rules = strong_rules.head(15).copy()
    top_rules['rule'] = top_rules.apply(
        lambda x: f"{', '.join(list(x['antecedents']))} → {', '.join(list(x['consequents']))}", 
        axis=1
    )
    
    # Create scatter plot with bubble size representing support
    scatter = ax_rules.scatter(
        top_rules['confidence'], 
        top_rules['lift'],
        s=top_rules['support'] * 10000,  # Scale support for bubble size
        c=range(len(top_rules)),
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5
    )
    
    # Add rule labels
    for idx, row in top_rules.iterrows():
        ax_rules.annotate(
            row['rule'],
            (row['confidence'], row['lift']),
            fontsize=8,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray')
        )
    
    ax_rules.set_xlabel('Confidence', fontsize=13, fontweight='bold')
    ax_rules.set_ylabel('Lift', fontsize=13, fontweight='bold')
    ax_rules.set_title('Top 15 Association Rules\n(Bubble size = Support)', fontsize=16, fontweight='bold', pad=20)
    ax_rules.grid(True, alpha=0.3, linestyle='--')
    ax_rules.set_facecolor('#f8f9fa')
    fig_rules.patch.set_facecolor('white')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_rules)
    cbar.set_label('Rule Rank', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(base / 'outputs' / 'Visualizations' / '7.1-association-rules-scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved association rules visualization to {base / 'outputs' / 'Visualizations' / '7.1-association-rules-scatter.png'}")
    plt.show()
    
    # Visualization 2: Heatmap of top product pairs
    if len(strong_rules) >= 10:
        fig_heat, ax_heat = plt.subplots(figsize=(12, 10))
        
        top_10_rules = strong_rules.head(10)
        heatmap_data = []
        labels = []
        
        for idx, row in top_10_rules.iterrows():
            ante = ', '.join(list(row['antecedents']))
            cons = ', '.join(list(row['consequents']))
            labels.append(f"{ante} → {cons}")
            heatmap_data.append([row['support'], row['confidence'], row['lift']])
        
        heatmap_df = pd.DataFrame(heatmap_data, columns=['Support', 'Confidence', 'Lift'], index=labels)
        
        # Normalize for better visualization
        im = ax_heat.imshow(heatmap_df.values, cmap='YlOrRd', aspect='auto')
        
        ax_heat.set_xticks(range(len(heatmap_df.columns)))
        ax_heat.set_yticks(range(len(heatmap_df.index)))
        ax_heat.set_xticklabels(heatmap_df.columns, fontsize=12, fontweight='bold')
        ax_heat.set_yticklabels(heatmap_df.index, fontsize=9)
        
        # Add values to cells
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                text = ax_heat.text(j, i, f'{heatmap_df.values[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        ax_heat.set_title('Top 10 Association Rules - Metrics Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax_heat)
        cbar.set_label('Metric Value', fontsize=11, fontweight='bold')
        
        fig_heat.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(base / 'outputs' / 'Visualizations' / '7.2-association-rules-heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved association rules heatmap to {base / 'outputs' / 'Visualizations' / '7.2-association-rules-heatmap.png'}")
        plt.show()

print("\n=== Association Rules Analysis Completed ===")
print(f"Key Insights:")
print(f"- Average Confidence: {strong_rules['confidence'].mean():.2%}")
print(f"- Average Lift: {strong_rules['lift'].mean():.2f}")
print(f"- Strongest rule has lift of {strong_rules['lift'].max():.2f}")