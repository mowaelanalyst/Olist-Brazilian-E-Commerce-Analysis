import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged order, item, customer, and review tables
order = pd.read_csv("Orders.csv")
item = pd.read_csv("Order_items.csv")
cust = pd.read_csv("Customers.csv")
reivew = pd.read_csv("Customers_review.csv")

# Bring item-level fields (product/category/price/freight) onto each order row.
# NOTE: this creates one row per item, not per order — order-level columns
# (e.g. payment_value, review_score) will repeat across an order's item rows,
# while item-level columns (price, freight_value) legitimately differ per row.
order = order.merge(item[["order_id","product_id","product_category_name_english","freight_value","price"]],on="order_id",how="left")
order = order.merge(cust[["customer_id","customer_city","customer_unique_id","customer_state"]],on="customer_id",how="left")
order = order.merge(reivew[["order_id","review_score","review_comment_message"]],on="order_id",how="left")

# Exclude canceled/unavailable orders — these never completed a real transaction
order = order[~order["order_status"].isin(["Canceled","Unavailable"])]

order["order_purchase_timestamp"] = pd.to_datetime(order["order_purchase_timestamp"]) 
order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"]) 

order["Delivery_duration_days"] = (order["order_delivered_customer_date"] - order["order_purchase_timestamp"]).dt.days

order["Year"] = order["order_purchase_timestamp"].dt.year
order["Month"] = order["order_purchase_timestamp"].dt.month
order["Day"] = order["order_purchase_timestamp"].dt.day

# Restrict to 2016-2017 (2018 is a partial year in this dataset)
order = order[order["Year"] <= 2017].copy()

order = order.sort_values(by=["Year","Month"])
# Overall monthly sales trend (item-level price summed per month) for the line chart
Previous_sales = order.groupby(["Year","Month"])["price"].sum().reset_index()

order = order.sort_values(by=["Year","Month"])
Previous_sales_MoM = order.groupby(["Year","Month"])["price"].sum().reset_index()
# Month-over-month growth as a simple sequential pct_change (compares each month
# to the one immediately before it, regardless of year)
Previous_sales_MoM["Growth_Rate"] = Previous_sales_MoM["price"].pct_change()*100
# Cap out extreme early-period spikes (near-zero prior-month base) as NaN rather
# than plotting a misleading capped number
Previous_sales_MoM["Growth_Rate"] = np.where(
    Previous_sales_MoM["Growth_Rate"] > 1000 ,
    np.nan,
    Previous_sales_MoM["Growth_Rate"] 
)

m_Mom = Previous_sales_MoM["Growth_Rate"].median()
MoM = Previous_sales_MoM[["Year","Month","Growth_Rate"]]

# ============================================================================
# SECTION A — March (baseline) vs May (event)
# ============================================================================

# Baseline period: Jan-Mar 2017 (a 3-month cumulative window, not March alone —
# used for the "leaders before May" charts)
Period = order[(order["Year"] == 2017) & (order["Month"].between(1,3))].copy()

# Sales per state during the Jan-Mar baseline, for context/comparison with the
# May contribution-to-change chart
Sales_per_state = Period.groupby("customer_state")["price"].sum().reset_index()
Sales_per_state_10 = Sales_per_state.sort_values(by="price",ascending=False).head(10)

# Units sold per category during the Jan-Mar baseline (same purpose as above)
Units_per_category = Period.groupby("product_category_name_english")["order_id"].count().reset_index()
Units_per_category_15 = Units_per_category.sort_values(by="order_id",ascending=False).head(10)

# Full 2017 data, used for both the March-vs-May and May-vs-June comparisons below
Target_Period = order[(order["Year"] == 2017)].copy() 

# Daily order count within May, to inspect the shape of the event within the month
Orders_pre_Days_May = Target_Period[Target_Period["Month"] == 5]
Orders_pre_May_Days = Orders_pre_Days_May.groupby("Day")["order_id"].nunique().reset_index()
Orders_pre_May_Days = Orders_pre_May_Days.sort_values(by="Day",ascending=True)

Target_Period["Year_Month"] = Target_Period["Year"].astype(str) + "-" + Target_Period["Month"].astype(str).str.zfill(2)


# Contribution to Change (units): for each category, compute the raw change in
# units sold between March and May, then express that change as a % of the
# TOTAL change across all categories — i.e. which categories drove the May
# shift, not just which categories are largest overall
category_contribution_to_change = Target_Period.groupby(["Year_Month","product_category_name_english"])["order_id"].count().reset_index()
category_contribution_to_change = (category_contribution_to_change.pivot(index="product_category_name_english",
columns="Year_Month",values="order_id").fillna(0)).reset_index()

category_contribution_to_change["Change"] = (
    category_contribution_to_change["2017-05"]
    -
    category_contribution_to_change["2017-03"]
)
Total_change = category_contribution_to_change["Change"].sum()
# NOTE: divides by raw Total_change (not abs(Total_change)) here, unlike the
# June-section equivalent below — if Total_change is negative, a declining
# category's sign would flip incorrectly. Flagging only, not changed.
category_contribution_to_change["contribution_to_change"] = ((category_contribution_to_change["Change"] / Total_change) *100)
category_contribution_to_change = category_contribution_to_change.sort_values(by="contribution_to_change",ascending=False).head(10)

# Same Contribution to Change logic, applied to sales value per state
contribution_to_change = Target_Period.groupby(["customer_state","Year_Month"])["price"].sum().reset_index()

contribution_to_change = (contribution_to_change.pivot(index="customer_state",columns="Year_Month",values="price").fillna(0)).reset_index()

contribution_to_change["Change"] = (
    contribution_to_change["2017-05"]
    -
    contribution_to_change["2017-03"]
)



Total_change = contribution_to_change["Change"].sum()
# Same note as above: raw Total_change is used here too, not its absolute value
contribution_to_change["contribution_to_change"] = ((contribution_to_change["Change"] / Total_change) *100)
contribution_to_change = contribution_to_change.sort_values(by="contribution_to_change",ascending=False).head(10)

# Cumulative comparison: Mar totals vs May totals, to quantify May's
# overall impact on the year-to-date figures
Total_Sales = Period[Period["Month"] == 3 ] ["price"].sum()
Total_Sales_after_May = Target_Period[Target_Period["Month"] == 5 ] ["price"].sum()
Growth_sales_cuse_May = (((Total_Sales_after_May - Total_Sales)/Total_Sales)*100)

# order_id.nunique() counts distinct orders (item-level fan-out doesn't inflate this)
Total_Orders = Period[Period["Month"] == 3 ]["order_id"].nunique()
Total_Orders_after_May = Target_Period[Target_Period["Month"] == 5 ] ["order_id"].nunique()
Growth_order_cuse_May = (((Total_Orders_after_May-Total_Orders)/Total_Orders)*100)

# order_id.count() here counts item rows, i.e. total units sold (one row per item)
Total_Sold_units = Period[Period["Month"] == 3 ]["order_id"].count()
Total_Sold_units_after_May = Target_Period[Target_Period["Month"] == 5 ] ["order_id"].count()
Growth_Sold_units_cuse_May = (((Total_Sold_units_after_May-Total_Sold_units)/Total_Sold_units)*100)

aov = (Total_Sales / Total_Orders)
aov_aetr_May= (Total_Sales_after_May / Total_Orders_after_May )

Total_Customers = Period[Period["Month"] == 3 ]["customer_unique_id"].nunique()
Total_Customers_after_May = Target_Period[Target_Period["Month"] == 5 ] ["customer_unique_id"].nunique()
# Growth is measured relative to the pre-event baseline (Total_Customers),
# consistent with how sales/orders growth is calculated above
Growth_Cust_cuse_May = (((Total_Customers_after_May-Total_Customers)/Total_Customers)*100)

# First-ever purchase date per customer (across the full order history), used
# to identify which of May's customers were brand-new vs returning
New_customers = order.groupby("customer_unique_id")["order_purchase_timestamp"].min().reset_index()
New_customers_May = New_customers[(New_customers["order_purchase_timestamp"].dt.year == 2017) &
(New_customers["order_purchase_timestamp"].dt.month == 5)]
Total_New_Customer = New_customers_May["customer_unique_id"].shape[0]
# Share of May's customer base that made their first-ever purchase in May
pre_of_new = f"{((Total_New_Customer / Total_Customers_after_May))*100:,.2f}"


avg_delivery_days = Period[Period["Month"] == 3 ]["Delivery_duration_days"].mean()
avg_delivery_days_after_May = Target_Period[Target_Period["Month"] == 5 ]["Delivery_duration_days"].mean()

frequency = Total_Sold_units / Total_Customers
frequency_after_May = Total_Sold_units_after_May / Total_Customers_after_May

# freight_value is item-level (varies per product within the same order), so it
# is summed directly across all item rows without deduplicating on order_id —
# deduplicating here would drop real freight amounts for multi-item orders
Total_freight_cost = Period[Period["Month"] == 3 ]["freight_value"].sum()
Avg_freight_cost = Period[Period["Month"] == 3 ]["freight_value"].mean()
Total_freight_cost_after_May = Target_Period[Target_Period["Month"] == 5 ]["freight_value"].sum()
Avg_freight_cost_after_May = Target_Period[Target_Period["Month"] == 5 ]["freight_value"].mean()
Growth_cost_cuse_May = (((Total_freight_cost_after_May-Total_freight_cost)/Total_freight_cost)*100)

# price is also item-level, so its median is computed directly on the item rows
# (no dedup) to reflect the true distribution of individual item prices
avg_price = Period[Period["Month"] == 3 ]["price"].median()
avg_price_after_May = Target_Period[Target_Period["Month"] == 5 ]["price"].median()

# review_score is order-level (one score per order, repeated across that
# order's item rows), so it requires dropping duplicate order_id rows first
# to avoid counting the same review multiple times

No_dup_period = Period.drop_duplicates("order_id").copy()
No_dup_Target_period = Target_Period.drop_duplicates("order_id").copy()

avg_reivew_score = No_dup_period[No_dup_period["Month"] == 3 ]["review_score"].mean()
# NOTE: this variable is named "_after_May" and holds May's review score. It is
# reused later in the June dashboard (Section B) under the "Jun" label — see
# the flag near avg_reivew_score_after_Jun below.
avg_reivew_score_after_May = No_dup_Target_period[No_dup_Target_period["Month"] == 5 ]["review_score"].mean()


# --- Dashboard 1 (May event): Contribution-to-change & baseline leaders ---
fg , ax = plt.subplots(2,2,figsize=(15,10))

bars = ax[0,0].barh(
    contribution_to_change["customer_state"],
    contribution_to_change["contribution_to_change"],
    color=["green" if x > 0 else "red" for x in contribution_to_change["contribution_to_change"]]
)
ax[0,0].set_title("Contribution To Change Of Sales Caused May")
ax[0,0].bar_label(bars,fmt="%.2f%%",padding=2)
ax[0,0].set_xlim(
    contribution_to_change["contribution_to_change"].min() * 1.0,
    contribution_to_change["contribution_to_change"].max() * 1.15
)


sales_bar = ax[0,1].barh(
    Sales_per_state_10["customer_state"],
    Sales_per_state_10["price"],
    color="skyblue"
)
ax[0,1].bar_label(sales_bar,fmt="{:,.0f}",padding=2)
ax[0,1].set_title("Sales Per State (Mar), Cities That Were Leading Sales Before May?")
ax[0,1].set_xlim(
    Sales_per_state_10["price"].min() * 1.0,
    Sales_per_state_10["price"].max() * 1.15
)

cat_CRM_bars = ax[1,0].barh(
    category_contribution_to_change["product_category_name_english"],
    category_contribution_to_change["contribution_to_change"],
    color=["green" if x > 0 else "red" for x in category_contribution_to_change["contribution_to_change"]]
)
ax[1,0].set_title("Contribution To Change Of Units Sold Caused May")
ax[1,0].bar_label(cat_CRM_bars,fmt="%.2f%%",padding=2)
ax[1,0].set_xlim(
    category_contribution_to_change["contribution_to_change"].min() * 1.0,
    category_contribution_to_change["contribution_to_change"].max() * 1.15
)

units_cat_bar = ax[1,1].barh(
    Units_per_category_15["product_category_name_english"],
    Units_per_category_15["order_id"],
    color="skyblue"
)
ax[1,1].bar_label(units_cat_bar,padding=2)
ax[1,1].set_title("Units Per Category (Mar), Cities That Were Leading Units Before May?")
ax[1,1].set_xlim(
    Units_per_category_15["order_id"].min() * 1.0,
    Units_per_category_15["order_id"].max() * 1.15
)

plt.tight_layout(h_pad=3.0)

# --- Dashboard 2 (May event): KPI summary text panel (Mar vs May) ---
f , k = plt.subplots(1,1,figsize=(15,10))
plt.axis("off")


f.text(0.5, 0.94, " 2017 Performance  (Mar vs May)", fontsize=15, fontweight="bold", ha="center")


col1, col2, col3, col4 = 0.125, 0.375, 0.625, 0.875
y_head = 0.84
y_row1 = 0.68
y_row2 = 0.48
y_row3 = 0.28


f.text(col1, y_head, "[ Sales & Revenue ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col1, y_row1, f"Total Sales:\nMar: {Total_Sales:,.0f}  |  May: {Total_Sales_after_May:,.0f}\n({Growth_sales_cuse_May:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f.text(col1, y_row2, f"AOV:\nMar: {aov:,.1f}  |  May: {aov_aetr_May:,.1f}", fontsize=10, fontweight="bold", ha="center")
f.text(col1, y_row3, f"Median Item Price:\nMar: {avg_price:,.1f}  |  May: {avg_price_after_May:,.1f}", fontsize=10, fontweight="bold", ha="center")


f.text(col2, y_head, "[ Orders & Volume ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col2, y_row1, f"Total Orders:\nMar: {Total_Orders:,.0f}  |  May: {Total_Orders_after_May:,.0f}\n({Growth_order_cuse_May:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f.text(col2, y_row2, f"Total Sold Units:\nMar: {Total_Sold_units:,.0f}  |  May: {Total_Sold_units_after_May:,.0f}\n({Growth_Sold_units_cuse_May:+.1f}%)", fontsize=10, fontweight="bold", ha="center")

f.text(col3, y_head, "[ Customers & Review ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col3, y_row1, f"Total Customers:\nMar: {Total_Customers:,.0f}  |  May: {Total_Customers_after_May:,.0f}\n({Growth_Cust_cuse_May:+.1f}%)\n New Customers: {Total_New_Customer}:({pre_of_new})%", fontsize=10, fontweight="bold", ha="center")
f.text(col3, y_row2, f"Purchase Frequency:\nMar: {frequency:.2f}  |  May: {frequency_after_May:.2f}", fontsize=10, fontweight="bold", ha="center")
f.text(col3, y_row3, f"Avg Review Score:\nMar: {avg_reivew_score:.2f}  |  May: {avg_reivew_score_after_May:.2f}", fontsize=10, fontweight="bold", ha="center")


f.text(col4, y_head, "[ Logistics & Freight ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col4, y_row1, f"Total Freight Cost:\nMar: {Total_freight_cost:,.0f}  |  May: {Total_freight_cost_after_May:,.0f}\n({Growth_cost_cuse_May:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f.text(col4, y_row2, f"Avg Freight Cost:\nMar: {Avg_freight_cost:,.1f}  |  May: {Avg_freight_cost_after_May:,.1f}", fontsize=10, fontweight="bold", ha="center")
f.text(col4, y_row3, f"Avg Delivery Days:\nMar: {avg_delivery_days:.1f}d  |  May: {avg_delivery_days_after_May:.1f}d", fontsize=10, fontweight="bold", ha="center")

# --- Dashboard 3: Overall MoM growth rate and monthly sales trend (2016-2017) ---
fig , axse = plt.subplots(1,2,figsize=(15,10))


sns.lineplot(data=MoM,
x="Month",
y="Growth_Rate",
hue="Year",
ax=axse[0],
marker="o"
)
axse[0].set_title(f"Growth Rate MoM (Median:{m_Mom:,.2f}% )")

for x, y in zip(MoM["Month"], MoM["Growth_Rate"]):
    axse[0].annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,8), ha='center')
    
sns.lineplot(data=Previous_sales,
x="Month",
y="price",
hue="Year",
ax=axse[1],
marker="o"
)

axse[1].set_title("Total Sales Overtime")

for x, y in zip(Previous_sales["Month"], Previous_sales["price"]):
        axse[1].annotate(f'{y:,.0f}$', (x, y), textcoords="offset points", 
                xytext=(0,3), ha='center')



plt.tight_layout(h_pad=3.0)


# ============================================================================
# SECTION B — May (baseline) vs June (event)
# ============================================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Baseline period: Jan-May 2017 (cumulative window, used for the "leaders
# before June" charts)
Period_Jun = order[(order["Year"] == 2017) & (order["Month"].between(1,5))].copy()

Sales_per_state_Jun = Period_Jun.groupby("customer_state")["price"].sum().reset_index()
Sales_per_state_Jun_10_J = Sales_per_state_Jun.sort_values(by="price",ascending=False).head(10)


Units_per_category_Jun = Period_Jun.groupby("product_category_name_english")["order_id"].count().reset_index()
Units_per_category_15_J = Units_per_category_Jun.sort_values(by="order_id",ascending=False).head(10)

# Contribution to Change (units), May vs June — same methodology as Section A
category_contribution_to_change_J = Target_Period.groupby(["Year_Month","product_category_name_english"])["order_id"].count().reset_index()
category_contribution_to_change_J = (category_contribution_to_change_J.pivot(index="product_category_name_english",
columns="Year_Month",values="order_id").fillna(0)).reset_index()

category_contribution_to_change_J["Change"] = (
    category_contribution_to_change_J["2017-06"]
    -
    category_contribution_to_change_J["2017-05"]
)
Total_change_J = category_contribution_to_change_J["Change"].sum()

# Uses abs(Total_change_J) so a category's sign always reflects its own
# direction, regardless of whether the overall total change was positive or
# negative (fixes the sign issue present in Section A's version above)
category_contribution_to_change_J["contribution_to_change_J"] = ((category_contribution_to_change_J["Change"] / abs(Total_change_J)) *100)

category_contribution_to_change_J = category_contribution_to_change_J.sort_values(by="contribution_to_change_J",ascending=True).head(10)


# Contribution to Change (sales value), May vs June
contribution_to_change_J = Target_Period.groupby(["customer_state","Year_Month"])["price"].sum().reset_index()

contribution_to_change_J = (contribution_to_change_J.pivot(index="customer_state",columns="Year_Month",values="price").fillna(0)).reset_index()

contribution_to_change_J["Change"] = (
    contribution_to_change_J["2017-06"]
    -
    contribution_to_change_J["2017-05"]
)

Total_change_J = contribution_to_change_J["Change"].sum()
contribution_to_change_J["contribution_to_change_J"] = ((contribution_to_change_J["Change"] / abs(Total_change_J) )*100)


contribution_to_change_J = contribution_to_change_J.sort_values(by="contribution_to_change_J",ascending=True).head(10)


# --- Direct month-over-month comparison: May vs June ---
Total_Sales_J = Period_Jun[Period_Jun["Month"] == 5 ]["price"].sum()
Total_Sales_after_Jun = Target_Period[Target_Period["Month"] == 6 ] ["price"].sum()
Total_Sales_Growth_Jun = (((Total_Sales_after_Jun - Total_Sales_J)/Total_Sales_J)*100)

Total_Orders_J = Period_Jun[Period_Jun["Month"] == 5 ]["order_id"].nunique()
Total_Orders_after_Jun = Target_Period[Target_Period["Month"] == 6 ] ["order_id"].nunique()
Total_Order_Growth_Jun = (((Total_Orders_after_Jun - Total_Orders_J)/Total_Orders_J)*100)

Total_Sold_units_J = Period_Jun[Period_Jun["Month"] == 5 ]["order_id"].count()
Total_Sold_units_after_Jun = Target_Period[Target_Period["Month"] == 6 ] ["order_id"].count()
Sold_Units_Growth_Jun= (((Total_Sold_units_after_Jun-Total_Sold_units_J)/Total_Sold_units_J)*100)


aov_J = (Total_Sales_J / Total_Orders_J )
aov_aetr_Jun = (Total_Sales_after_Jun / Total_Orders_after_Jun )

Total_Customers_J = Period_Jun[Period_Jun["Month"] == 5 ]["customer_unique_id"].nunique()
Total_Customers_after_Jun = Target_Period[Target_Period["Month"] == 6 ] ["customer_unique_id"].nunique()
# Growth is measured relative to the pre-event baseline (Total_Customers_J),
# consistent with every other growth calculation in this script
The_impact_of_Cust_cuse_Jun = (((Total_Customers_after_Jun-Total_Customers_J)/Total_Customers_J)*100)

avg_delivery_days_J = Period_Jun[Period_Jun["Month"] == 5 ]["Delivery_duration_days"].mean()
avg_delivery_days_after_Jun = Target_Period[Target_Period["Month"] == 6 ]["Delivery_duration_days"].mean()

frequency_J = Total_Sold_units_J / Total_Customers_J
frequency_after_Jun = Total_Sold_units_after_Jun / Total_Customers_after_Jun


# price and freight_value are item-level, so both are computed directly on
# Period_Jun / Target_Period (no dedup) — deduplicating on order_id here would
# drop real per-item values for multi-item orders
avg_price_J = Period_Jun[Period_Jun["Month"] == 5 ]["price"].median()
avg_price_after_Jun = Target_Period[Target_Period["Month"] == 6 ]["price"].median()


Total_freight_cost_Jun = Period_Jun[Period_Jun["Month"] == 5 ]["freight_value"].sum()
Avg_freight_cost_Jun = Period_Jun[Period_Jun["Month"] == 5 ]["freight_value"].mean()
Total_freight_cost_after_Jun = Target_Period[Target_Period["Month"] == 6 ]["freight_value"].sum()
Avg_freight_cost_after_Jun = Target_Period[Target_Period["Month"] == 6 ]["freight_value"].mean()
The_impact_of_cost_cuse_Jun = (((Total_freight_cost_after_Jun-Total_freight_cost_Jun)/Total_freight_cost_Jun)*100)

# review_score is order-level, so dedup on order_id before averaging
No_dup_period_J = Period_Jun.drop_duplicates("order_id").copy()
No_dup_Target_period_J = Target_Period.drop_duplicates("order_id").copy()


avg_reivew_score_J = No_dup_period_J[No_dup_period_J["Month"] == 5 ]["review_score"].mean()
avg_reivew_score_after_Jun = No_dup_Target_period_J[No_dup_Target_period_J["Month"] == 6 ]["review_score"].mean()

# --- Dashboard 1 (June event): Contribution-to-change & baseline leaders ---
fg_J , ax_J = plt.subplots(2,2,figsize=(15,10))

bars = ax_J[0,0].barh(
    contribution_to_change_J["customer_state"],
    contribution_to_change_J["contribution_to_change_J"],
    color=["green" if x > 0 else "red" for x in contribution_to_change_J["contribution_to_change_J"]]
)
ax_J[0,0].set_title("Contribution To Change Of Sales Caused Jun")
ax_J[0,0].bar_label(bars,fmt="%.2f%%",padding=2)
ax_J[0,0].set_xlim(
    contribution_to_change_J["contribution_to_change_J"].min()* 1.5 ,
    contribution_to_change_J["contribution_to_change_J"].max() *1.15
)
cat_CRM_bars = ax_J[1,0].barh(
    category_contribution_to_change_J["product_category_name_english"],
    category_contribution_to_change_J["contribution_to_change_J"],
    color=["green" if x > 0 else "red" for x in category_contribution_to_change_J["contribution_to_change_J"]]
)
ax_J[1,0].set_title("Contribution To Change Of Units Caused Jun")
ax_J[1,0].bar_label(cat_CRM_bars,fmt="%.2f%%",padding=2)
ax_J[1,0].set_xlim(
    category_contribution_to_change_J["contribution_to_change_J"].min() * 1.5,
    category_contribution_to_change_J["contribution_to_change_J"].max() * 1.15
)
sales_bar = ax_J[0,1].barh(
    Sales_per_state_Jun_10_J["customer_state"],
    Sales_per_state_Jun_10_J["price"],
    color="skyblue"
)
ax_J[0,1].bar_label(sales_bar,fmt="{:,.0f}",padding=2)
ax_J[0,1].set_title("Sales Per State(Jan-May), Cities That Were Leading Sales Before Jun?")
ax_J[0,1].set_xlim(
    Sales_per_state_Jun_10_J["price"].min() * 1.0,
    Sales_per_state_Jun_10_J["price"].max() * 1.15
)

units_cat_bar_J = ax_J[1,1].barh(
    Units_per_category_15_J["product_category_name_english"],
    Units_per_category_15_J["order_id"],
    color="skyblue"
)
ax_J[1,1].bar_label(units_cat_bar_J,padding=2)
ax_J[1,1].set_title("Units per Category(Jan-May), Cities That Were Leading Sales Before Jun?")
ax_J[1,1].set_xlim(
    Units_per_category_15_J["order_id"].min() * 1.0,
    Units_per_category_15_J["order_id"].max() * 1.15
)

plt.tight_layout(h_pad=3.0)
f_J , k = plt.subplots(1,1,figsize=(15,10))
plt.axis("off")


f_J.text(0.5, 0.94, "E-Commerce Performance Metrics (May vs. Jun)", fontsize=15, fontweight="bold", ha="center")

col1, col2, col3, col4 = 0.125, 0.375, 0.625, 0.875
y_head = 0.84
y_row1 = 0.68
y_row2 = 0.48
y_row3 = 0.28


f_J.text(col1, y_head, "[ Sales & Revenue ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f_J.text(col1, y_row1, f"Total Sales:\nMay: {Total_Sales_J:,.0f}  |  Jun: {Total_Sales_after_Jun:,.0f}\n({Total_Sales_Growth_Jun:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f_J.text(col1, y_row2, f"AOV:\nJan-May: {aov_J:,.1f}  |  Jan-Jun: {aov_aetr_Jun:,.1f}", fontsize=10, fontweight="bold", ha="center")
f_J.text(col1, y_row3, f"Median Item Price:\nMay: {avg_price_J:,.1f}  |  Jun: {avg_price_after_Jun:,.1f}", fontsize=10, fontweight="bold", ha="center")


f_J.text(col2, y_head, "[ Orders & Volume ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f_J.text(col2, y_row1, f"Total Orders:\nMay: {Total_Orders_J:,.0f}  |  Jun: {Total_Orders_after_Jun:,.0f}\n({Total_Order_Growth_Jun:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f_J.text(col2, y_row2, f"Total Sold Units:\nMay: {Total_Sold_units_J:,.0f}  |  Jun: {Total_Sold_units_after_Jun:,.0f}\n({Sold_Units_Growth_Jun:+.1f}%)", fontsize=10, fontweight="bold", ha="center")


f_J.text(col3, y_head, "[ Customers & Review ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f_J.text(col3, y_row1, f"Total Customers:\nMay: {Total_Customers_J:,.0f}  |  Jun: {Total_Customers_after_Jun:,.0f}\n({The_impact_of_Cust_cuse_Jun:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f_J.text(col3, y_row2, f"Purchase Frequency:\nMay: {frequency_J:.2f}  |  Jun: {frequency_after_Jun:.2f}", fontsize=10, fontweight="bold", ha="center")


f_J.text(col3, y_row3, f"Avg Review Score:\nMay: {avg_reivew_score_J:.2f}  |  Jun: {avg_reivew_score_after_Jun:.2f}", fontsize=10, fontweight="bold", ha="center")


f_J.text(col4, y_head, "[ Logistics & Freight ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f_J.text(col4, y_row1, f"Total Freight Cost:\nMay: {Total_freight_cost_Jun:,.0f}  |  Jun: {Total_freight_cost_after_Jun:,.0f}\n({The_impact_of_cost_cuse_Jun:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f_J.text(col4, y_row2, f"Avg Freight Cost:\nMay: {Avg_freight_cost_Jun:,.1f}  |  Jun: {Avg_freight_cost_after_Jun:,.1f}", fontsize=10, fontweight="bold", ha="center")
f_J.text(col4, y_row3, f"Avg Delivery Days:\nMay: {avg_delivery_days_J:.1f}d  |  Jun: {avg_delivery_days_after_Jun:.1f}d", fontsize=10, fontweight="bold", ha="center")


plt.tight_layout(h_pad=3.0)

plt.show()
