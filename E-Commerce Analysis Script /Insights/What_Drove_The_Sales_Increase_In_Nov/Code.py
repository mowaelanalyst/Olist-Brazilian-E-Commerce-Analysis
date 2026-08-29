import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# UNIFIED COLOR PALETTE & REUSABLE PLOTTING HELPERS
# ============================================================================
COLOR_UP = "#2ecc71"
COLOR_DOWN = "#e74c3c"
COLOR_NEUTRAL = "#87CEEB"
COLOR_AVG_LINE = "#7f7f7f"


def kpi_line(ax, x, y, label, before, after, growth=None, fmt=",.1f", unit="", size=10, extra=""):
    """Draws one color/arrow-coded KPI block (green+up / red+down / black+neutral)."""
    if growth is None:
        color, arrow = "black", ""
    else:
        color = COLOR_UP if growth > 0 else COLOR_DOWN if growth < 0 else "black"
        arrow = "▲" if growth > 0 else "▼" if growth < 0 else "►"

    text = f"{label}\n{before:{fmt}}{unit}  →  {after:{fmt}}{unit}"
    if growth is not None:
        text += f"\n{arrow} {growth:+.1f}%"
    if extra:
        text += f"\n{extra}"

    ax.text(x, y, text, fontsize=size, fontweight="bold", ha="center", color=color)


def plot_contribution_to_change(ax, df, label_col, value_col, title):
    """Contribution-to-change barh chart with a zero reference line."""
    bars = ax.barh(
        df[label_col],
        df[value_col],
        color=[COLOR_UP if v > 0 else COLOR_DOWN for v in df[value_col]]
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    pad = (df[value_col].max() - df[value_col].min()) * 0.15
    ax.set_xlim(df[value_col].min() - pad, df[value_col].max() + pad)


def plot_leaders_bar(ax, df, label_col, value_col, title, fmt="{:,.0f}"):
    """Plain 'leaders before the event' barh chart, using the neutral color."""
    bars = ax.barh(df[label_col], df[value_col], color=COLOR_NEUTRAL)
    ax.bar_label(bars, fmt=fmt, padding=3)
    ax.set_title(title)
    pad = df[value_col].max() * 0.15
    ax.set_xlim(df[value_col].min() * 1.0, df[value_col].max() + pad)


def plot_daily_orders(daily_df, day_col, count_col, month_label):
    """Daily order-count line chart with a monthly-average reference line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_df[day_col], daily_df[count_col], marker="o", color=COLOR_NEUTRAL, linewidth=2)

    avg_val = daily_df[count_col].mean()
    ax.axhline(avg_val, color=COLOR_AVG_LINE, linestyle="--", alpha=0.7,
               label=f"Monthly Avg: {avg_val:,.0f}")

    for x, y in zip(daily_df[day_col], daily_df[count_col]):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, fontweight="bold")

    ax.set_title(f"Orders Over {month_label}")
    ax.set_xlabel(f"Day of {month_label}")
    ax.set_ylabel("Orders")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax

order = pd.read_csv("Orders.csv")
item = pd.read_csv("Order_items.csv")
cust = pd.read_csv("Customers.csv")
reivew = pd.read_csv("Customers_review.csv")

order = order.merge(item[["order_id","product_id","product_category_name_english","freight_value","price"]],on="order_id",how="left")
order = order.merge(cust[["customer_id","customer_city","customer_unique_id","customer_state"]],on="customer_id",how="left")
order = order.merge(reivew[["order_id","review_score","review_comment_message"]],on="order_id",how="left")

order = order[~order["order_status"].isin(["Canceled","Unavailable"])]

order["order_purchase_timestamp"] = pd.to_datetime(order["order_purchase_timestamp"]) 
order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"]) 
order["Delivery_duration_days"] = (order["order_delivered_customer_date"] - order["order_purchase_timestamp"]).dt.days

order["Year"] = order["order_purchase_timestamp"].dt.year
order["Month"] = order["order_purchase_timestamp"].dt.month
order["Day"] = order["order_purchase_timestamp"].dt.day

order = order[order["Year"] <= 2017].copy()

order = order.sort_values(by=["Year","Month"])
Previous_sales = order.groupby(["Year","Month"])["price"].sum().reset_index()

Previous_sales_MoM = order.groupby(["Year","Month"])["price"].sum().reset_index()
Previous_sales_MoM["Growth_Rate"] = Previous_sales_MoM["price"].pct_change()*100
Previous_sales_MoM["Growth_Rate"] = np.where(
    Previous_sales_MoM["Growth_Rate"] > 1000, np.nan, Previous_sales_MoM["Growth_Rate"]
)
m_Mom = Previous_sales_MoM["Growth_Rate"].median()
MoM = Previous_sales_MoM[["Year","Month","Growth_Rate"]]

Target_Period = order[(order["Year"] == 2017)].copy()
Target_Period["Year_Month"] = Target_Period["Year"].astype(str) + "-" + Target_Period["Month"].astype(str).str.zfill(2)



Period = order[(order["Year"] == 2017) & (order["Month"].between(1,10))].copy()

Sales_per_state = Period.groupby("customer_state")["price"].sum().reset_index()
Sales_per_state_10 = Sales_per_state.sort_values(by="price",ascending=False).head(10)

Units_per_category = Period.groupby("product_category_name_english")["order_id"].count().reset_index()
Units_per_category_15 = Units_per_category.sort_values(by="order_id",ascending=False).head(10)

Orders_pre_Days_Nov = Target_Period[Target_Period["Month"] == 11]
Orders_pre_Nov_Days = Orders_pre_Days_Nov.groupby("Day")["order_id"].nunique().reset_index()

# Contribution to Change (units), October vs November
category_contribution_to_change = Target_Period.groupby(["Year_Month","product_category_name_english"])["order_id"].count().reset_index()
category_contribution_to_change = (category_contribution_to_change.pivot(index="product_category_name_english",
columns="Year_Month",values="order_id").fillna(0)).reset_index()
category_contribution_to_change["Change"] = (
    category_contribution_to_change["2017-11"] - category_contribution_to_change["2017-10"]
)
Total_change = category_contribution_to_change["Change"].sum()
# FIX: divide by abs(Total_change) so a category's sign always reflects its
# own direction, regardless of the overall total change's sign
category_contribution_to_change["contribution_to_change"] = (
    category_contribution_to_change["Change"] / abs(Total_change)
) * 100
category_contribution_to_change = category_contribution_to_change.sort_values(by="contribution_to_change",ascending=False).head(10)

# Contribution to Change (sales), October vs November
contribution_to_change = Target_Period.groupby(["customer_state","Year_Month"])["price"].sum().reset_index()
contribution_to_change = (contribution_to_change.pivot(index="customer_state",columns="Year_Month",values="price").fillna(0)).reset_index()
contribution_to_change["Change"] = (
    contribution_to_change["2017-11"] - contribution_to_change["2017-10"]
)
Total_change = contribution_to_change["Change"].sum()
contribution_to_change["contribution_to_change"] = (
    contribution_to_change["Change"] / abs(Total_change)
) * 100
contribution_to_change = contribution_to_change.sort_values(by="contribution_to_change",ascending=False).head(10)

# Direct month-over-month comparison: October vs November
Total_Sales = Period[Period["Month"] == 10]["price"].sum()
Total_Sales_after_Nov = Target_Period[Target_Period["Month"] == 11]["price"].sum()
Growth_sales_cuse_Nov = ((Total_Sales_after_Nov - Total_Sales)/Total_Sales)*100

Total_Orders = Period[Period["Month"] == 10]["order_id"].nunique()
Total_Orders_after_Nov = Target_Period[Target_Period["Month"] == 11]["order_id"].nunique()
Growth_order_cuse_Nov = ((Total_Orders_after_Nov - Total_Orders)/Total_Orders)*100

Total_Sold_units = Period[Period["Month"] == 10]["order_id"].count()
Total_Sold_units_after_Nov = Target_Period[Target_Period["Month"] == 11]["order_id"].count()
Growth_Sold_units_cuse_Nov = ((Total_Sold_units_after_Nov - Total_Sold_units)/Total_Sold_units)*100

aov = Total_Sales / Total_Orders
aov_aetr = Total_Sales_after_Nov / Total_Orders_after_Nov

Total_Customers = Period[Period["Month"] == 10]["customer_unique_id"].nunique()
Total_Customers_after_Nov = Target_Period[Target_Period["Month"] == 11]["customer_unique_id"].nunique()
Growth_Cust_cuse_Nov = ((Total_Customers_after_Nov - Total_Customers)/Total_Customers)*100

New_customers = order.groupby("customer_unique_id")["order_purchase_timestamp"].min().reset_index()
New_customers_Nov = New_customers[(New_customers["order_purchase_timestamp"].dt.year == 2017) &
                                   (New_customers["order_purchase_timestamp"].dt.month == 11)]
Total_New_Customer = New_customers_Nov["customer_unique_id"].shape[0]
pre_of_new = (Total_New_Customer / Total_Customers_after_Nov) * 100

avg_delivery_days = Period[Period["Month"] == 10]["Delivery_duration_days"].mean()
avg_delivery_days_after_Nov = Target_Period[Target_Period["Month"] == 11]["Delivery_duration_days"].mean()

frequency = Total_Sold_units / Total_Customers
frequency_after_Nov = Total_Sold_units_after_Nov / Total_Customers_after_Nov

Total_freight_cost = Period[Period["Month"] == 10]["freight_value"].sum()
Avg_freight_cost = Period[Period["Month"] == 10]["freight_value"].mean()
Total_freight_cost_after_Nov = Target_Period[Target_Period["Month"] == 11]["freight_value"].sum()
Avg_freight_cost_after_Nov = Target_Period[Target_Period["Month"] == 11]["freight_value"].mean()
Growth_cost_cuse_Nov = ((Total_freight_cost_after_Nov - Total_freight_cost)/Total_freight_cost)*100

avg_price = Period[Period["Month"] == 10]["price"].median()
avg_price_after_Nov = Target_Period[Target_Period["Month"] == 11]["price"].median()

No_dup_period = Period.drop_duplicates("order_id").copy()
No_dup_Target_period = Target_Period.drop_duplicates("order_id").copy()
avg_reivew_score = No_dup_period[No_dup_period["Month"] == 10]["review_score"].mean()
avg_reivew_score_after_Nov = No_dup_Target_period[No_dup_Target_period["Month"] == 11]["review_score"].mean()


# --- Dashboard 1 (Nov event): Contribution-to-change & baseline leaders ---
fg, ax = plt.subplots(2, 2, figsize=(15, 10))

plot_contribution_to_change(ax[0,0], contribution_to_change, "customer_state",
                             "contribution_to_change", "Contribution To Change Of Sales Caused Nov")
plot_leaders_bar(ax[0,1], Sales_per_state_10, "customer_state", "price",
                  "Sales Per State (Jan-Oct) — Leaders Before Nov")
plot_contribution_to_change(ax[1,0], category_contribution_to_change, "product_category_name_english",
                             "contribution_to_change", "Contribution To Change Of Units Sold Caused Nov")
plot_leaders_bar(ax[1,1], Units_per_category_15, "product_category_name_english", "order_id",
                  "Units Per Category (Jan-Oct) — Leaders Before Nov", fmt="{:,.0f}")

plt.tight_layout(h_pad=3.0)

# --- Dashboard 2 (Nov event): KPI summary text panel (color-coded) ---
f, k = plt.subplots(1, 1, figsize=(15, 10))
plt.axis("off")
f.text(0.5, 0.94, "2017 Performance (Oct vs Nov)", fontsize=15, fontweight="bold", ha="center")

col1, col2, col3, col4 = 0.125, 0.375, 0.625, 0.875
y_head, y_row1, y_row2, y_row3 = 0.84, 0.68, 0.48, 0.28

f.text(col1, y_head, "[ Sales & Revenue ]", fontsize=11, fontweight="bold", ha="center", color="navy")
kpi_line(k, col1, y_row1, "Total Sales", Total_Sales, Total_Sales_after_Nov, growth=Growth_sales_cuse_Nov, fmt=",.0f")
kpi_line(k, col1, y_row2, "AOV", aov, aov_aetr, fmt=",.1f")
kpi_line(k, col1, y_row3, "Median Item Price", avg_price, avg_price_after_Nov, fmt=",.1f")

f.text(col2, y_head, "[ Orders & Volume ]", fontsize=11, fontweight="bold", ha="center", color="navy")
kpi_line(k, col2, y_row1, "Total Orders", Total_Orders, Total_Orders_after_Nov, growth=Growth_order_cuse_Nov, fmt=",.0f")
kpi_line(k, col2, y_row2, "Total Sold Units", Total_Sold_units, Total_Sold_units_after_Nov, growth=Growth_Sold_units_cuse_Nov, fmt=",.0f")

f.text(col3, y_head, "[ Customers & Review ]", fontsize=11, fontweight="bold", ha="center", color="navy")
kpi_line(k, col3, y_row1, "Total Customers", Total_Customers, Total_Customers_after_Nov,
         growth=Growth_Cust_cuse_Nov, fmt=",.0f", extra=f"New Customers: {Total_New_Customer} ({pre_of_new:.2f}%)")
kpi_line(k, col3, y_row2, "Purchase Frequency", frequency, frequency_after_Nov, fmt=".2f")
kpi_line(k, col3, y_row3, "Avg Review Score", avg_reivew_score, avg_reivew_score_after_Nov,
         growth=(avg_reivew_score_after_Nov - avg_reivew_score), fmt=".2f")

f.text(col4, y_head, "[ Logistics & Freight ]", fontsize=11, fontweight="bold", ha="center", color="navy")
kpi_line(k, col4, y_row1, "Total Freight Cost", Total_freight_cost, Total_freight_cost_after_Nov, growth=Growth_cost_cuse_Nov, fmt=",.0f")
kpi_line(k, col4, y_row2, "Avg Freight Cost", Avg_freight_cost, Avg_freight_cost_after_Nov, fmt=",.1f")
kpi_line(k, col4, y_row3, "Avg Delivery Days", avg_delivery_days, avg_delivery_days_after_Nov, fmt=".1f", unit="d")

# --- Dashboard 3: Overall MoM growth rate and monthly sales trend ---
fig, axse = plt.subplots(1, 2, figsize=(15, 10))
sns.lineplot(data=MoM, x="Month", y="Growth_Rate", hue="Year", ax=axse[0], marker="o")
axse[0].set_title(f"Growth Rate MoM (Median:{m_Mom:,.2f}%)")
for x, y in zip(MoM["Month"], MoM["Growth_Rate"]):
    axse[0].annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,8), ha='center')

sns.lineplot(data=Previous_sales, x="Month", y="price", hue="Year", ax=axse[1], marker="o")
axse[1].set_title("Total Sales Overtime")
for x, y in zip(Previous_sales["Month"], Previous_sales["price"]):
    axse[1].annotate(f'{y:,.0f}$', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout(h_pad=3.0)

# --- Daily order count within November, with average reference line ---
plot_daily_orders(Orders_pre_Nov_Days, "Day", "order_id", "November")


# ============================================================================
# SECTION B — November (baseline) vs December (event)
# ============================================================================

Period_Dec = order[(order["Year"] == 2017) & (order["Month"].between(1,11))].copy()

Sales_per_state_Dec = Period_Dec.groupby("customer_state")["price"].sum().reset_index()
Sales_per_state_10_Dec = Sales_per_state_Dec.sort_values(by="price",ascending=False).head(10)

Units_per_category_Dec = Period_Dec.groupby("product_category_name_english")["order_id"].count().reset_index()
Units_per_category_15_Dec = Units_per_category_Dec.sort_values(by="order_id",ascending=False).head(10)

category_contribution_to_change_Dec = Target_Period.groupby(["Year_Month","product_category_name_english"])["order_id"].count().reset_index()
category_contribution_to_change_Dec = (category_contribution_to_change_Dec.pivot(index="product_category_name_english",
columns="Year_Month",values="order_id").fillna(0)).reset_index()
category_contribution_to_change_Dec["Change"] = (
    category_contribution_to_change_Dec["2017-12"] - category_contribution_to_change_Dec["2017-11"]
)
Total_change_Dec = category_contribution_to_change_Dec["Change"].sum()
category_contribution_to_change_Dec["contribution_to_change"] = (
    category_contribution_to_change_Dec["Change"] / abs(Total_change_Dec)
) * 100
category_contribution_to_change_Dec = category_contribution_to_change_Dec.sort_values(by="contribution_to_change",ascending=True).head(10)

contribution_to_change_Dec = Target_Period.groupby(["customer_state","Year_Month"])["price"].sum().reset_index()
contribution_to_change_Dec = (contribution_to_change_Dec.pivot(index="customer_state",columns="Year_Month",values="price").fillna(0)).reset_index()
contribution_to_change_Dec["Change"] = (
    contribution_to_change_Dec["2017-12"] - contribution_to_change_Dec["2017-11"]
)
Total_change_Dec = contribution_to_change_Dec["Change"].sum()
contribution_to_change_Dec["contribution_to_change"] = (
    contribution_to_change_Dec["Change"] / abs(Total_change_Dec)
) * 100
contribution_to_change_Dec = contribution_to_change_Dec.sort_values(by="contribution_to_change",ascending=True).head(10)

Total_Sales_Dec = Period_Dec[Period_Dec["Month"] == 11]["price"].sum()
Total_Sales_after_Dec = Target_Period[Target_Period["Month"] == 12]["price"].sum()
The_impact_of_sales_cuse_Dec = ((Total_Sales_after_Dec - Total_Sales_Dec)/Total_Sales_Dec)*100

Total_Orders_Dec = Period_Dec[Period_Dec["Month"] == 11]["order_id"].nunique()
Total_Orders_after_Dec = Target_Period[Target_Period["Month"] == 12]["order_id"].nunique()
The_impact_of_order_cuse_Dec = ((Total_Orders_after_Dec - Total_Orders_Dec)/Total_Orders_Dec)*100

Total_Sold_units_Dec = Period_Dec[Period_Dec["Month"] == 11]["order_id"].count()
Total_Sold_units_after_Dec = Target_Period[Target_Period["Month"] == 12]["order_id"].count()
The_impact_of_Sold_units_cuse_Dec = ((Total_Sold_units_after_Dec - Total_Sold_units_Dec)/Total_Sold_units_Dec)*100

aov_Dec = Total_Sales_Dec / Total_Orders_Dec
aov_aetr_Dec = Total_Sales_after_Dec / Total_Orders_after_Dec

Total_Customers_Dec = Period_Dec[Period_Dec["Month"] == 11]["customer_unique_id"].nunique()
Total_Customers_after_Dec = Target_Period[Target_Period["Month"] == 12]["customer_unique_id"].nunique()
The_impact_of_Cust_cuse_Dec = ((Total_Customers_after_Dec - Total_Customers_Dec)/Total_Customers_Dec)*100

avg_delivery_days_Dec = Period_Dec[Period_Dec["Month"] == 11]["Delivery_duration_days"].mean()
avg_delivery_days_after_Dec = Target_Period[Target_Period["Month"] == 12]["Delivery_duration_days"].mean()

frequency_Dec = Total_Sold_units_Dec / Total_Customers_Dec
frequency_after_Dec = Total_Sold_units_after_Dec / Total_Customers_after_Dec

avg_price_Dec = Period_Dec[Period_Dec["Month"] == 11]["price"].median()
avg_price_after_Dec = Target_Period[Target_Period["Month"] == 12]["price"].median()

Total_freight_cost_Dec = Period_Dec[Period_Dec["Month"] == 11]["freight_value"].sum()
Avg_freight_cost_Dec = Period_Dec[Period_Dec["Month"] == 11]["freight_value"].mean()
Total_freight_cost_after_Dec = Target_Period[Target_Period["Month"] == 12]["freight_value"].sum()
Avg_freight_cost_after_Dec = Target_Period[Target_Period["Month"] == 12]["freight_value"].mean()
The_impact_of_cost_cuse_Dec = ((Total_freight_cost_after_Dec - Total_freight_cost_Dec)/Total_freight_cost_Dec)*100

No_dup_period_Dec = Period_Dec.drop_duplicates("order_id").copy()
No_dup_Target_period_Dec = Target_Period.drop_duplicates("order_id").copy()
avg_reivew_score_Dec = No_dup_period_Dec[No_dup_period_Dec["Month"] == 11]["review_score"].mean()
avg_reivew_score_after_Dec = No_dup_Target_period_Dec[No_dup_Target_period_Dec["Month"] == 12]["review_score"].mean()


# --- Dashboard 1 (Dec event): Contribution-to-change & baseline leaders ---
fg_Dec, ax_Dec = plt.subplots(2, 2, figsize=(15, 10))

plot_contribution_to_change(ax_Dec[0,0], contribution_to_change_Dec, "customer_state",
                             "contribution_to_change", "Which state drove the December sales change?")
plot_leaders_bar(ax_Dec[0,1], Sales_per_state_10_Dec, "customer_state", "price",
                  "Sales Per State (Jan-Nov) — Leaders Before Dec")
plot_contribution_to_change(ax_Dec[1,0], category_contribution_to_change_Dec, "product_category_name_english",
                             "contribution_to_change", "Which category drove the December units change?")
plot_leaders_bar(ax_Dec[1,1], Units_per_category_15_Dec, "product_category_name_english", "order_id",
                  "Units Per Category (Jan-Nov) — Leaders Before Dec", fmt="{:,.0f}")

plt.tight_layout(h_pad=3.0)

# --- Dashboard 2 (Dec event): KPI summary text panel (color-coded) ---
f_Dec, k_Dec = plt.subplots(1, 1, figsize=(15, 10))
plt.axis("off")
f_Dec.text(0.5, 0.94, "E-Commerce Performance Metrics (Nov vs. Dec)", fontsize=15, fontweight="bold", ha="center")

col1, col2, col3, col4 = 0.125, 0.375, 0.625, 0.875
y_head, y_row1, y_row2, y_row3 = 0.84, 0.68, 0.48, 0.28

f_Dec.text(col1, y_head, "[ Sales & Revenue ]", fontsize=11, fontweight="bold", ha="center", color="navy")
kpi_line(k_Dec, col1, y_row1, "Total Sales", Total_Sales_Dec, Total_Sales_after_Dec, growth=The_impact_of_sales_cuse_Dec, fmt=",.0f")
kpi_line(k_Dec, col1, y_row2, "AOV", aov_Dec, aov_aetr_Dec, fmt=",.1f")
kpi_line(k_Dec, col1, y_row3, "Median Item Price", avg_price_Dec, avg_price_after_Dec, fmt=",.1f")

f_Dec.text(col2, y_head, "[ Orders & Volume ]", fontsize=11, fontweight="bold", ha="center", color="navy")
# label fixed: "Nov" instead of the original typo "JNov"
kpi_line(k_Dec, col2, y_row1, "Total Orders", Total_Orders_Dec, Total_Orders_after_Dec, growth=The_impact_of_order_cuse_Dec, fmt=",.0f")
kpi_line(k_Dec, col2, y_row2, "Total Sold Units", Total_Sold_units_Dec, Total_Sold_units_after_Dec, growth=The_impact_of_Sold_units_cuse_Dec, fmt=",.0f")

f_Dec.text(col3, y_head, "[ Customers & Review ]", fontsize=11, fontweight="bold", ha="center", color="navy")
kpi_line(k_Dec, col3, y_row1, "Total Customers", Total_Customers_Dec, Total_Customers_after_Dec, growth=The_impact_of_Cust_cuse_Dec, fmt=",.0f")
kpi_line(k_Dec, col3, y_row2, "Purchase Frequency", frequency_Dec, frequency_after_Dec, fmt=".2f")
kpi_line(k_Dec, col3, y_row3, "Avg Review Score", avg_reivew_score_Dec, avg_reivew_score_after_Dec,
         growth=(avg_reivew_score_after_Dec - avg_reivew_score_Dec), fmt=".2f")

f_Dec.text(col4, y_head, "[ Logistics & Freight ]", fontsize=11, fontweight="bold", ha="center", color="navy")
# label fixed: "Dec" instead of the original mismatched "Jan-Dec"
kpi_line(k_Dec, col4, y_row1, "Total Freight Cost", Total_freight_cost_Dec, Total_freight_cost_after_Dec, growth=The_impact_of_cost_cuse_Dec, fmt=",.0f")
kpi_line(k_Dec, col4, y_row2, "Avg Freight Cost", Avg_freight_cost_Dec, Avg_freight_cost_after_Dec, fmt=",.1f")
kpi_line(k_Dec, col4, y_row3, "Avg Delivery Days", avg_delivery_days_Dec, avg_delivery_days_after_Dec, fmt=".1f", unit="d")



plt.show()
