# Good-and-tax-GST-Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:\\Users\\vikra\\OneDrive\\Desktop\\7102_source_data.csv")

# Setup
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ----------------------
# 1️⃣ Objective: Data Structure & Quality
# -----------------------------
print("📋 Dataset Info")
print(df.info())
print("\n📊 Missing Values:\n", df.isnull().sum())
print("\n📎 Duplicates:", df.duplicated().sum())
print("\n🔍 First Few Rows:\n", df.head())

# -----------------------------
# 2️⃣ Objective: Analyze Tax Trends
# -----------------------------
# Convert month/year to datetime
df['date'] = pd.to_datetime(df['srcMonth'], errors='coerce')

# Group by return type
gst_by_type = df.groupby("GST ( Goods and Service Tax ) Return Type")[
    [
        "Payer eligible for GST ( Goods and Service Tax ) registration",
        "GST ( Goods and Service Tax ) Payers registered before due date",
        "GST ( Goods and Service Tax ) Payers registered after due date"
    ]
].sum()

print("\n📈 GST Summary by Return Type:\n", gst_by_type)

# Group by state
gst_by_state = df.groupby("srcStateName")[
    "Payer eligible for GST ( Goods and Service Tax ) registration"
].sum().sort_values(ascending=False)

# -----------------------------
# 3️⃣ Objective: Visualize Key Patterns
# -----------------------------
# Bar plot: top 10 states
top_states = gst_by_state.head(10)
sns.barplot(x=top_states.values, y=top_states.index, palette="coolwarm")
plt.title("Top 10 States by GST Eligible Payers")
plt.xlabel("Eligible Payers")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# Time series plot
df_date = df.dropna(subset=["date"])
monthly_gst = df_date.groupby("date")[
    "Payer eligible for GST ( Goods and Service Tax ) registration"
].sum()
monthly_gst.plot(marker='o', color='teal')
plt.title("Monthly GST Eligible Payers Over Time")
plt.ylabel("Eligible Payers")
plt.xlabel("Month")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 4️⃣ Objective: Outlier Detection
# -----------------------------
# Boxplot
cols = [
    "Payer eligible for GST ( Goods and Service Tax ) registration",
    "GST ( Goods and Service Tax ) Payers registered before due date",
    "GST ( Goods and Service Tax ) Payers registered after due date"
]
sns.boxplot(data=df[cols])
plt.title("Outlier Detection: GST Payers Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Detect outliers using z-score
clean_df = df[cols].dropna()
z_scores = np.abs((clean_df - clean_df.mean()) / clean_df.std())
outliers = clean_df[(z_scores > 3).any(axis=1)]
print("\n🚨 Potential Outliers Detected (Z-Score > 3):")
print(outliers.head())

# -----------------------------
# 5️⃣ Objective: Summary Insights
# -----------------------------
print("\n📊 Summary Statistics:\n", df[cols].describe())

# Correlation heatmap
sns.heatmap(df[cols].corr(), annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation between GST Metrics")
plt.tight_layout()
plt.show()

# -----------------------------
# 🎯 Bonus: Donut Chart
# -----------------------------
gst_type_totals = df.groupby("GST ( Goods and Service Tax ) Return Type")[
    "Payer eligible for GST ( Goods and Service Tax ) registration"
].sum()

colors = sns.color_palette('pastel')[0:len(gst_type_totals)]
plt.pie(
    gst_type_totals, 
    labels=gst_type_totals.index, 
    colors=colors, 
    autopct='%1.1f%%',
    startangle=140, 
    wedgeprops={'width': 0.4}
)
plt.title("Donut Chart: GST Return Type Distribution by Eligible Payers")
plt.tight_layout()
plt.show()

# -----------------------------
# 🎯 Bonus: State vs Return Type Heatmap
# -----------------------------
pivot_table = df.pivot_table(
    index="srcStateName",
    columns="GST ( Goods and Service Tax ) Return Type",
    values="Payer eligible for GST ( Goods and Service Tax ) registration",
    aggfunc="sum"
)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=False, linewidths=0.5)
plt.title("Heatmap: GST Eligible Payers by State and Return Type")
plt.ylabel("State")
plt.xlabel("GST Return Type")
plt.tight_layout()
plt.show()
