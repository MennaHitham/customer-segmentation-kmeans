# %% [markdown]
# # Customer Segmentation with K-Means (RFM)
# Dataset: Online Retail (UCI / Kaggle) -> file: Online Retail.xlsx or online_retail.csv
# Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

# %% 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

# %% 2. Load data
df = pd.read_excel("Online Retail.xlsx")   # or pd.read_csv("online_retail.csv", encoding="ISO-8859-1")
print(df.shape)
df.head()

# %% 3. First look
df.info()
print(df.describe(include="all").T)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicates:", df.duplicated().sum())

# %% 4. Data cleaning
df_clean = df.copy()

# 4.1 Customers without ID can't be segmented
df_clean = df_clean.dropna(subset=["CustomerID"])
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

# 4.2 Remove duplicates
df_clean = df_clean.drop_duplicates()

# 4.3 Remove cancelled invoices (InvoiceNo starts with "C")
df_clean["InvoiceNo"] = df_clean["InvoiceNo"].astype(str)
df_clean = df_clean[~df_clean["InvoiceNo"].str.startswith("C")]

# 4.4 Remove invalid quantities / prices
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

# 4.5 Fix types + create TotalPrice
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]

print("Before:", df.shape, "| After:", df_clean.shape)

# %% 5. EDA
# 5.1 Top countries
df_clean["Country"].value_counts().head(10).plot(kind="barh", title="Top 10 Countries by Transactions")
plt.show()

# 5.2 Monthly revenue
monthly = df_clean.set_index("InvoiceDate")["TotalPrice"].resample("M").sum()
monthly.plot(marker="o", title="Monthly Revenue")
plt.show()

# 5.3 Distributions (note the skew)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df_clean["Quantity"].clip(upper=df_clean["Quantity"].quantile(0.99)), ax=ax[0]).set_title("Quantity")
sns.histplot(df_clean["UnitPrice"].clip(upper=df_clean["UnitPrice"].quantile(0.99)), ax=ax[1]).set_title("UnitPrice")
sns.histplot(df_clean["TotalPrice"].clip(upper=df_clean["TotalPrice"].quantile(0.99)), ax=ax[2]).set_title("TotalPrice")
plt.tight_layout()
plt.show()

# 5.4 Top products
df_clean.groupby("Description")["Quantity"].sum().nlargest(10).plot(kind="barh", title="Top 10 Products")
plt.show()

# %% 6. Feature engineering: RFM
snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df_clean.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
)
print(rfm.describe())

# %% 7. RFM EDA
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(rfm.columns):
    sns.histplot(rfm[col], ax=ax[i], kde=True).set_title(col)
plt.tight_layout()
plt.show()

sns.heatmap(rfm.corr(), annot=True, cmap="coolwarm")
plt.title("RFM Correlation")
plt.show()

# %% 8. Outliers + skewness handling
# RFM is heavily right-skewed -> log transform (log1p) before K-Means
rfm_log = np.log1p(rfm)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(rfm_log.columns):
    sns.histplot(rfm_log[col], ax=ax[i], kde=True).set_title(f"log({col})")
plt.tight_layout()
plt.show()

# %% 9. Scaling
scaler = StandardScaler()
X = scaler.fit_transform(rfm_log)

# %% 10. Choose K (Elbow + Silhouette)
inertias, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(K_range, inertias, marker="o"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
ax[1].plot(K_range, sil_scores, marker="o"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("K")
plt.show()

# %% 11. Final model (change K based on the plots above)
K = 4
kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(X)
print("Silhouette:", silhouette_score(X, rfm["Cluster"]))

# %% 12. Visualize clusters with PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=rfm["Cluster"], palette="tab10", alpha=0.7)
plt.title("Customer Segments (PCA 2D)")
plt.show()

# %% 13. Cluster profiling (the most important part)
profile = rfm.groupby("Cluster").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
    Customers=("Recency", "size"),
).round(1)
profile["% of Customers"] = (profile["Customers"] / profile["Customers"].sum() * 100).round(1)
print(profile)

# Heatmap of relative cluster averages
rel = profile[["Recency", "Frequency", "Monetary"]] / rfm[["Recency", "Frequency", "Monetary"]].mean() - 1
sns.heatmap(rel, annot=True, cmap="RdYlGn", center=0)
plt.title("Cluster vs. Overall Average")
plt.show()

# %% 14. Name your segments (adjust after reading the profile table)
# Example logic:
#   Low Recency + High Frequency + High Monetary -> "Champions"
#   Low Recency + Low Frequency                  -> "New Customers"
#   High Recency + High Frequency (past)         -> "At Risk"
#   High Recency + Low Frequency + Low Monetary  -> "Lost"
# segment_names = {0: "Champions", 1: "At Risk", 2: "New Customers", 3: "Lost"}
# rfm["Segment"] = rfm["Cluster"].map(segment_names)

# %% 15. Save results
rfm.to_csv("customer_segments.csv")
