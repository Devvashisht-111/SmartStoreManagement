#Loading the data
import pandas as pd
df = pd.read_csv('/content/sales_data.csv')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[['Price', 'Sales_quantity']])
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(scaled_df)
df['Cluster'] = kmedoids.labels_
plt.figure(figsize=(8, 6))
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Standardized Price')
plt.ylabel('Standardized Sales Quantity')
plt.title('K-Medoids Clustering (k=3)')
plt.colorbar(label='Cluster')
plt.show()
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(scaled_df, df['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")
