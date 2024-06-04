import pandas as pd
import numpy as np
import random
import datetime

np.random.seed(0)
random.seed(0)

Categories = ['Electronic' ,'Home Appliances' , 'Clothing ', 'Home Decor']
Brands = ['Samsung','Apple','LG','Nike','Sony', 'Adidas','IKEA','Google','Whirlpool','Home Depot']
Seasons = ['Summer','Winter','Spring','Fall']
Data = []
Starting_date = datetime.datetime(2020,1,1)

for i in range(200):
    Product_ID = i + 1
    Category = random.choice(Categories)
    Brand = random.choice(Brands)
    Price = round(random.uniform(10, 200), 2)
    Sales_quantity = random.randint(0, 50)
    Season = random.choice(Seasons)  # Choose a single season randomly
    date = Starting_date + datetime.timedelta(days=i)
    Data.append([Product_ID, Category, Brand, Price, Sales_quantity, Season, date])

df = pd.DataFrame(Data, columns=['Product_ID', 'Category', 'Brand', 'Price', 'Sales_quantity', 'Season', 'date'])
df.to_csv('sales_data.csv', index=False)
#Loading the data
import pandas as pd
df = pd.read_csv('/content/sales_data.csv')
#Data points before applying
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
%matplotlib inline
plt.scatter(df['Price'],df['Sales_quantity'])
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Feature scaling
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[['Price', 'Sales_quantity']])

# Fit K-Means algorithm (choose an appropriate value for k)
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(scaled_df)

# Assign cluster labels to the data
df['Cluster'] = kmeans.labels_

# Scatter plot to visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Standardized Price')
plt.ylabel('Standardized Sales Quantity')
plt.title('K-Means Clustering (k=3)')
plt.colorbar(label='Cluster')
plt.show()
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(scaled_df, df['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('sales_data.csv')

df['date'] = pd.to_datetime(df['date'])

df.set_index('date', inplace=True)

# Line chart for sales trends over time
df.resample('M').sum()['Sales_quantity'].plot(kind='line')
plt.title('Monthly Sales Trends')
plt.ylabel('Total Sales Quantity')
plt.xlabel('Date')
plt.show()

# Correlation heatmap
corr = df[['Price', 'Sales_quantity']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
