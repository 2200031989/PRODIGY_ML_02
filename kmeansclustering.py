import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the CSV file
file_path = 'Mall_Customers.csv'  # Replace this with your actual file path
try:
    df = pd.read_csv(file_path)
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Display the first few rows to ensure it loaded correctly
print(df.head())

# Step 2: Data Preprocessing
# Extract the features for clustering
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Step 3: Determine the optimal number of clusters using the elbow method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_normalized)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Choose the optimal number of clusters from the elbow plot
optimal_k = 5  # Adjust this based on your elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_normalized)

# Step 4: Evaluation
sil_score = silhouette_score(features_normalized, df['Cluster'])
print(f'Silhouette Score: {sil_score}')

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.show()

# Display the dataframe with cluster assignments
print(df.head())
