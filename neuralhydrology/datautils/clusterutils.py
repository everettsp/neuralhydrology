
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List
from pathlib import Path


# Function to cluster basin attributes and select representative basins
def cluster_basin_attributes(attributes, n, k, random_state=0, required_basins=[]):

    for basin in required_basins:
        if basin not in attributes.index:
            raise ValueError(f"Basin {basin} not found in the attributes DataFrame.")

    # Select only numeric columns for clustering
    numeric_columns = attributes.loc[:,:].select_dtypes(include=['float64']).columns
    data_for_clustering = attributes.loc[:, numeric_columns].dropna()

    # Standardize the data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    data_for_clustering['Cluster'] = kmeans.fit_predict(scaled_data)

    # Add cluster labels to the original dataframe
    attributes.loc[data_for_clustering.index, 'Cluster'] = data_for_clustering['Cluster']

    selected_basins = []
    print(data_for_clustering["Cluster"].value_counts())
    print(required_basins[0] in data_for_clustering.index)
    for cluster in range(k):
        cluster_basins = data_for_clustering[data_for_clustering['Cluster'] == cluster].index.tolist()

        # Always include required basins in this cluster
        required_in_cluster = [b for b in required_basins if b in cluster_basins]
        
        n_to_sample = max(0, n - len(required_in_cluster))
        print(n_to_sample)
        sampled = []
        if n_to_sample > 0:
            sampled = pd.Index(cluster_basins).difference(required_in_cluster).to_list()
            sampled = pd.Series(sampled).sample(n=n_to_sample, random_state=random_state, replace=False).tolist()
        selected_basins.extend(required_in_cluster + sampled)

    # Ensure all required basins are included
    for b in required_basins:
        if b not in selected_basins and b in data_for_clustering.index:
            selected_basins.append(b)

    return selected_basins, attributes