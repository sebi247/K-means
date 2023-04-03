import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def k_means_clustering(k, D):
    def initialize_centroids(k, D):
        return D[np.random.choice(D.shape[0], k, replace=False)]

    def assign_clusters(D, centroids):
        distances = np.linalg.norm(D[:, np.newaxis] - centroids, axis=2) ** 2
        return np.argmin(distances, axis=1)

    def update_centroids(D, assignments, k):
        return np.array([D[assignments == i].mean(axis=0) for i in range(k)])

    def has_converged(previous_centroids, centroids, tolerance=1e-4):
        return np.linalg.norm(previous_centroids - centroids) < tolerance

    centroids = initialize_centroids(k, D)
    previous_centroids = None

    while True:
        assignments = assign_clusters(D, centroids)
  
        previous_centroids = centroids.copy()
        centroids = update_centroids(D, assignments, k)

        if has_converged(previous_centroids, centroids):
            break

    return assignments, centroids


def bisecting_kmeans(dataset, k):
    # Initialize tree T to contain a single vertex(root) with entire dataset
    tree = [dataset]
    
    # Repeat until the number of leaf clusters is k
    while len(tree) < k:
        # Select a leaf node L in T that has the largest sum of square distance
        max_dist = -1
        for i in range(len(tree)):
            if isinstance(tree[i], np.ndarray):
                dist = np.sum(np.square(tree[i] - np.mean(tree[i], axis=0)))
                if dist > max_dist:
                    max_dist = dist
                    leaf_index = i
        
        # Split L into 2 clusters L1, L2 using k-means algorithm
        L = tree.pop(leaf_index)
        assignments, centroids = k_means_clustering(2, L)
        L1 = L[assignments == 0]
        L2 = L[assignments == 1]
        
        # Add L1, L2 as children in T
        tree.append(L1)
        tree.append(L2)
        
    # Return the leaf clusters
    return tree


def k_means_plus_plus_init(k, D):
    centroids = np.empty((k, D.shape[1]))
    
    # Step 1: Randomly select one centroid from the dataset
    first_centroid = D[np.random.choice(D.shape[0], 1)]
    centroids[0] = first_centroid

    # Step 2: Select remaining centroids
    for i in range(1, k):
        distances = np.linalg.norm(D[:, np.newaxis] - centroids[:i], axis=2) ** 2
        min_distances = np.min(distances, axis=1)
        probabilities = min_distances / np.sum(min_distances)
        next_centroid = D[np.random.choice(D.shape[0], 1, p=probabilities)]
        centroids[i] = next_centroid

    return centroids

def k_means_plus_plus_clustering(k, D):
    # Step 1: Initialize centroids using k-means++ initialization algorithm
    centroids = k_means_plus_plus_init(k, D)
    
    # Step 2: Perform standard k-means clustering with the initialized centroids
    previous_centroids = None
    tolerance = 1e-4

    while True:
        # Assignment phase
        distances = np.linalg.norm(D[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)

        # Update the clusters according to the new assignments
        new_centroids = np.array([D[assignments == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if previous_centroids is not None and np.linalg.norm(new_centroids - previous_centroids) < tolerance:
            break

        previous_centroids = new_centroids
        centroids = new_centroids

    return assignments, centroids



# Read the data
def read_data(file_name):
    data = pd.read_csv(file_name, sep=" ", header=None, skiprows=1)
    data = data.drop(0, axis=1)
    data = data.dropna(axis=1)
    return data.values

# Compute Silhouette score
def silhouette_score(D, assignments, centroids):
    n_clusters = len(centroids)
    if n_clusters == 1:
        return 0
    
    silhouette_scores = []

    for i, x in enumerate(D):
        cluster_i = assignments[i]
        other_clusters = set(range(n_clusters)) - {cluster_i}
        
        a = np.mean([np.linalg.norm(x - D[j]) for j in range(len(D)) if assignments[j] == cluster_i])
        
        b = float('inf')
        for other_cluster in other_clusters:
            avg_distance = np.mean([np.linalg.norm(x - D[j]) for j in range(len(D)) if assignments[j] == other_cluster])
            b = min(b, avg_distance)
        
        silhouette_scores.append((b - a) / max(a, b))

    return np.mean(silhouette_scores)

# Run k-means clustering and compute Silhouette coefficient
def k_means_silhouette(data):
    k_values = range(1, 10)
    silhouette_scores = []

    for k in k_values:
        assignments, centroids = k_means_clustering(k, data)
        silhouette_avg = silhouette_score(data, assignments, centroids)
        silhouette_scores.append(silhouette_avg)

    return k_values, silhouette_scores

# Plot the results
def plot_silhouette(k_values, silhouette_scores):
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette coefficient")
    plt.title("Silhouette coefficient vs k")
    plt.show()

# Main function
def main():
    data = read_data('dataset')
    k_range = list(range(1, 10))

    silhouette_scores = {
        'k-means': [],
        'k-means++': [],
        'bisecting_kmeans': []
    }

    for k in k_range:
        # K-means clustering
        assignments, centroids = k_means_clustering(k, data)
        k_means_silhouette = silhouette_score(data, assignments, centroids)
        silhouette_scores['k-means'].append(k_means_silhouette)
        print(f'k-means (k={k}): {k_means_silhouette}')

        # K-means++ clustering
        assignments, centroids = k_means_plus_plus_clustering(k, data)
        k_means_pp_silhouette = silhouette_score(data, assignments, centroids)
        silhouette_scores['k-means++'].append(k_means_pp_silhouette)
        print(f'k-means++ (k={k}): {k_means_pp_silhouette}')

        # Bisecting K-means clustering
        clusters = bisecting_kmeans(data, k)
        assignments = np.zeros(data.shape[0])
        for idx, cluster in enumerate(clusters):
            for i, point in enumerate(data):
                if np.any(np.all(cluster == point, axis=1)):
                    assignments[i] = idx
        centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        bisecting_kmeans_silhouette = silhouette_score(data, assignments, centroids)
        silhouette_scores['bisecting_kmeans'].append(bisecting_kmeans_silhouette)
        print(f'Bisecting k-means (s={k}): {bisecting_kmeans_silhouette}')

    # Plot results
    for algorithm, label in [('k-means', 'k-means'), ('k-means++', 'k-means++'), ('bisecting_kmeans', 'Bisecting k-means')]:
        plt.plot(k_range, silhouette_scores[algorithm], label=label)
        plt.xlabel('k / s')
        plt.ylabel('Silhouette coefficient')
        plt.legend()
        plt.title(f'{label} Silhouette coefficient')
        plt.show()

if __name__ == '__main__':
    main()

