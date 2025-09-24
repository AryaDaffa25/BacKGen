# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from converter.io import read_jsonl, write_jsonl
from sklearn.cluster import KMeans
import numpy as np
import pickle
import pandas as pd
from typing import List, Dict, Any, Tuple

class EmbeddingClusterer:
    """
    Class untuk melakukan clustering pada embeddings menggunakan k-Means
    Menggantikan frame clustering pada BacKGen original
    """
    
    def __init__(self, n_clusters: int, random_state: int = 42, max_iter: int = 300):
        """
        Initialize k-Means clusterer
        
        Args:
            n_clusters: jumlah cluster (k = jumlah data / 5)
            random_state: seed untuk reproducibility
            max_iter: maksimum iterasi k-Means
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        
    def fit_cluster(self, embeddings: np.ndarray, verbose: str = "yes") -> np.ndarray:
        """
        Melakukan k-Means clustering pada embeddings
        
        Args:
            embeddings: numpy array of embeddings
            verbose: menampilkan progress
            
        Returns:
            cluster labels untuk setiap embedding
        """
        if verbose == "yes":
            print(f"Starting k-Means clustering with k={self.n_clusters}")
            print(f"Embedding shape: {embeddings.shape}")
        
        # Initialize dan fit k-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=10
        )
        
        cluster_labels = self.kmeans.fit_predict(embeddings)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        if verbose == "yes":
            print(f"Clustering completed. Inertia: {self.kmeans.inertia_:.2f}")
            print(f"Unique clusters: {len(np.unique(cluster_labels))}")
        
        return cluster_labels
    
    def find_medoids(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                    data_list: List[Dict], verbose: str = "yes") -> Dict[int, Dict]:
        """
        Mencari medoid untuk setiap cluster
        Medoid adalah data point yang paling dekat dengan cluster center
        
        Args:
            embeddings: numpy array of embeddings
            cluster_labels: hasil clustering
            data_list: list of original data
            verbose: menampilkan progress
            
        Returns:
            dictionary mapping cluster_id -> medoid data
        """
        medoids = {}
        
        for cluster_id in range(self.n_clusters):
            # Ambil indices untuk cluster ini
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Ambil embeddings untuk cluster ini
            cluster_embeddings = embeddings[cluster_indices]
            cluster_center = self.cluster_centers_[cluster_id]
            
            # Hitung jarak ke cluster center
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            
            # Ambil index medoid (yang terdekat dengan center)
            medoid_idx_in_cluster = np.argmin(distances)
            medoid_idx_global = cluster_indices[medoid_idx_in_cluster]
            
            # Simpan medoid data
            medoid_data = data_list[medoid_idx_global].copy()
            medoid_data['cluster_id'] = cluster_id
            medoid_data['medoid_distance'] = distances[medoid_idx_in_cluster]
            medoid_data['cluster_size'] = len(cluster_indices)
            
            medoids[cluster_id] = medoid_data
            
            if verbose == "yes" and cluster_id % 50 == 0:
                print(f"Processed cluster {cluster_id}, size: {len(cluster_indices)}")
        
        return medoids
    
    def create_clustered_data(self, embeddings: np.ndarray, cluster_labels: np.ndarray,
                            data_list: List[Dict], verbose: str = "yes") -> List[Dict]:
        """
        Membuat data dengan informasi cluster
        
        Args:
            embeddings: numpy array of embeddings
            cluster_labels: hasil clustering
            data_list: list of original data
            verbose: menampilkan progress
            
        Returns:
            list of data dengan cluster information
        """
        clustered_data = []
        
        for i, (data, cluster_id) in enumerate(zip(data_list, cluster_labels)):
            clustered_entry = data.copy()
            clustered_entry['cluster_id'] = int(cluster_id)
            clustered_entry['data_index'] = i
            
            # Hitung jarak ke cluster center
            distance_to_center = np.linalg.norm(
                embeddings[i] - self.cluster_centers_[cluster_id]
            )
            clustered_entry['distance_to_center'] = float(distance_to_center)
            
            clustered_data.append(clustered_entry)
        
        if verbose == "yes":
            print(f"Created clustered data with {len(clustered_data)} entries")
        
        return clustered_data

def cluster_sentiment_embeddings(filtered_data_path: str, embeddings_path: str,
                                output_clustered_path: str, output_medoids_path: str,
                                polarity: str, k_ratio: float = 0.2,
                                verbose: str = "yes") -> Tuple[List[Dict], Dict[int, Dict]]:
    """
    Melakukan clustering pada sentiment embeddings
    Menggantikan langkah 3 pada BK Generation process
    
    Args:
        filtered_data_path: path ke filtered sentiment data
        embeddings_path: path ke embeddings pickle file
        output_clustered_path: path output untuk clustered data
        output_medoids_path: path output untuk medoids
        polarity: sentiment polarity ('positive' atau 'negative')
        k_ratio: rasio untuk menentukan k (default: 0.2 = 1/5)
        verbose: menampilkan progress
        
    Returns:
        tuple: (clustered_data, medoids)
    """
    # Load data
    filtered_data = read_jsonl(filtered_data_path)
    
    with open(embeddings_path, 'rb') as f:
        all_embeddings = pickle.load(f)
    
    if verbose == "yes":
        print(f"Processing {polarity} sentiment clustering")
        print(f"Filtered data count: {len(filtered_data)}")
        print(f"All embeddings shape: {all_embeddings.shape}")
    
    # Ambil embeddings yang sesuai dengan filtered data
    filtered_embeddings = []
    filtered_indices = []
    
    for data in filtered_data:
        # Ambil embedding dari data
        if 'embedding' in data:
            embedding = np.array(data['embedding'])
            filtered_embeddings.append(embedding)
        else:
            # Jika tidak ada embedding di data, skip
            print(f"Warning: No embedding found for data ID: {data.get('id')}")
            continue
    
    filtered_embeddings = np.array(filtered_embeddings)
    
    if len(filtered_embeddings) == 0:
        print(f"Error: No embeddings found for {polarity} data")
        return [], {}
    
    # Tentukan jumlah cluster: k = jumlah data / 5
    n_clusters = max(1, int(len(filtered_embeddings) * k_ratio))
    
    if verbose == "yes":
        print(f"Number of clusters (k): {n_clusters}")
        print(f"Filtered embeddings shape: {filtered_embeddings.shape}")
    
    # Lakukan clustering
    clusterer = EmbeddingClusterer(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_cluster(filtered_embeddings, verbose)
    
    # Buat clustered data
    clustered_data = clusterer.create_clustered_data(
        filtered_embeddings, cluster_labels, filtered_data, verbose
    )
    
    # Cari medoids
    medoids = clusterer.find_medoids(
        filtered_embeddings, cluster_labels, filtered_data, verbose
    )
    
    # Filter singleton clusters jika diperlukan
    cluster_sizes = {}
    for entry in clustered_data:
        cluster_id = entry['cluster_id']
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    
    # Remove singleton clusters
    non_singleton_data = [
        entry for entry in clustered_data 
        if cluster_sizes[entry['cluster_id']] > 1
    ]
    
    non_singleton_medoids = {
        cluster_id: medoid for cluster_id, medoid in medoids.items()
        if cluster_sizes[cluster_id] > 1
    }
    
    if verbose == "yes":
        print(f"Removed {len(clustered_data) - len(non_singleton_data)} singleton clusters")
        print(f"Final clusters: {len(non_singleton_medoids)}")
    
    # Save results
    write_jsonl(non_singleton_data, output_clustered_path)
    write_jsonl(list(non_singleton_medoids.values()), output_medoids_path)
    
    if verbose == "yes":
        print(f"Clustered data saved to: {output_clustered_path}")
        print(f"Medoids saved to: {output_medoids_path}")
    
    return non_singleton_data, non_singleton_medoids

def cluster_all_sentiments(positive_data_path: str, negative_data_path: str,
                          embeddings_path: str, output_folder: str,
                          k_ratio: float = 0.2, verbose: str = "yes"):
    """
    Cluster semua sentiment polarities
    
    Args:
        positive_data_path: path ke positive filtered data
        negative_data_path: path ke negative filtered data
        embeddings_path: path ke embeddings
        output_folder: folder output
        k_ratio: rasio untuk k
        verbose: menampilkan progress
    """
    if output_folder and not output_folder.endswith('/'):
        output_folder += '/'
    
    # Cluster positive sentiment
    pos_clustered_path = f"{output_folder}clustered_positive.jsonl"
    pos_medoids_path = f"{output_folder}medoids_positive.jsonl"
    
    pos_clustered, pos_medoids = cluster_sentiment_embeddings(
        positive_data_path, embeddings_path, pos_clustered_path, pos_medoids_path,
        'positive', k_ratio, verbose
    )
    
    # Cluster negative sentiment
    neg_clustered_path = f"{output_folder}clustered_negative.jsonl"
    neg_medoids_path = f"{output_folder}medoids_negative.jsonl"
    
    neg_clustered, neg_medoids = cluster_sentiment_embeddings(
        negative_data_path, embeddings_path, neg_clustered_path, neg_medoids_path,
        'negative', k_ratio, verbose
    )
    
    return {
        'positive': {'clustered': pos_clustered, 'medoids': pos_medoids},
        'negative': {'clustered': neg_clustered, 'medoids': neg_medoids}
    }