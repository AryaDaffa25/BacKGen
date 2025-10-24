import pickle
import json
import numpy as np
import os
import argparse
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from tqdm import tqdm

# Import BacKGen utilities  
from converter.io import read_jsonl, write_jsonl

class SBERTClusterer:
    """
    Class untuk melakukan K-means clustering pada SBERT embeddings
    mengikuti approach BacKGen dengan k = data_count / 5
    """
    
    def __init__(self, random_state=42):
        """
        Initialize clusterer
        
        Args:
            random_state: Random state untuk reproducibility
        """
        self.random_state = random_state
        self.kmeans_models = {}
        
    def load_data_from_jsonl(self, jsonl_path):
        """
        Load data dari file JSONL yang sudah berisi embeddings
        
        Args:
            jsonl_path: Path ke file .jsonl yang berisi text dan embeddings
            
        Returns:
            tuple: (embeddings, text_data)
        """
        text_data = []
        embeddings = []
        
        print(f"Loading data from: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract embeddings (assume key 'embedding' or 'embeddings')
                    if 'embedding' in data:
                        embedding = np.array(data['embedding'])
                    elif 'embeddings' in data:
                        embedding = np.array(data['embeddings'])
                    else:
                        raise KeyError("No 'embedding' or 'embeddings' key found in data")
                    
                    embeddings.append(embedding)
                    
                    # Create text data structure
                    text_item = {
                        'text': data.get('text', ''),
                        'metadata': {k: v for k, v in data.items() if k not in ['embedding', 'embeddings']}
                    }
                    text_data.append(text_item)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing line {line_num + 1}: {e}")
                    continue
        
        embeddings = np.array(embeddings)
        
        print(f"Loaded embeddings shape: {embeddings.shape}")
        print(f"Loaded text data: {len(text_data)} entries")
        
        return embeddings, text_data
        
    def load_embeddings_and_texts(self, pkl_path, jsonl_path):
        """
        Load embeddings dan text data (untuk backward compatibility)
        
        Args:
            pkl_path: Path ke file .pkl embeddings
            jsonl_path: Path ke file .jsonl text data
            
        Returns:
            tuple: (embeddings, text_data)
        """
        # Load embeddings
        with open(pkl_path, 'rb') as f:
            embedding_data = pickle.load(f)
        
        embeddings = embedding_data['embeddings']
        
        # Load text data
        text_data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                text_item = {
                    'text': data.get('text', ''),
                    'metadata': {k: v for k, v in data.items() if k != 'text'}
                }
                text_data.append(text_item)
        
        print(f"Loaded embeddings shape: {embeddings.shape}")
        print(f"Loaded text data: {len(text_data)} entries")
        
        # Verify consistency
        if len(embeddings) != len(text_data):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(text_data)} texts")
        
        return embeddings, text_data
    
    def calculate_optimal_k(self, embeddings, min_k=2, max_k=None):
        """
        Calculate optimal k menggunakan BacKGen approach (data_count / 5)
        dengan validasi menggunakan silhouette score
        
        Args:
            embeddings: numpy array of embeddings
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters (default: min(50, data_count/3))
            
        Returns:
            tuple: (recommended_k, silhouette_scores)
        """
        data_count = len(embeddings)
        
        # BacKGen recommendation: k = data_count / 5
        backgen_k = max(min_k, data_count // 5)
        
        if max_k is None:
            max_k = min(50, data_count // 3)
        
        print(f"Data count: {data_count}")
        print(f"BacKGen recommended k: {backgen_k}")
        print(f"Evaluating k range: {min_k} to {max_k}")
        
        # Evaluate different k values
        silhouette_scores = {}
        k_range = range(min_k, min(max_k + 1, data_count))
        
        for k in tqdm(k_range, desc="Evaluating k values"):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores[k] = score
        
        # Find best k by silhouette score
        best_k = max(silhouette_scores.keys(), key=lambda k: silhouette_scores[k])
        
        print(f"Best k by silhouette score: {best_k} (score: {silhouette_scores[best_k]:.3f})")
        print(f"BacKGen k: {backgen_k} (score: {silhouette_scores.get(backgen_k, 'N/A')})")
        
        return backgen_k, silhouette_scores
    
    def perform_clustering(self, embeddings, k):
        """
        Perform K-means clustering
        
        Args:
            embeddings: numpy array of embeddings
            k: number of clusters
            
        Returns:
            tuple: (kmeans_model, cluster_labels, cluster_centers)
        """
        print(f"Performing K-means clustering with k={k}")
        
        kmeans = KMeans(
            n_clusters=k, 
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate clustering metrics
        inertia = kmeans.inertia_
        if k > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            print(f"Clustering completed:")
            print(f"  Inertia: {inertia:.2f}")
            print(f"  Silhouette Score: {silhouette_avg:.3f}")
        else:
            print(f"Clustering completed with inertia: {inertia:.2f}")
        
        return kmeans, cluster_labels, kmeans.cluster_centers_
    
    def find_medoid(self, embeddings, cluster_indices, cluster_center):
        """
        Find medoid (text terdekat ke centroid) dalam cluster
        
        Args:
            embeddings: All embeddings array
            cluster_indices: Indices of texts in this cluster
            cluster_center: Center of the cluster
            
        Returns:
            int: Index of medoid in the original embeddings array
        """
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate distances from each point to cluster center
        distances = euclidean_distances(cluster_embeddings, cluster_center.reshape(1, -1))
        
        # Find index of minimum distance
        medoid_local_idx = np.argmin(distances)
        medoid_global_idx = cluster_indices[medoid_local_idx]
        
        return medoid_global_idx
    
    def analyze_clusters_with_medoid(self, embeddings, cluster_labels, cluster_centers, text_data, 
                                    drop_singleton=True):
        """
        Analyze cluster distribution dengan medoid calculation
        
        Args:
            embeddings: numpy array of embeddings
            cluster_labels: Array of cluster assignments
            cluster_centers: Cluster centers from KMeans
            text_data: List of text data dictionaries
            drop_singleton: Boolean, if True drop clusters with size 1
            
        Returns:
            dict: Cluster analysis results with medoid info
        """
        unique_clusters = np.unique(cluster_labels)
        cluster_analysis = {}
        
        # ===== BARIS BARU: TRACKING SINGLETON CLUSTERS =====
        singleton_clusters = []
        dropped_texts = []
        # ===================================================
        
        print(f"\n=== CLUSTER ANALYSIS WITH MEDOID ===")
        print(f"Number of clusters (before filtering): {len(unique_clusters)}")
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # ===== BARIS BARU: CHECK IF SINGLETON =====
            if drop_singleton and cluster_size == 1:
                singleton_clusters.append(cluster_id)
                dropped_texts.append(text_data[cluster_indices[0]]['text'])
                print(f"Cluster {cluster_id}: DROPPED (singleton - only 1 text)")
                print(f"  Text: {text_data[cluster_indices[0]]['text'][:100]}...")
                continue  # Skip cluster ini, tidak masuk ke cluster_analysis
            # =========================================
            
            # Find medoid for this cluster
            medoid_idx = self.find_medoid(embeddings, cluster_indices, cluster_centers[cluster_id])
            
            # Get texts in this cluster
            cluster_texts = [text_data[i]['text'] for i in cluster_indices]
            cluster_metadata = [text_data[i]['metadata'] for i in cluster_indices]
            
            # Get medoid info
            medoid_text = text_data[medoid_idx]['text']
            medoid_metadata = text_data[medoid_idx]['metadata']
            
            cluster_analysis[int(cluster_id)] = {
                'size': cluster_size,
                'percentage': cluster_size / len(cluster_labels) * 100,
                'medoid_index': int(medoid_idx),
                'medoid_text': medoid_text,
                'medoid_metadata': medoid_metadata,
                'texts': cluster_texts,
                'metadata': cluster_metadata,
                'indices': cluster_indices.tolist()
            }
            
            print(f"Cluster {cluster_id}: {cluster_size} texts ({cluster_size/len(cluster_labels)*100:.1f}%)")
            print(f"  Medoid: {medoid_text[:80]}...")
            
            # Show sample texts
            sample_texts = cluster_texts[:3]
            for i, text in enumerate(sample_texts):
                print(f"  Sample {i+1}: {text[:80]}...")
            
            if len(cluster_texts) > 3:
                print(f"  ... and {len(cluster_texts)-3} more texts")
            print()
        
        # ===== BARIS BARU: SUMMARY AFTER FILTERING =====
        print(f"\n=== FILTERING SUMMARY ===")
        print(f"Total singleton clusters dropped: {len(singleton_clusters)}")
        print(f"Remaining clusters: {len(cluster_analysis)}")
        if singleton_clusters:
            print(f"Dropped cluster IDs: {singleton_clusters}")
        # ===============================================
        
        return cluster_analysis
    
    def save_clustering_results(self, cluster_labels, cluster_analysis, text_data, 
                           input_file, output_dir, polarity):
        """
        Save clustering results dalam format JSON dengan medoid dan list_text_source
        (Hanya menyimpan cluster yang ada di cluster_analysis - singleton sudah di-drop)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Ambil nama file input tanpa ekstensi
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Tentukan nama output
        output_file = os.path.join(output_dir, f"{base_name}_clusters.jsonl")
        summary_file = os.path.join(output_dir, f"{base_name}_summary.json")
        
        # Format output seperti contoh: satu line per cluster
        cluster_results = []
        for cluster_id in sorted(cluster_analysis.keys()):
            analysis = cluster_analysis[cluster_id]
            
            cluster_entry = {
                'cluster_id': f"{polarity}_cluster_{cluster_id}",
                'polarity_label': polarity,
                'medoid_text': analysis['medoid_text'],
                'medoid_metadata': analysis['medoid_metadata'],
                'list_text_source': analysis['texts'],
                'cluster_size': analysis['size'],
                'cluster_percentage': round(analysis['percentage'], 2)
            }
            cluster_results.append(cluster_entry)
        
        # Save as JSONL (one cluster per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for cluster_entry in cluster_results:
                f.write(json.dumps(cluster_entry, ensure_ascii=False) + '\n')
        
        # Save summary
        cluster_summary = {
            'total_texts': len(text_data),
            'num_clusters': len(cluster_analysis),
            'polarity': polarity,
            'clusters': {}
        }
        for cluster_id, analysis in cluster_analysis.items():
            cluster_summary['clusters'][cluster_id] = {
                'size': analysis['size'],
                'percentage': analysis['percentage'],
                'medoid_preview': analysis['medoid_text'][:100] + '...',
                'sample_texts': [t[:80] + '...' for t in analysis['texts'][:3]]
            }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Clustering results saved to: {output_file}")
        print(f"Cluster summary saved to: {summary_file}")
        
        return output_file, summary_file


def cluster_polarity_data(positive_pkl, positive_jsonl, negative_pkl, negative_jsonl, 
                         output_dir, evaluate_k=False, drop_singleton=True):
    """
    Main function untuk clustering both positive dan negative data
    
    Args:
        positive_pkl: Path ke positive embeddings (.pkl)
        positive_jsonl: Path ke positive text data (.jsonl)
        negative_pkl: Path ke negative embeddings (.pkl) 
        negative_jsonl: Path ke negative text data (.jsonl)
        output_dir: Output directory
        evaluate_k: Whether to evaluate different k values
        drop_singleton: Whether to drop singleton clusters (default: True)
        
    Returns:
        dict: Clustering results for both polarities
    """
    start_time_total = time.time()
    clusterer = SBERTClusterer()
    results = {}
    
    # Process positive data
    print("=== CLUSTERING POSITIVE DATA ===")
    start_time_pos = time.time()
    pos_embeddings, pos_texts = clusterer.load_embeddings_and_texts(positive_pkl, positive_jsonl)
    
    if evaluate_k:
        pos_k, pos_silhouette = clusterer.calculate_optimal_k(pos_embeddings)
    else:
        pos_k = max(2, len(pos_embeddings) // 5)  # BacKGen approach
        print(f"Using BacKGen k={pos_k} for positive data ({len(pos_embeddings)} texts)")
    
    pos_kmeans, pos_labels, pos_centers = clusterer.perform_clustering(pos_embeddings, pos_k)
    
    # ===== BARIS BARU: PASS drop_singleton PARAMETER =====
    pos_analysis = clusterer.analyze_clusters_with_medoid(
        pos_embeddings, pos_labels, pos_centers, pos_texts, 
        drop_singleton=drop_singleton
    )
    # =====================================================
    
    # Save positive results
    clusterer.save_clustering_results(pos_labels, pos_analysis, pos_texts, 
                                     positive_jsonl, output_dir, 'positive')
    end_time_pos = time.time()
    print(f"\n🕒 Waktu clustering positif: {end_time_pos - start_time_pos:.2f} detik")
    
    results['positive'] = {
        'k': pos_k,
        'labels': pos_labels,
        'analysis': pos_analysis
    }
    
    # Process negative data
    print("\n=== CLUSTERING NEGATIVE DATA ===")
    start_time_neg = time.time()
    neg_embeddings, neg_texts = clusterer.load_embeddings_and_texts(negative_pkl, negative_jsonl)
    
    if evaluate_k:
        neg_k, neg_silhouette = clusterer.calculate_optimal_k(neg_embeddings)
    else:
        neg_k = max(2, len(neg_embeddings) // 5)  # BacKGen approach
        print(f"Using BacKGen k={neg_k} for negative data ({len(neg_embeddings)} texts)")
    
    neg_kmeans, neg_labels, neg_centers = clusterer.perform_clustering(neg_embeddings, neg_k)
    
    # ===== BARIS BARU: PASS drop_singleton PARAMETER =====
    neg_analysis = clusterer.analyze_clusters_with_medoid(
        neg_embeddings, neg_labels, neg_centers, neg_texts,
        drop_singleton=drop_singleton
    )
    # =====================================================
    
    # Save negative results
    clusterer.save_clustering_results(neg_labels, neg_analysis, neg_texts,
                                     negative_jsonl, output_dir, 'negative')
    end_time_neg = time.time()
    print(f"\n🕒 Waktu clustering negatif: {end_time_neg - start_time_neg:.2f} detik")
    
    results['negative'] = {
        'k': neg_k,
        'labels': neg_labels,
        'analysis': neg_analysis
    }
    total_time = time.time() - start_time_total
    print(f"\n⏱️ Total runtime clustering: {total_time:.2f} detik")
    return results

def main():
    """
    Main function dengan argument parsing
    """
    parser = argparse.ArgumentParser(
        description='SBERT K-means Clustering with Medoid for BacKGen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: drop singleton clusters
  python clustering_with_medoid_no_singleton.py \
    --positive_jsonl data/embeddings/train_positive_embeddings.jsonl \
    --positive_pkl data/embeddings/train_positive_embeddings.pkl \
    --negative_jsonl data/embeddings/train_negative_embeddings.jsonl \
    --negative_pkl data/embeddings/train_negative_embeddings.pkl \
    --output_base data/clustered
  
  # Keep singleton clusters
  python clustering_with_medoid_no_singleton.py \
    --positive_jsonl data/embeddings/train_positive_embeddings.jsonl \
    --positive_pkl data/embeddings/train_positive_embeddings.pkl \
    --negative_jsonl data/embeddings/train_negative_embeddings.jsonl \
    --negative_pkl data/embeddings/train_negative_embeddings.pkl \
    --output_base data/clustered \
    --keep_singleton
        """
    )
    
    parser.add_argument('--output_base', type=str, required=True,
                        help='Base path for output directory')
    parser.add_argument('--evaluate_k', action='store_true',
                        help='Evaluate different k values using silhouette score')
    parser.add_argument('--legacy_mode', action='store_true',
                        help='Use legacy mode with default positive/negative files')
    parser.add_argument("--positive_jsonl", type=str, help="Path to positive .jsonl file")
    parser.add_argument("--positive_pkl", type=str, help="Path to positive .pkl embeddings file")
    parser.add_argument("--negative_jsonl", type=str, help="Path to negative .jsonl file")
    parser.add_argument("--negative_pkl", type=str, help="Path to negative .pkl embeddings file")
    
    # ===== BARIS BARU: ARGUMENT UNTUK DROP SINGLETON =====
    parser.add_argument('--keep_singleton', action='store_true',
                        help='Keep singleton clusters (default: drop them)')
    # ====================================================

    args = parser.parse_args()
    
    # ===== BARIS BARU: SET drop_singleton =====
    drop_singleton = not args.keep_singleton
    # =========================================
    
    try:
        print("=== SBERT K-MEANS CLUSTERING WITH MEDOID ===")
        print(f"Singleton clusters will be: {'KEPT' if not drop_singleton else 'DROPPED'}")
        
        if args.legacy_mode:
            # Mode 1: Legacy
            positive_pkl = "data/embeddings/positive_embeddings.pkl"
            positive_jsonl = "data/embeddings/positive_embeddings.jsonl"
            negative_pkl = "data/embeddings/negative_embeddings.pkl"
            negative_jsonl = "data/embeddings/negative_embeddings.jsonl"
            
            print("Running in LEGACY mode...")
            results = cluster_polarity_data(
                positive_pkl=positive_pkl,
                positive_jsonl=positive_jsonl,
                negative_pkl=negative_pkl,
                negative_jsonl=negative_jsonl,
                output_dir=args.output_base,
                evaluate_k=args.evaluate_k,
                drop_singleton=drop_singleton  # Pass parameter
            )
        
        elif args.positive_jsonl and args.positive_pkl and args.negative_jsonl and args.negative_pkl:
            # Mode 2: Custom polarity
            print("Running in CUSTOM POLARITY mode...")
            results = cluster_polarity_data(
                positive_pkl=args.positive_pkl,
                positive_jsonl=args.positive_jsonl,
                negative_pkl=args.negative_pkl,
                negative_jsonl=args.negative_jsonl,
                output_dir=args.output_base,
                evaluate_k=args.evaluate_k,
                drop_singleton=drop_singleton  # Pass parameter
            )
        
        else:
            parser.error("Please provide one of the following modes:\n"
                         "  --legacy_mode, OR\n"
                         "  --positive_jsonl --positive_pkl --negative_jsonl --negative_pkl")
        
        print(f"\n=== CLUSTERING COMPLETED ===")
        print(f"Positive clusters: {results['positive']['k']} initial clusters")
        print(f"  -> {len(results['positive']['analysis'])} clusters after filtering")
        print(f"Negative clusters: {results['negative']['k']} initial clusters")
        print(f"  -> {len(results['negative']['analysis'])} clusters after filtering")

        print(f"\nOutput format:")
        print(f"- Each line in output JSONL = 1 cluster")
        print(f"- 'medoid_text' = representative text (closest to centroid)")
        print(f"- 'list_text_source' = all texts in cluster")
        if drop_singleton:
            print(f"- Singleton clusters (size=1) have been DROPPED")
        else:
            print(f"- Singleton clusters (size=1) are KEPT")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the input file path or run the embedding generation first!")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages: pip install scikit-learn")
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()