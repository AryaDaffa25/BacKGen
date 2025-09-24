import pickle
import json
import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
                text_data.append(json.loads(line.strip()))
        
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
    
    def analyze_clusters(self, cluster_labels, text_data):
        """
        Analyze cluster distribution dan content
        
        Args:
            cluster_labels: Array of cluster assignments
            text_data: List of text data dictionaries
            
        Returns:
            dict: Cluster analysis results
        """
        unique_clusters = np.unique(cluster_labels)
        cluster_analysis = {}
        
        print(f"\n=== CLUSTER ANALYSIS ===")
        print(f"Number of clusters: {len(unique_clusters)}")
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # Get texts in this cluster
            cluster_texts = [text_data[i]['text'] for i in cluster_indices]
            cluster_metadata = [text_data[i]['metadata'] for i in cluster_indices]
            
            cluster_analysis[int(cluster_id)] = {
                'size': cluster_size,
                'percentage': cluster_size / len(cluster_labels) * 100,
                'texts': cluster_texts,
                'metadata': cluster_metadata,
                'indices': cluster_indices.tolist()
            }
            
            print(f"Cluster {cluster_id}: {cluster_size} texts ({cluster_size/len(cluster_labels)*100:.1f}%)")
            
            # Show sample texts
            sample_texts = cluster_texts[:3]
            for i, text in enumerate(sample_texts):
                print(f"  Sample {i+1}: {text[:80]}...")
            
            if len(cluster_texts) > 3:
                print(f"  ... and {len(cluster_texts)-3} more texts")
            print()
        
        return cluster_analysis
    
    def save_clustering_results(self, cluster_labels, cluster_analysis, text_data, 
                           input_file, output_dir, polarity):
        """
        Save clustering results dalam format yang compatible dengan BacKGen
        dan nama file output menyesuaikan input file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Ambil nama file input tanpa ekstensi
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Tentukan nama output
        output_file = os.path.join(output_dir, f"{base_name}.jsonl")
        summary_file = os.path.join(output_dir, f"{base_name}_summary.json")
        
        clustering_results = []
        for i, (cluster_id, text_item) in enumerate(zip(cluster_labels, text_data)):
            result_item = {
                'index': i,
                'cluster_id': int(cluster_id),
                'text': text_item['text'],
                'metadata': text_item['metadata'],
                'polarity': polarity
            }
            clustering_results.append(result_item)
        
        # Save clustering results
        write_jsonl(clustering_results, output_file)
        
        # Save summary
        cluster_summary = {
            'total_texts': len(text_data),
            'num_clusters': len(np.unique(cluster_labels)),
            'polarity': polarity,
            'clusters': {}
        }
        for cluster_id, analysis in cluster_analysis.items():
            cluster_summary['clusters'][cluster_id] = {
                'size': analysis['size'],
                'percentage': analysis['percentage'],
                'sample_texts': analysis['texts'][:5]
            }
        with open(summary_file, 'w') as f:
            json.dump(cluster_summary, f, indent=2)
        
        print(f"Clustering results saved to: {output_file}")
        print(f"Cluster summary saved to: {summary_file}")
        
        return output_file, summary_file


# def cluster_single_file(input_file, output_base, evaluate_k=False):
#     """
#     Main function untuk clustering single input file
    
#     Args:
#         input_file: Path ke input file (.jsonl dengan embeddings)
#         output_base: Base path untuk output directory
#         evaluate_k: Whether to evaluate different k values
        
#     Returns:
#         dict: Clustering results
#     """
#     clusterer = SBERTClusterer()
    
#     print(f"=== CLUSTERING DATA FROM {input_file} ===")
    
#     # Load data
#     embeddings, text_data = clusterer.load_data_from_jsonl(input_file)
    
#     # Calculate k
#     if evaluate_k:
#         k, silhouette_scores = clusterer.calculate_optimal_k(embeddings)
#     else:
#         k = max(2, len(embeddings) // 5)  # BacKGen approach
#         print(f"Using BacKGen k={k} for data ({len(embeddings)} texts)")
    
#     # Perform clustering
#     kmeans_model, cluster_labels, cluster_centers = clusterer.perform_clustering(embeddings, k)
#     cluster_analysis = clusterer.analyze_clusters(cluster_labels, text_data)
    
#     # Prepare output path
#     input_basename = os.path.splitext(os.path.basename(input_file))[0]
#     output_file = os.path.join(output_base, f"{input_basename}_clustered.jsonl")
    
#     # Save results
#     clusterer.save_clustering_results(cluster_labels, cluster_analysis, text_data, output_file)
    
#     results = {
#         'k': k,
#         'labels': cluster_labels,
#         'analysis': cluster_analysis,
#         'output_file': output_file
#     }
    
#     return results

def cluster_polarity_data(positive_pkl, positive_jsonl, negative_pkl, negative_jsonl, 
                         output_dir, evaluate_k=False):
    """
    Main function untuk clustering both positive dan negative data (backward compatibility)
    
    Args:
        positive_pkl: Path ke positive embeddings (.pkl)
        positive_jsonl: Path ke positive text data (.jsonl)
        negative_pkl: Path ke negative embeddings (.pkl) 
        negative_jsonl: Path ke negative text data (.jsonl)
        output_dir: Output directory
        evaluate_k: Whether to evaluate different k values
        
    Returns:
        dict: Clustering results for both polarities
    """
    clusterer = SBERTClusterer()
    results = {}
    
    # Process positive data
    print("=== CLUSTERING POSITIVE DATA ===")
    pos_embeddings, pos_texts = clusterer.load_embeddings_and_texts(positive_pkl, positive_jsonl)
    
    if evaluate_k:
        pos_k, pos_silhouette = clusterer.calculate_optimal_k(pos_embeddings)
    else:
        pos_k = max(2, len(pos_embeddings) // 5)  # BacKGen approach
        print(f"Using BacKGen k={pos_k} for positive data ({len(pos_embeddings)} texts)")
    
    pos_kmeans, pos_labels, pos_centers = clusterer.perform_clustering(pos_embeddings, pos_k)
    pos_analysis = clusterer.analyze_clusters(pos_labels, pos_texts)
    
    # Save positive results
    pos_base = os.path.splitext(os.path.basename(positive_jsonl))[0]
    pos_output = os.path.join(output_dir, f"{pos_base}_clustered.jsonl")
    clusterer.save_clustering_results(pos_labels, pos_analysis, pos_texts, pos_output, output_dir, 'positive')
    
    results['positive'] = {
        'k': pos_k,
        'labels': pos_labels,
        'analysis': pos_analysis,
        'output_file': pos_output
    }
    
    # Process negative data
    print("\n=== CLUSTERING NEGATIVE DATA ===")
    neg_embeddings, neg_texts = clusterer.load_embeddings_and_texts(negative_pkl, negative_jsonl)
    
    if evaluate_k:
        neg_k, neg_silhouette = clusterer.calculate_optimal_k(neg_embeddings)
    else:
        neg_k = max(2, len(neg_embeddings) // 5)  # BacKGen approach
        print(f"Using BacKGen k={neg_k} for negative data ({len(neg_embeddings)} texts)")
    
    neg_kmeans, neg_labels, neg_centers = clusterer.perform_clustering(neg_embeddings, neg_k)
    neg_analysis = clusterer.analyze_clusters(neg_labels, neg_texts)
    
    # Save negative results
    neg_base = os.path.splitext(os.path.basename(negative_jsonl))[0]
    neg_output = os.path.join(output_dir, f"{neg_base}_clustered.jsonl")
    clusterer.save_clustering_results(neg_labels, neg_analysis, neg_texts, neg_output, output_dir, 'negative')
    
    results['negative'] = {
        'k': neg_k,
        'labels': neg_labels,
        'analysis': neg_analysis,
        'output_file': neg_output
    }
    
    return results

def main():
    """
    Main function dengan argument parsing
    """
    parser = argparse.ArgumentParser(
        description='SBERT K-means Clustering for BacKGen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file clustering
  python clustering.py --input_file data/input.jsonl --output_base data
  
  # Legacy mode (positive/negative files dengan nama default)
  python clustering.py --legacy_mode --output_base data/clusters
  
  # Custom polarity mode
  python clustering.py \
    --positive_jsonl data/embeddings/train_positive_embeddings.jsonl \
    --positive_pkl data/embeddings/train_positive_embeddings.pkl \
    --negative_jsonl data/embeddings/train_negative_embeddings.jsonl \
    --negative_pkl data/embeddings/train_negative_embeddings.pkl \
    --output_base data/clustered/fold_1/test
        """
    )
    
    parser.add_argument('--input_file', type=str, 
                        help='Path to input JSONL file containing text and embeddings')
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

    args = parser.parse_args()
    
    try:
        print("=== SBERT K-MEANS CLUSTERING FOR BACKGEN ===")
        
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
                evaluate_k=args.evaluate_k
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
                evaluate_k=args.evaluate_k
            )
        
        # elif args.input_file:
        #     # Mode 3: Single file
        #     print("Running in SINGLE FILE mode...")
        #     results = cluster_single_file(
        #         input_file=args.input_file,
        #         output_base=args.output_base,
        #         evaluate_k=args.evaluate_k
        #     )
        
        else:
            parser.error("Please provide one of the following modes:\n"
                         "  --legacy_mode, OR\n"
                         "  --positive_jsonl --positive_pkl --negative_jsonl --negative_pkl, OR\n"
                         "  --input_file")
        
        print(f"\n=== CLUSTERING COMPLETED ===")
        if 'positive' in results:
            print(f"Positive clusters: {results['positive']['k']} clusters")
            print(f"  -> {results['positive']['output_file']}")
        if 'negative' in results:
            print(f"Negative clusters: {results['negative']['k']} clusters") 
            print(f"  -> {results['negative']['output_file']}")
        # if 'k' in results:  # single file
        #     print(f"Number of clusters: {results['k']}")
        #     print(f"Output file: {results['output_file']}")

        print(f"\nNext steps for BacKGen:")
        print(f"1. Use clustered data for Background Knowledge generation")
        print(f"2. Each cluster represents semantically similar texts")
        print(f"3. Generate BK for each cluster separately")
        
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
