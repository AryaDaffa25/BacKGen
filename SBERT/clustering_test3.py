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
    
    def find_medoid(self, embeddings, cluster_indices):
        """
        Find medoid (most representative point) dalam cluster
        
        Args:
            embeddings: All embeddings
            cluster_indices: Indices of points in this cluster
            
        Returns:
            int: Index of medoid point
        """
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate pairwise distances within cluster
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(cluster_embeddings)
        
        # Find point with minimum sum of distances to all other points
        sum_distances = np.sum(distances, axis=1)
        medoid_idx_in_cluster = np.argmin(sum_distances)
        medoid_idx_global = cluster_indices[medoid_idx_in_cluster]
        
        return medoid_idx_global
    
    def extract_key_features(self, text):
        """
        Extract key features from text untuk interpretability
        
        Args:
            text: Input text string
            
        Returns:
            dict: Dictionary of extracted features
        """
        import re
        from collections import Counter
        
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        
        # Extract potential keywords (words longer than 3 chars, not common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                       'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                       'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 
                       'that', 'these', 'those', 'a', 'an'}
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in common_words]
        keyword_freq = Counter(keywords)
        
        # Extract hashtags and mentions if present
        hashtags = re.findall(r'#\w+', text)
        mentions = re.findall(r'@\w+', text)
        
        # Detect potential sentiment words (basic approach)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best', 'perfect'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'stupid', 'ugly', 'annoying'}
        
        pos_sentiment = sum(1 for w in words if w in positive_words)
        neg_sentiment = sum(1 for w in words if w in negative_words)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'top_keywords': keyword_freq.most_common(5),
            'hashtags': hashtags,
            'mentions': mentions,
            'positive_sentiment_words': pos_sentiment,
            'negative_sentiment_words': neg_sentiment,
            'sentiment_score': pos_sentiment - neg_sentiment
        }
    
    def analyze_clusters(self, cluster_labels, text_data, embeddings):
        """
        Analyze cluster distribution dan content dengan medoid analysis
        
        Args:
            cluster_labels: Array of cluster assignments
            text_data: List of text data dictionaries
            embeddings: Array of embeddings untuk medoid calculation
            
        Returns:
            dict: Cluster analysis results with medoids
        """
        unique_clusters = np.unique(cluster_labels)
        cluster_analysis = {}
        
        print(f"\n=== CLUSTER ANALYSIS WITH MEDOIDS ===")
        print(f"Number of clusters: {len(unique_clusters)}")
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # Get texts and metadata in this cluster
            cluster_texts = [text_data[i]['text'] for i in cluster_indices]
            cluster_metadata = [text_data[i]['metadata'] for i in cluster_indices]
            
            # Find medoid
            medoid_idx = self.find_medoid(embeddings, cluster_indices)
            medoid_text = text_data[medoid_idx]['text']
            medoid_metadata = text_data[medoid_idx]['metadata']
            medoid_features = self.extract_key_features(medoid_text)
            
            # Analyze cluster characteristics
            all_keywords = []
            sentiment_scores = []
            for i in cluster_indices:
                features = self.extract_key_features(text_data[i]['text'])
                all_keywords.extend([kw[0] for kw in features['top_keywords']])
                sentiment_scores.append(features['sentiment_score'])
            
            # Find most common keywords in cluster
            from collections import Counter
            cluster_keywords = Counter(all_keywords).most_common(10)
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            cluster_analysis[int(cluster_id)] = {
                'cluster_id': f"cluster_{cluster_id}",
                'size': cluster_size,
                'percentage': cluster_size / len(cluster_labels) * 100,
                'texts': cluster_texts,
                'metadata': cluster_metadata,
                'indices': cluster_indices.tolist(),
                
                # Medoid information
                'medoid_index': int(medoid_idx),
                'medoid_text': medoid_text,
                'medoid_metadata': medoid_metadata,
                'medoid_features': medoid_features,
                
                # Cluster characteristics
                'top_cluster_keywords': cluster_keywords,
                'average_sentiment': avg_sentiment,
                'sentiment_distribution': {
                    'positive': sum(1 for s in sentiment_scores if s > 0),
                    'neutral': sum(1 for s in sentiment_scores if s == 0),
                    'negative': sum(1 for s in sentiment_scores if s < 0)
                }
            }
            
            print(f"\nCluster {cluster_id}: {cluster_size} texts ({cluster_size/len(cluster_labels)*100:.1f}%)")
            print(f"  Medoid text: {medoid_text[:100]}...")
            print(f"  Top keywords: {[kw[0] for kw in cluster_keywords[:5]]}")
            print(f"  Average sentiment: {avg_sentiment:.2f}")
            
            # Show sample texts
            sample_texts = cluster_texts[:3]
            for i, text in enumerate(sample_texts):
                print(f"  Sample {i+1}: {text[:80]}...")
            
            if len(cluster_texts) > 3:
                print(f"  ... and {len(cluster_texts)-3} more texts")
        
        return cluster_analysis
    
    def save_clustering_results(self, cluster_labels, cluster_analysis, text_data, 
                               output_path):
        """
        Save clustering results dalam format yang compatible dengan BacKGen
        
        Args:
            cluster_labels: Array of cluster assignments
            cluster_analysis: Cluster analysis results
            text_data: Original text data
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        clustering_results = []
        
        for i, (cluster_id, text_item) in enumerate(zip(cluster_labels, text_data)):
            result_item = {
                'index': i,
                'cluster_id': int(cluster_id),
                'text': text_item['text'],
                'metadata': text_item['metadata']
            }
            clustering_results.append(result_item)
        
        # Save clustering results
        write_jsonl(clustering_results, output_path)
        
        # Save cluster summary
        summary_path = output_path.replace('.jsonl', '_summary.json')
        cluster_summary = {
            'total_texts': len(text_data),
            'num_clusters': len(np.unique(cluster_labels)),
            'clusters': {}
        }
        
        for cluster_id, analysis in cluster_analysis.items():
            cluster_summary['clusters'][cluster_id] = {
                'size': analysis['size'],
                'percentage': analysis['percentage'],
                'sample_texts': analysis['texts'][:5]  # First 5 texts as samples
            }
        
        with open(summary_path, 'w') as f:
            json.dump(cluster_summary, f, indent=2)
        
        print(f"Clustering results saved to: {output_path}")
        print(f"Cluster summary saved to: {summary_path}")

def cluster_single_file(input_file, output_base, evaluate_k=False):
    """
    Main function untuk clustering single input file
    
    Args:
        input_file: Path ke input file (.jsonl dengan embeddings)
        output_base: Base path untuk output directory
        evaluate_k: Whether to evaluate different k values
        
    Returns:
        dict: Clustering results
    """
    clusterer = SBERTClusterer()
    
    print(f"=== CLUSTERING DATA FROM {input_file} ===")
    
    # Load data
    embeddings, text_data = clusterer.load_data_from_jsonl(input_file)
    
    # Calculate k
    if evaluate_k:
        k, silhouette_scores = clusterer.calculate_optimal_k(embeddings)
    else:
        k = max(2, len(embeddings) // 5)  # BacKGen approach
        print(f"Using BacKGen k={k} for data ({len(embeddings)} texts)")
    
    # Perform clustering
    kmeans_model, cluster_labels, cluster_centers = clusterer.perform_clustering(embeddings, k)
    cluster_analysis = clusterer.analyze_clusters(cluster_labels, text_data)
    
    # Prepare output path
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_base, f"{input_basename}_clustered.jsonl")
    
    # Save results
    clusterer.save_clustering_results(cluster_labels, cluster_analysis, text_data, output_file)
    
    results = {
        'k': k,
        'labels': cluster_labels,
        'analysis': cluster_analysis,
        'output_file': output_file
    }
    
    return results

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
    pos_output = os.path.join(output_dir, 'positive_clustered.jsonl')
    clusterer.save_clustering_results(pos_labels, pos_analysis, pos_texts, pos_output)
    
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
    neg_output = os.path.join(output_dir, 'negative_clustered.jsonl')
    clusterer.save_clustering_results(neg_labels, neg_analysis, neg_texts, neg_output)
    
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
  python clustering.py --input_file data/mwe/backgen_1_extracted_frame/mwe-train-frame.jsonl --output_base data
  
  # Legacy mode (positive/negative files)
  python clustering.py --legacy_mode --output_base data/clusters
  
  # Evaluate optimal k
  python clustering.py --input_file data/input.jsonl --output_base data --evaluate_k
        """
    )
    
    parser.add_argument('--input_file', type=str, 
                       help='Path to input JSONL file containing text and embeddings')
    parser.add_argument('--output_base', type=str, required=True,
                       help='Base path for output directory')
    parser.add_argument('--evaluate_k', action='store_true',
                       help='Evaluate different k values using silhouette score')
    parser.add_argument('--legacy_mode', action='store_true',
                       help='Use legacy mode with separate positive/negative files')
    
    args = parser.parse_args()
    
    try:
        print("=== SBERT K-MEANS CLUSTERING FOR BACKGEN ===")
        
        if args.legacy_mode:
            # Legacy mode dengan positive/negative files
            positive_pkl = "data/embeddings/positive_embeddings.pkl"
            positive_jsonl = "data/embeddings/positive_embeddings.jsonl"
            negative_pkl = "data/embeddings/negative_embeddings.pkl"
            negative_jsonl = "data/embeddings/negative_embeddings.jsonl"
            
            print("Running in legacy mode...")
            print(f"Input files:")
            print(f"  Positive: {positive_pkl}, {positive_jsonl}")
            print(f"  Negative: {negative_pkl}, {negative_jsonl}")
            
            # Check input files
            required_files = [positive_pkl, positive_jsonl, negative_pkl, negative_jsonl]
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")
            
            results = cluster_polarity_data(
                positive_pkl=positive_pkl,
                positive_jsonl=positive_jsonl,
                negative_pkl=negative_pkl,
                negative_jsonl=negative_jsonl,
                output_dir=args.output_base,
                evaluate_k=args.evaluate_k
            )
            
            # Final summary
            print(f"\n=== CLUSTERING COMPLETED ===")
            print(f"Positive clusters: {results['positive']['k']} clusters")
            print(f"  -> {results['positive']['output_file']}")
            print(f"Negative clusters: {results['negative']['k']} clusters") 
            print(f"  -> {results['negative']['output_file']}")
            
        else:
            # Single file mode
            if not args.input_file:
                parser.error("--input_file is required when not using --legacy_mode")
            
            if not os.path.exists(args.input_file):
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
            print(f"Input file: {args.input_file}")
            print(f"Output base: {args.output_base}")
            
            results = cluster_single_file(
                input_file=args.input_file,
                output_base=args.output_base,
                evaluate_k=args.evaluate_k
            )
            
            # Final summary
            print(f"\n=== CLUSTERING COMPLETED ===")
            print(f"Number of clusters: {results['k']}")
            print(f"Output file: {results['output_file']}")
        
        print(f"\nNext steps for BacKGen:")
        print(f"1. Use clustered data for Background Knowledge generation")
        print(f"2. Each cluster represents semantically similar texts")
        print(f"3. Generate BK for each cluster separately")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        if args.legacy_mode:
            print("Please run the embedding generation script first!")
        else:
            print("Please check the input file path!")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages: pip install scikit-learn")
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()