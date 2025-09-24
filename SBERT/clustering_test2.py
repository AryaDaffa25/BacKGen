import pickle
import json
import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Import BacKGen utilities
from converter.io import read_jsonl, write_jsonl


class SBERTClusterer:
    """
    Class untuk melakukan K-means clustering pada SBERT embeddings
    mengikuti approach BacKGen dengan k = data_count / 5
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.kmeans_models = {}

    def load_data_from_jsonl(self, jsonl_path):
        text_data = []
        embeddings = []

        print(f"Loading data from: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    if 'embedding' in data:
                        embedding = np.array(data['embedding'])
                    elif 'embeddings' in data:
                        embedding = np.array(data['embeddings'])
                    else:
                        raise KeyError("No 'embedding' or 'embeddings' key found in data")

                    embeddings.append(embedding)
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
        with open(pkl_path, 'rb') as f:
            embedding_data = pickle.load(f)
        embeddings = embedding_data['embeddings']

        text_data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                text_data.append(json.loads(line.strip()))

        print(f"Loaded embeddings shape: {embeddings.shape}")
        print(f"Loaded text data: {len(text_data)} entries")

        if len(embeddings) != len(text_data):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(text_data)} texts")
        return embeddings, text_data

    def calculate_optimal_k(self, embeddings, min_k=2, max_k=None):
        data_count = len(embeddings)
        backgen_k = max(min_k, data_count // 5)

        if max_k is None:
            max_k = min(50, data_count // 3)

        print(f"Data count: {data_count}")
        print(f"BacKGen recommended k: {backgen_k}")
        print(f"Evaluating k range: {min_k} to {max_k}")

        silhouette_scores = {}
        k_range = range(min_k, min(max_k + 1, data_count))
        for k in tqdm(k_range, desc="Evaluating k values"):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores[k] = score

        best_k = max(silhouette_scores.keys(), key=lambda k: silhouette_scores[k])
        print(f"Best k by silhouette score: {best_k} (score: {silhouette_scores[best_k]:.3f})")
        print(f"BacKGen k: {backgen_k} (score: {silhouette_scores.get(backgen_k, 'N/A')})")
        return backgen_k, silhouette_scores

    def perform_clustering(self, embeddings, k):
        print(f"Performing K-means clustering with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(embeddings)

        inertia = kmeans.inertia_
        if k > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            print(f"Clustering completed: Inertia={inertia:.2f}, Silhouette={silhouette_avg:.3f}")
        else:
            print(f"Clustering completed with inertia: {inertia:.2f}")

        return kmeans, cluster_labels, kmeans.cluster_centers_

    def find_medoid(self, embeddings, cluster_indices):
        if len(cluster_indices) == 1:
            return cluster_indices[0]

        cluster_embeddings = embeddings[cluster_indices]
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(cluster_embeddings)
        sum_distances = np.sum(distances, axis=1)
        medoid_idx_in_cluster = np.argmin(sum_distances)
        medoid_idx_global = cluster_indices[medoid_idx_in_cluster]
        return medoid_idx_global

    def analyze_clusters(self, cluster_labels, text_data, embeddings):
        unique_clusters = np.unique(cluster_labels)
        cluster_analysis = {}
        print(f"\n=== CLUSTER ANALYSIS WITH MEDOIDS ===")
        print(f"Number of clusters: {len(unique_clusters)}")

        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_size = len(cluster_indices)

            cluster_texts = [text_data[i]['text'] for i in cluster_indices]
            cluster_metadata = [text_data[i]['metadata'] for i in cluster_indices]

            medoid_idx = self.find_medoid(embeddings, cluster_indices)
            medoid_text = text_data[medoid_idx]['text']
            medoid_metadata = text_data[medoid_idx]['metadata']

            cluster_analysis[int(cluster_id)] = {
                'cluster_id': f"cluster_{cluster_id}",
                'size': cluster_size,
                'percentage': cluster_size / len(cluster_labels) * 100,
                'texts': cluster_texts,
                'metadata': cluster_metadata,
                'indices': cluster_indices.tolist(),
                'medoid_index': int(medoid_idx),
                'medoid_text': medoid_text,
                'medoid_metadata': medoid_metadata
            }

            print(f"\nCluster {cluster_id}: {cluster_size} texts ({cluster_size/len(cluster_labels)*100:.1f}%)")
            print(f"  Medoid text: {medoid_text[:100]}...")
            for i, text in enumerate(cluster_texts[:3]):
                print(f"  Sample {i+1}: {text[:80]}...")
            if len(cluster_texts) > 3:
                print(f"  ... and {len(cluster_texts)-3} more texts")
        return cluster_analysis

    def save_clustering_results(self, cluster_labels, cluster_analysis, text_data, output_path):
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

        write_jsonl(clustering_results, output_path)

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
                'medoid_text': analysis['medoid_text'],
                'medoid_index': analysis['medoid_index'],
                'sample_texts': analysis['texts'][:5]
            }
        with open(summary_path, 'w') as f:
            json.dump(cluster_summary, f, indent=2)

        print(f"Clustering results saved to: {output_path}")
        print(f"Cluster summary saved to: {summary_path}")


def cluster_polarity_data(positive_pkl, positive_jsonl, negative_pkl, negative_jsonl,
                          output_dir, evaluate_k=False):
    clusterer = SBERTClusterer()
    results = {}

    # Positive
    print("=== CLUSTERING POSITIVE DATA ===")
    pos_embeddings, pos_texts = clusterer.load_embeddings_and_texts(positive_pkl, positive_jsonl)
    pos_k = max(2, len(pos_embeddings) // 5) if not evaluate_k else clusterer.calculate_optimal_k(pos_embeddings)[0]
    pos_kmeans, pos_labels, _ = clusterer.perform_clustering(pos_embeddings, pos_k)
    pos_analysis = clusterer.analyze_clusters(pos_labels, pos_texts, pos_embeddings)
    pos_output = os.path.join(output_dir, 'positive_clustered.jsonl')
    clusterer.save_clustering_results(pos_labels, pos_analysis, pos_texts, pos_output)
    results['positive'] = {'k': pos_k, 'labels': pos_labels, 'analysis': pos_analysis, 'output_file': pos_output}

    # Negative
    print("\n=== CLUSTERING NEGATIVE DATA ===")
    neg_embeddings, neg_texts = clusterer.load_embeddings_and_texts(negative_pkl, negative_jsonl)
    neg_k = max(2, len(neg_embeddings) // 5) if not evaluate_k else clusterer.calculate_optimal_k(neg_embeddings)[0]
    neg_kmeans, neg_labels, _ = clusterer.perform_clustering(neg_embeddings, neg_k)
    neg_analysis = clusterer.analyze_clusters(neg_labels, neg_texts, neg_embeddings)
    neg_output = os.path.join(output_dir, 'negative_clustered.jsonl')
    clusterer.save_clustering_results(neg_labels, neg_analysis, neg_texts, neg_output)
    results['negative'] = {'k': neg_k, 'labels': neg_labels, 'analysis': neg_analysis, 'output_file': neg_output}

    return results


def main():
    parser = argparse.ArgumentParser(description='SBERT K-means Clustering with Medoid Support')
    parser.add_argument("--positive_jsonl", type=str, help="Path to positive .jsonl file")
    parser.add_argument("--positive_pkl", type=str, help="Path to positive .pkl embeddings file")
    parser.add_argument("--negative_jsonl", type=str, help="Path to negative .jsonl file")
    parser.add_argument("--negative_pkl", type=str, help="Path to negative .pkl embeddings file")
    parser.add_argument("--output_base", type=str, required=True, help="Output directory")
    parser.add_argument("--evaluate_k", action="store_true", help="Evaluate optimal k")
    args = parser.parse_args()

    try:
        print("=== SBERT K-MEANS CLUSTERING (MEDOID ENABLED) ===")
        if args.positive_jsonl and args.positive_pkl and args.negative_jsonl and args.negative_pkl:
            results = cluster_polarity_data(
                positive_pkl=args.positive_pkl,
                positive_jsonl=args.positive_jsonl,
                negative_pkl=args.negative_pkl,
                negative_jsonl=args.negative_jsonl,
                output_dir=args.output_base,
                evaluate_k=args.evaluate_k
            )
            print("\n=== CLUSTERING COMPLETED ===")
            print(f"Positive clusters: {results['positive']['k']} -> {results['positive']['output_file']}")
            print(f"Negative clusters: {results['negative']['k']} -> {results['negative']['output_file']}")
        else:
            parser.error("Please provide --positive_jsonl, --positive_pkl, --negative_jsonl, --negative_pkl")
    except Exception as e:
        print(f"Error during clustering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
