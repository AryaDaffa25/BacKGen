import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

# Import BacKGen utilities
from converter.io import read_jsonl, write_jsonl

class SBERTEmbeddingGenerator:
    """
    Class untuk generate embeddings menggunakan SentenceBERT
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize SBERT model
        
        Args:
            model_name: Nama model SentenceBERT
                       - 'all-MiniLM-L6-v2': Fast and good performance (default)
                       - 'all-mpnet-base-v2': Best quality but slower
                       - 'paraphrase-multilingual-MiniLM-L12-v2': For multilingual
        """
        print(f"Loading SentenceBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model loaded successfully!")
    
    def generate_embeddings(self, texts, batch_size=32, show_progress=True):
        """
        Generate embeddings untuk list of texts
        
        Args:
            texts: List of strings
            batch_size: Batch size untuk encoding (default: 32)
            show_progress: Show progress bar (default: True)
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if show_progress:
            print(f"Generating embeddings for {len(texts)} texts...")
            
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings, output_path):
        """
        Save embeddings ke file pickle
        
        Args:
            embeddings: numpy array of embeddings
            output_path: Path untuk save embeddings
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as pickle for efficiency
        with open(output_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'model_name': self.model_name,
                'shape': embeddings.shape
            }, f)
        
        print(f"Embeddings saved to: {output_path}")
    
    def load_embeddings(self, embedding_path):
        """
        Load embeddings dari file
        
        Args:
            embedding_path: Path ke embedding file
            
        Returns:
            dict: Dictionary containing embeddings and metadata
        """
        with open(embedding_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded embeddings: {data['shape']} from model: {data['model_name']}")
        return data

def process_jsonl_to_embeddings(input_jsonl_path, output_embedding_path, 
                               text_field='text', model_name='all-MiniLM-L6-v2'):
    """
    Process JSONL file untuk generate embeddings
    
    Args:
        input_jsonl_path: Path ke input JSONL file
        output_embedding_path: Path untuk save embeddings (.pkl)
        text_field: Field name untuk text yang akan di-embed (default: 'text')
        model_name: SentenceBERT model name
        
    Returns:
        tuple: (embeddings, texts, metadata)
    """
    print(f"Processing: {input_jsonl_path}")
    
    # Read JSONL data
    data = read_jsonl(input_jsonl_path)
    print(f"Loaded {len(data)} entries from {input_jsonl_path}")
    
    # Extract texts
    texts = []
    metadata = []
    
    for item in data:
        text = item.get(text_field, '')
        if text.strip():  # Only add non-empty texts
            texts.append(text.strip())
            metadata.append({
                'id': item.get('id'),
                'st_id': item.get('st_id'),
                'st_span': item.get('st_span'),
                'st_polarity': item.get('st_polarity')
            })
    
    print(f"Extracted {len(texts)} valid texts")
    
    # Generate embeddings
    generator = SBERTEmbeddingGenerator(model_name=model_name)
    embeddings = generator.generate_embeddings(texts)
    
    # Save embeddings as pickle
    generator.save_embeddings(embeddings, output_embedding_path)
    
    # Save metadata and texts as JSONL
    jsonl_path = output_embedding_path.replace('.pkl', '.jsonl')
    jsonl_data = []
    
    for i, (text, meta) in enumerate(zip(texts, metadata)):
        jsonl_data.append({
            'index': i,
            'text': text,
            'metadata': meta,
            'model_name': model_name
        })
    
    write_jsonl(jsonl_data, jsonl_path)
    print(f"Text and metadata saved to: {jsonl_path}")
    
    return embeddings, texts, metadata

def generate_polarity_embeddings(positive_file, negative_file, output_dir, 
                                model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings untuk both positive dan negative polarity files
    
    Args:
        positive_file: Path ke positive polarity JSONL
        negative_file: Path ke negative polarity JSONL  
        output_dir: Directory untuk save embeddings
        model_name: SentenceBERT model name
        
    Returns:
        dict: Dictionary with embedding paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Process positive data
    print("\n=== PROCESSING POSITIVE DATA ===")
    positive_embedding_path = os.path.join(output_dir, 'positive_embeddings.pkl')
    pos_embeddings, pos_texts, pos_metadata = process_jsonl_to_embeddings(
        positive_file, positive_embedding_path, model_name=model_name
    )
    results['positive'] = {
        'embedding_path': positive_embedding_path,
        'embeddings': pos_embeddings,
        'texts': pos_texts,
        'metadata': pos_metadata,
        'count': len(pos_texts)
    }
    
    # Process negative data
    print("\n=== PROCESSING NEGATIVE DATA ===")
    negative_embedding_path = os.path.join(output_dir, 'negative_embeddings.pkl')
    neg_embeddings, neg_texts, neg_metadata = process_jsonl_to_embeddings(
        negative_file, negative_embedding_path, model_name=model_name
    )
    results['negative'] = {
        'embedding_path': negative_embedding_path,
        'embeddings': neg_embeddings,
        'texts': neg_texts,
        'metadata': neg_metadata,
        'count': len(neg_texts)
    }
    
    return results

def main():
    """
    Main function untuk generate embeddings dari polarity files
    """
    # File paths
    positive_file = "data/output/polarity_positive.jsonl"
    negative_file = "data/output/polarity_negative.jsonl"
    embedding_output_dir = "data/embeddings"
    
    # Model options:
    # - 'all-MiniLM-L6-v2': Fast, good performance (384 dimensions)
    # - 'all-mpnet-base-v2': Best quality (768 dimensions) 
    # - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual support
    model_name = 'all-MiniLM-L6-v2'
    
    try:
        print(f"=== SBERT EMBEDDING GENERATION ===")
        print(f"Model: {model_name}")
        print(f"Input files:")
        print(f"  - Positive: {positive_file}")
        print(f"  - Negative: {negative_file}")
        print(f"Output directory: {embedding_output_dir}")
        
        # Check if input files exist
        if not os.path.exists(positive_file):
            raise FileNotFoundError(f"Positive file not found: {positive_file}")
        if not os.path.exists(negative_file):
            raise FileNotFoundError(f"Negative file not found: {negative_file}")
        
        # Generate embeddings
        results = generate_polarity_embeddings(
            positive_file=positive_file,
            negative_file=negative_file,
            output_dir=embedding_output_dir,
            model_name=model_name
        )
        
        # Summary
        print(f"\n=== EMBEDDING GENERATION COMPLETED ===")
        print(f"Positive embeddings: {results['positive']['count']} texts")
        print(f"  -> {results['positive']['embedding_path']}")
        print(f"  -> Shape: {results['positive']['embeddings'].shape}")
        
        print(f"Negative embeddings: {results['negative']['count']} texts")
        print(f"  -> {results['negative']['embedding_path']}")
        print(f"  -> Shape: {results['negative']['embeddings'].shape}")
        
        print(f"\nFiles ready for clustering:")
        print(f"1. Load embeddings: pickle.load()")
        print(f"2. Apply K-means with k = data_count / 5")
        print(f"3. Continue with BacKGen pipeline")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the polarity splitting script first!")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages: pip install sentence-transformers")
    except Exception as e:
        print(f"Error during embedding generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Utility functions untuk load dan inspect embeddings
def load_and_inspect_embeddings(embedding_path):
    """
    Utility function untuk load dan inspect embeddings
    
    Args:
        embedding_path: Path ke embedding file
    """
    generator = SBERTEmbeddingGenerator()
    data = generator.load_embeddings(embedding_path)
    
    print(f"Embedding details:")
    print(f"  Model: {data['model_name']}")
    print(f"  Shape: {data['shape']}")
    print(f"  Sample embedding (first 5 dims): {data['embeddings'][0][:5]}")
    
    return data

def prepare_for_clustering(positive_embedding_path, negative_embedding_path):
    """
    Prepare embeddings untuk clustering
    
    Args:
        positive_embedding_path: Path ke positive embeddings
        negative_embedding_path: Path ke negative embeddings
        
    Returns:
        dict: Dictionary dengan data untuk clustering
    """
    # Load embeddings
    with open(positive_embedding_path, 'rb') as f:
        pos_data = pickle.load(f)
    
    with open(negative_embedding_path, 'rb') as f:
        neg_data = pickle.load(f)
    
    # Calculate recommended k values (data_count / 5)
    pos_k = max(1, len(pos_data['embeddings']) // 5)
    neg_k = max(1, len(neg_data['embeddings']) // 5)
    
    clustering_data = {
        'positive': {
            'embeddings': pos_data['embeddings'],
            'recommended_k': pos_k,
            'data_count': len(pos_data['embeddings'])
        },
        'negative': {
            'embeddings': neg_data['embeddings'], 
            'recommended_k': neg_k,
            'data_count': len(neg_data['embeddings'])
        }
    }
    
    print(f"Clustering preparation:")
    print(f"  Positive: {clustering_data['positive']['data_count']} samples, recommended k: {pos_k}")
    print(f"  Negative: {clustering_data['negative']['data_count']} samples, recommended k: {neg_k}")
    
    return clustering_data