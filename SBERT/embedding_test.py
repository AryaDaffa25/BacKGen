import numpy as np
import json
import os
import argparse
import time
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
        print(f"Loading SentenceBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model loaded successfully!")
    
    def generate_embeddings(self, texts, batch_size=32, show_progress=True):
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'model_name': self.model_name,
                'shape': embeddings.shape
            }, f)
        print(f"Embeddings saved to: {output_path}")
    
    def load_embeddings(self, embedding_path):
        with open(embedding_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded embeddings: {data['shape']} from model: {data['model_name']}")
        return data

def process_jsonl_to_embeddings(input_jsonl_path, output_embedding_path, 
                               text_field='text', model_name='all-MiniLM-L6-v2'):
    print(f"Processing: {input_jsonl_path}")
    
    data = read_jsonl(input_jsonl_path)
    print(f"Loaded {len(data)} entries from {input_jsonl_path}")
    
    texts, metadata = [], []
    for item in data:
        text = item.get(text_field, '')
        if text.strip():
            texts.append(text.strip())
            metadata.append({
                'id': item.get('id'),
                'st_id': item.get('st_id'),
                'st_span': item.get('st_span'),
                'st_polarity': item.get('st_polarity')
            })
    
    print(f"Extracted {len(texts)} valid texts")
    
    generator = SBERTEmbeddingGenerator(model_name=model_name)
    embeddings = generator.generate_embeddings(texts)
    
    generator.save_embeddings(embeddings, output_embedding_path)
    
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
    start_time = time.time()
    # Process positive data
    print("\n=== PROCESSING POSITIVE DATA ===")
    positive_basename = os.path.splitext(os.path.basename(positive_file))[0]
    positive_embedding_path = os.path.join(output_dir, f"{positive_basename}_embeddings.pkl")
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
    negative_basename = os.path.splitext(os.path.basename(negative_file))[0]
    negative_embedding_path = os.path.join(output_dir, f"{negative_basename}_embeddings.pkl")
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
    end_time = time.time()
    print(f"\n⏱️ Total runtime embedding generation: {end_time - start_time:.2f} detik")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate SBERT embeddings from positive & negative JSONL files"
    )
    parser.add_argument("--positive_file", type=str, required=True,
                        help="Path ke positive polarity JSONL file")
    parser.add_argument("--negative_file", type=str, required=True,
                        help="Path ke negative polarity JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory untuk menyimpan embeddings")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceBERT model name (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()
    
    try:
        print(f"=== SBERT EMBEDDING GENERATION ===")
        print(f"Model: {args.model_name}")
        print(f"Input files: {args.positive_file}, {args.negative_file}")
        print(f"Output directory: {args.output_dir}")
        
        if not os.path.exists(args.positive_file):
            raise FileNotFoundError(f"Positive file not found: {args.positive_file}")
        if not os.path.exists(args.negative_file):
            raise FileNotFoundError(f"Negative file not found: {args.negative_file}")
        
        results = generate_polarity_embeddings(
            positive_file=args.positive_file,
            negative_file=args.negative_file,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        
        print(f"\n=== EMBEDDING GENERATION COMPLETED ===")
        print(f"Positive embeddings: {results['positive']['count']} texts")
        print(f"  -> {results['positive']['embedding_path']}")
        print(f"  -> Shape: {results['positive']['embeddings'].shape}")
        
        print(f"Negative embeddings: {results['negative']['count']} texts")
        print(f"  -> {results['negative']['embedding_path']}")
        print(f"  -> Shape: {results['negative']['embeddings'].shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
