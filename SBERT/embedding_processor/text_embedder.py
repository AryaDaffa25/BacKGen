# Import libraries langsung (karena sudah dalam struktur BacKGen)
from sentence_transformers import SentenceTransformer
from converter.io import read_jsonl, write_jsonl
import numpy as np
import pickle
from typing import List, Dict, Any

class TextEmbedder:
    """
    Class untuk melakukan embedding extraction menggunakan SentenceBERT
    Menggantikan frame extraction pada BacKGen original
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceBERT model
        
        Args:
            model_name: nama model SentenceBERT yang akan digunakan
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def extract_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract embeddings dari list of texts
        
        Args:
            texts: list of text strings
            
        Returns:
            numpy array of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def process_sentiment_data(self, input_file_path: str, output_file_path: str, 
                             embedding_output_path: str, verbose: str = "yes") -> Dict[str, Any]:
        """
        Process data sentiment dan extract embeddings
        Menggantikan frame extraction step pada BacKGen original
        
        Args:
            input_file_path: path ke file jsonl input (misal: mwe-train.jsonl)
            output_file_path: path untuk menyimpan processed data
            embedding_output_path: path untuk menyimpan embeddings
            verbose: menampilkan progress atau tidak
            
        Returns:
            Dictionary berisi processed data dan metadata
        """
        # Load data
        data = read_jsonl(input_file_path)
        
        if verbose == "yes":
            print(f"Loading data from {input_file_path}")
            print(f"Total data: {len(data)}")
        
        # Extract texts dan metadata
        processed_data = []
        texts = []
        
        for idx, item in enumerate(data):
            # Extract text dan sentiment information
            text = item.get('text', '')
            sent_id = item.get('id', idx)
            
            # Process sentiment tuples jika ada
            sentiment_tuples = item.get('oesc_tuple', [])
            
            # Create processed entry
            processed_entry = {
                'id': sent_id,
                'text': text,
                'sentiment_tuples': sentiment_tuples,
                'original_data': item
            }
            
            processed_data.append(processed_entry)
            texts.append(text)
        
        if verbose == "yes":
            print("Extracting embeddings using SentenceBERT...")
        
        # Extract embeddings
        embeddings = self.extract_text_embeddings(texts)
        
        # Add embeddings to processed data
        for i, entry in enumerate(processed_data):
            entry['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization
        
        # Save processed data
        write_jsonl(processed_data, output_file_path)
        
        # Save embeddings separately
        with open(embedding_output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        if verbose == "yes":
            print(f"Processed data saved to: {output_file_path}")
            print(f"Embeddings saved to: {embedding_output_path}")
            print(f"Embedding shape: {embeddings.shape}")
        
        return {
            'processed_data': processed_data,
            'embeddings': embeddings,
            'texts': texts,
            'metadata': {
                'model_name': self.model_name,
                'num_samples': len(data),
                'embedding_dim': embeddings.shape[1]
            }
        }

def extract_embeddings_from_frame_data(frame_file_path: str, output_processed_path: str, 
                                     output_embedding_path: str, model_name: str = "all-MiniLM-L6-v2",
                                     verbose: str = "yes"):
    """
    Fungsi utility untuk extract embeddings dari frame data
    Menggantikan langkah 1 pada BK Generation process
    
    Args:
        frame_file_path: path ke file frame data (misal: mwe-train-frame.jsonl)
        output_processed_path: path output untuk processed data
        output_embedding_path: path output untuk embeddings
        model_name: nama model SentenceBERT
        verbose: menampilkan progress
    """
    embedder = TextEmbedder(model_name)
    
    # Load frame data
    frame_data = read_jsonl(frame_file_path)
    
    if verbose == "yes":
        print(f"Processing frame data from: {frame_file_path}")
        print(f"Total frame entries: {len(frame_data)}")
    
    # Extract texts dari frame data
    processed_data = []
    texts = []
    
    for idx, item in enumerate(frame_data):
        text = item.get('text', '')
        sent_id = item.get('sent_id', idx)
        frame_tree = item.get('frame_tree', [])
        
        processed_entry = {
            'id': sent_id,
            'text': text,
            'frame_tree': frame_tree,
            'original_frame_data': item
        }
        
        processed_data.append(processed_entry)
        texts.append(text)
    
    if verbose == "yes":
        print("Extracting embeddings...")
    
    # Extract embeddings
    embeddings = embedder.extract_text_embeddings(texts)
    
    # Add embeddings to processed data
    for i, entry in enumerate(processed_data):
        entry['embedding'] = embeddings[i].tolist()
    
    # Save results
    write_jsonl(processed_data, output_processed_path)
    
    with open(output_embedding_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    if verbose == "yes":
        print(f"Processed data saved to: {output_processed_path}")
        print(f"Embeddings saved to: {output_embedding_path}")
        print(f"Embedding shape: {embeddings.shape}")
    
    return processed_data, embeddings