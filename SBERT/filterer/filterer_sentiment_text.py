# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from converter.io import read_jsonl, write_jsonl
from converter.converter_utility import spacy_tokenizer
from filterer.filterer_utility import all_indices

def filter_sentiment_text_embedding(data_sentiment, data_text_embedding):
    """
    Filter text embeddings berdasarkan sentiment labels
    Adaptasi dari filter_sentiment_frame_utt untuk embedding-based approach
    
    Args:
        data_sentiment: data dengan sentiment labels
        data_text_embedding: data dengan embeddings
        
    Returns:
        tuple: (text_positive, text_negative, text_neutral) dengan embeddings
    """
    text_positive, text_negative, text_neutral = [], [], []
    
    id_key = data_sentiment.get("id")
    text = data_text_embedding.get("text", "").lower()
    embedding = data_text_embedding.get("embedding")
    
    sentiment_tuples = data_sentiment.get("oesc_tuple", [])
    
    # Base entry dengan embedding
    base_entry = {
        "id": id_key, 
        "text": text, 
        "embedding": embedding,
        "original_data": data_text_embedding.get("original_frame_data", data_text_embedding)
    }
    
    if sentiment_tuples == []:
        # Jika tidak ada sentiment tuple, masukkan ke neutral
        text_neutral.append(base_entry)
    else:
        # Process sentiment tuples
        text_token = spacy_tokenizer(text)
        
        for sentiment_tuple in sentiment_tuples:
            sentiment_span = sentiment_tuple[0].lower()
            sentiment_polarity = sentiment_tuple[1]
            
            sentiment_span_token = spacy_tokenizer(sentiment_span)
            
            try:
                idx_first_word = all_indices(text_token, sentiment_span_token[0])
                idx_last_word = all_indices(text_token, sentiment_span_token[-1])
                idx_sentiment_span = [idx_first_word[-1], idx_last_word[0]]
            except:
                # Skip jika tidak bisa find sentiment span
                continue
            
            # Create entry dengan sentiment info
            sentiment_entry = {
                **base_entry,
                "sentiment_span": sentiment_span,
                "sentiment_polarity": sentiment_polarity,
                "sentiment_span_indices": idx_sentiment_span
            }
            
            # Kategorisasi berdasarkan polarity
            if sentiment_polarity == "Exp_Positive":
                if sentiment_entry not in text_positive:
                    text_positive.append(sentiment_entry)
            elif sentiment_polarity == "Exp_Negative":
                if sentiment_entry not in text_negative:
                    text_negative.append(sentiment_entry)
            else:
                # Sentiment lain dianggap neutral
                if sentiment_entry not in text_neutral:
                    text_neutral.append(sentiment_entry)
    
    return text_positive, text_negative, text_neutral

def filter_sentiment_text_list(data_sentiment_list, data_text_embedding_list):
    """
    Filter sentiment untuk list of data
    
    Args:
        data_sentiment_list: list data sentiment
        data_text_embedding_list: list data dengan embeddings
        
    Returns:
        tuple: (list_text_positive, list_text_negative, list_text_neutral)
    """
    list_text_positive, list_text_negative, list_text_neutral = [], [], []
    
    for i in range(len(data_sentiment_list)):
        # Find matching embedding data berdasarkan ID
        sentiment_id = data_sentiment_list[i].get('id')
        
        # Cari data embedding yang matching
        matching_embedding_data = None
        for emb_data in data_text_embedding_list:
            if str(emb_data.get('id')) == str(sentiment_id):
                matching_embedding_data = emb_data
                break
        
        if matching_embedding_data is None:
            print(f"Warning: No embedding data found for sentiment ID: {sentiment_id}")
            continue
            
        text_positive, text_negative, text_neutral = filter_sentiment_text_embedding(
            data_sentiment_list[i], matching_embedding_data
        )
        
        list_text_positive.extend(text_positive)
        list_text_negative.extend(text_negative)
        list_text_neutral.extend(text_neutral)
    
    return list_text_positive, list_text_negative, list_text_neutral

def jsonl2jsonl_filter_sentiment_text_embedding(sentiment_file_path, embedding_file_path, 
                                               output_folder_path="", verbose="yes"):
    """
    Filter sentiment text dengan embeddings dan save ke file
    Menggantikan jsonl2jsonl_filter_sentiment_frame untuk embedding approach
    
    Args:
        sentiment_file_path: path ke file sentiment data
        embedding_file_path: path ke file embedding data  
        output_folder_path: folder output untuk filtered files
        verbose: menampilkan progress
    """
    # Load data
    data_sentiment = read_jsonl(sentiment_file_path)
    data_embedding = read_jsonl(embedding_file_path)
    
    if verbose == "yes":
        print(f"Loading sentiment data from: {sentiment_file_path}")
        print(f"Loading embedding data from: {embedding_file_path}")
        print(f"Sentiment data count: {len(data_sentiment)}")
        print(f"Embedding data count: {len(data_embedding)}")
    
    # Filter berdasarkan sentiment
    list_text_positive, list_text_negative, list_text_neutral = filter_sentiment_text_list(
        data_sentiment, data_embedding
    )
    
    # Prepare output paths
    if output_folder_path != "" and output_folder_path[-1] != "/":
        output_folder_path = f"{output_folder_path}/"
    
    file_name_base = os.path.basename(embedding_file_path).replace('.jsonl', '')
    positive_file_path = f"{output_folder_path}positive-{file_name_base}.jsonl"
    negative_file_path = f"{output_folder_path}negative-{file_name_base}.jsonl"
    neutral_file_path = f"{output_folder_path}neutral-{file_name_base}.jsonl"
    
    # Save filtered data
    write_jsonl(list_text_positive, positive_file_path)
    write_jsonl(list_text_negative, negative_file_path)
    write_jsonl(list_text_neutral, neutral_file_path)
    
    if verbose == "yes":
        print(f"Positive texts saved to: {positive_file_path} ({len(list_text_positive)} items)")
        print(f"Negative texts saved to: {negative_file_path} ({len(list_text_negative)} items)")
        print(f"Neutral texts saved to: {neutral_file_path} ({len(list_text_neutral)} items)")
    
    return {
        'positive': list_text_positive,
        'negative': list_text_negative, 
        'neutral': list_text_neutral,
        'file_paths': {
            'positive': positive_file_path,
            'negative': negative_file_path,
            'neutral': neutral_file_path
        }
    }