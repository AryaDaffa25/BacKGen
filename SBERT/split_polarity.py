import json
from collections import defaultdict
import os

# Import BacKGen utilities
from converter.io import read_jsonl, write_jsonl
from converter.converter_utility import spacy_tokenizer
from filterer.filterer_utility import all_indices

def filter_sentiment_polarity_data(data):
    """
    Split data menjadi positive dan negative, dengan drop duplicate berdasarkan ID
    
    Args:
        data: List of dictionaries dari JSONL file
        
    Returns:
        tuple: (positive_data, negative_data)
    """
    positive_data = []
    negative_data = []
    
    # Track unique IDs untuk drop duplicate
    positive_ids = set()
    negative_ids = set()
    
    positive_count = 0
    negative_count = 0
    positive_duplicates = 0
    negative_duplicates = 0
    
    for item in data:
        text_id = item.get("id")
        polarity = item.get("st_polarity")
        
        if polarity == "positive":
            if text_id not in positive_ids:
                positive_data.append(item)
                positive_ids.add(text_id)
                positive_count += 1
            else:
                positive_duplicates += 1
                print(f"Dropping duplicate positive ID: {text_id}")
                
        elif polarity == "negative":
            if text_id not in negative_ids:
                negative_data.append(item)
                negative_ids.add(text_id)
                negative_count += 1
            else:
                negative_duplicates += 1
                print(f"Dropping duplicate negative ID: {text_id}")
    
    print("\n=== FILTERING SUMMARY ===")
    print(f"Unique positive entries: {positive_count}")
    print(f"Unique negative entries: {negative_count}")
    print(f"Positive duplicates dropped: {positive_duplicates}")
    print(f"Negative duplicates dropped: {negative_duplicates}")
    print(f"Total entries processed: {len(data)}")
    
    return positive_data, negative_data

def save_filtered_data(positive_data, negative_data, output_folder="data/filtered_polarity", verbose=True):
    """
    Save filtered data menggunakan format yang sama dengan BacKGen
    
    Args:
        positive_data: List of positive polarity entries
        negative_data: List of negative polarity entries
        output_folder: Output directory path
        verbose: Whether to print save messages
    """
    # Create output directory if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save positive data
    positive_file_path = f"{output_folder}/positive-filtered.jsonl"
    write_jsonl(positive_data, positive_file_path)
    if verbose:
        print(f"Positive data has been saved into {positive_file_path}.")
    
    # Save negative data  
    negative_file_path = f"{output_folder}/negative-filtered.jsonl"
    write_jsonl(negative_data, negative_file_path)
    if verbose:
        print(f"Negative data has been saved into {negative_file_path}.")
    
    return positive_file_path, negative_file_path

def analyze_polarity_distribution(data):
    """
    Analyze distribusi polarity dalam dataset
    
    Args:
        data: List of dictionaries dari JSONL file
    """
    print("\n=== ORIGINAL DATA ANALYSIS ===")
    
    total_entries = len(data)
    unique_ids = set(item.get("id") for item in data)
    polarities = [item.get("st_polarity") for item in data]
    
    polarity_counts = {
        "positive": polarities.count("positive"),
        "negative": polarities.count("negative")
    }
    
    print(f"Total entries: {total_entries}")
    print(f"Unique text IDs: {len(unique_ids)}")
    print(f"Polarity distribution: {polarity_counts}")
    
    # Check for duplicates
    id_counts = defaultdict(int)
    for item in data:
        id_counts[item.get("id")] += 1
    
    duplicates = sum(1 for count in id_counts.values() if count > 1)
    print(f"IDs with duplicates: {duplicates}")

def jsonl2jsonl_filter_polarity(input_file_path, output_folder="data/filtered_polarity", verbose=True):
    """
    Main function untuk filtering polarity dari JSONL file
    Mengikuti pattern yang sama dengan BacKGen filtering functions
    
    Args:
        input_file_path: Path ke input JSONL file
        output_folder: Output directory path
        verbose: Whether to print detailed messages
        
    Returns:
        tuple: (positive_file_path, negative_file_path)
    """
    print(f"Reading data from: {input_file_path}")
    
    # Read data using BacKGen IO utility
    data = read_jsonl(input_file_path)
    
    # Analyze original distribution
    analyze_polarity_distribution(data)
    
    # Filter data berdasarkan polarity
    print("\n=== FILTERING PROCESS ===")
    positive_data, negative_data = filter_sentiment_polarity_data(data)
    
    # Save filtered data
    print("\n=== SAVING FILTERED DATA ===")
    positive_file_path, negative_file_path = save_filtered_data(
        positive_data, negative_data, output_folder, verbose
    )
    
    return positive_file_path, negative_file_path

def split_polarity_data(input_file_path, positive_output_path, negative_output_path):
    """
    Split data berdasarkan polarity dengan parameter file yang spesifik
    
    Args:
        input_file_path: Path ke input JSONL file
        positive_output_path: Path untuk output file positive
        negative_output_path: Path untuk output file negative
    
    Returns:
        tuple: (positive_count, negative_count)
    """
    print(f"Reading data from: {input_file_path}")
    
    # Read data using BacKGen IO utility
    data = read_jsonl(input_file_path)
    
    # Analyze original distribution
    analyze_polarity_distribution(data)
    
    # Filter data berdasarkan polarity
    print("\n=== SPLITTING PROCESS ===")
    positive_data, negative_data = filter_sentiment_polarity_data(data)
    
    # Create output directories if not exist
    os.makedirs(os.path.dirname(positive_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(negative_output_path), exist_ok=True)
    
    # Save positive data
    write_jsonl(positive_data, positive_output_path)
    print(f"Positive data saved to: {positive_output_path}")
    
    # Save negative data  
    write_jsonl(negative_data, negative_output_path)
    print(f"Negative data saved to: {negative_output_path}")
    
    return len(positive_data), len(negative_data)

def main():
    """
    Main function dengan parameter file yang sudah ditentukan
    """
    # File paths sesuai spesifikasi
    input_file = "data/fold_1-test-spc.jsonl"
    positive_output = "data/output/polarity_positive.jsonl"
    negative_output = "data/output/polarity_negative.jsonl"
    
    try:
        # Run splitting process
        positive_count, negative_count = split_polarity_data(
            input_file_path=input_file,
            positive_output_path=positive_output,
            negative_output_path=negative_output
        )
        
        print(f"\n=== PROCESS COMPLETED ===")
        print(f"Positive entries: {positive_count} -> {positive_output}")  
        print(f"Negative entries: {negative_count} -> {negative_output}")
        print(f"Total processed: {positive_count + negative_count}")
        
        print("\nNext steps for BacKGen adaptation:")
        print("1. Generate SBERT embeddings untuk setiap dataset")
        print("2. Apply K-means clustering dengan k = data_count / 5")
        print("3. Proceed dengan BK generation process")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure the input file exists: data/fold-1-test_spc.jsonl")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()