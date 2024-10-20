# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import library
from converter.io import read_jsonl,write_jsonl

# Function to check intersection
def all_indices(list_, item):  
    return [index for index, value in enumerate(list_) if value == item]  

# Filter singleton
def remove_singleton_jsonl2jsonl(input_file_path,output_file_path):
    js_data = read_jsonl(input_file_path)
    list_removed_singleton = []
    for js in js_data:
        if len(js.get("list_frame_symbolic")) != 1:
            list_removed_singleton.append(js)
    write_jsonl(list_removed_singleton,output_file_path)
    print(f"Removed singleton data has been saved into {output_file_path}")