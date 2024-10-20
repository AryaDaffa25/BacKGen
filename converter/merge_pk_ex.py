# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import library
from converter.io import read_jsonl,write_jsonl

def merge_pk_ex(pk_input_file_path,ex_input_file_path,output_file_path):
    js_merged = read_jsonl(pk_input_file_path)
    js_ex = read_jsonl(ex_input_file_path)
    for js_idx in range(len(js_merged)):
        js_merged[js_idx]["list_negative_example_random"] = js_ex[js_idx].get('list_negative_example_random')
        js_merged[js_idx]["list_positive_example_random"] = js_ex[js_idx].get('list_positive_example_random')
    write_jsonl(js_merged,output_file_path)
    print(f"The merged file of pk and example has been saved into {output_file_path}")