# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import library
from converter.io import read_jsonl,write_jsonl

def filter_sentiment_splitted_st(splitted_st_input_path,negative_output_path,positive_output_path):
    js_splitted_st = read_jsonl(splitted_st_input_path)
    js_negative = [js for js in js_splitted_st if js.get('st_polarity') == "negative"]
    js_positive = [js for js in js_splitted_st if js.get('st_polarity') == "positive"]
    write_jsonl(js_negative,negative_output_path)
    write_jsonl(js_positive,positive_output_path)