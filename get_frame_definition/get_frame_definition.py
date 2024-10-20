# Add library path
import sys
sys.path.append("/Users/okkyibrohim/github-okky/reveal-internship")

# Import libraries
from converter.io import read_jsonl, write_jsonl
from nltk.corpus import framenet as fn

def get_frame_list(js):
    list_frame = []
    list_frame_symbolic = js.get('list_frame_symbolic')
    for fs in list_frame_symbolic:
        list_frame.append(fs.split('(')[0])
    return list(dict.fromkeys(list_frame))

def get_frame_definition(input_file_path,output_file_path,with_full_definition="yes",return_result="yes"):
    js_data = read_jsonl(input_file_path)
    for js_idx in range(len(js_data)):
        list_frame = get_frame_list(js_data[js_idx])
        list_short_definition,list_full_definition = [],[]
        for frame in list_frame:
            try:
                full_definition = fn.frame(frame).definition
                short_definition = full_definition.split(".")[0]+"."
                list_short_definition.append((frame,short_definition))
                if with_full_definition == "yes":
                    list_full_definition.append((frame,full_definition))
            except:
                continue
        js_data[js_idx]["list_frame_short_definition"] = list_short_definition
        if with_full_definition == "yes":
            js_data[js_idx]["list_frame_full_definition"] = list_full_definition
    write_jsonl(js_data,output_file_path)
    if return_result == "yes":
        return js_data