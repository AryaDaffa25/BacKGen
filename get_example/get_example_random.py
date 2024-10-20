from converter.io import read_jsonl,write_jsonl
import random

def get_example_random(input_file_path,output_file_path,negative_example_path,positive_example_path,return_result="yes"):
    js_ssa = read_jsonl(input_file_path)
    negative_example = read_jsonl(negative_example_path)
    positive_example = read_jsonl(positive_example_path)
    # Get random example for each data
    for js_idx in range(len(js_ssa)):
        chosen_negative_example,chosen_positive_example = [],[]
        # Collect negative example randomly
        negative_1 = random.choice(negative_example)
        chosen_negative_example.append((negative_1.get("text"),negative_1.get("st_span"),negative_1.get("st_polarity")))
        negative_2 = random.choice(negative_example)
        while negative_1.get("id") == negative_2.get("id"):
            negative_2 = random.choice(negative_example)
        chosen_negative_example.append((negative_2.get("text"),negative_2.get("st_span"),negative_2.get("st_polarity")))
        # Collect positive example randomly
        positive_1 = random.choice(positive_example)
        chosen_positive_example.append((positive_1.get("text"),positive_1.get("st_span"),positive_1.get("st_polarity")))
        positive_2 = random.choice(positive_example)
        while positive_1.get("id") == positive_2.get("id"):
            positive_2 = random.choice(positive_example)
        chosen_positive_example.append((positive_2.get("text"),positive_2.get("st_span"),positive_2.get("st_polarity")))
        js_ssa[js_idx]["list_negative_example_random"] = chosen_negative_example
        js_ssa[js_idx]["list_positive_example_random"] = chosen_positive_example
    write_jsonl(js_ssa,output_file_path)
    if return_result == "yes":
        return js_ssa

