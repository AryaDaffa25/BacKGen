# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from converter.io import read_jsonl, write_jsonl

def get_list_sim_score_from_txt(txt_file_path):
    list_sim_score = []
    with open(txt_file_path,"r") as txt_file:
        for line in txt_file:
            sim_score = line.split()
            sim_score = [float(score) for score in sim_score]
            list_sim_score.append(sim_score)
    return list_sim_score

def get_list_pk(pk_jsonl_path,pk_attribute_name="cleaned_answer"):
    js_pk = read_jsonl(pk_jsonl_path)
    return [js.get(pk_attribute_name) for js in js_pk]

def get_top_n_pk_for_each_frame(js_splitted_frame,list_pk,list_sim_score,pk_polarity,top_n=1):
    for i in range(len(js_splitted_frame)):
        list_sim_score_i = list_sim_score[i]
        top_n_candidate = sorted(zip(list_sim_score_i, list_pk), reverse=True)[:top_n]
        js_splitted_frame[i][f"{pk_polarity}_top_n_pk_sim_score"] = [t[0] for t in top_n_candidate]
        js_splitted_frame[i][f"{pk_polarity}_top_n_pk"] = [t[1] for t in top_n_candidate]
    return js_splitted_frame

def get_top_n_pk_for_each_id_each_polarity(js_id,js_splitted_frame,splitted_st="yes"):
    if splitted_st == "yes":
        js_splitted_frame_id = [js for js in js_splitted_frame if str(js.get("st_id")) == js_id]
    else:
        js_splitted_frame_id = [js for js in js_splitted_frame if str(js.get("id")) == js_id]
    if js_splitted_frame_id == []:
        list_all_score_each_frame = "No background knowledge can be provided."
        list_all_pk_each_frame = "No background knowledge can be provided."
        list_best_score_each_frame = "No background knowledge can be provided."
        list_best_pk_each_frame = "No background knowledge can be provided."
        return list_all_score_each_frame,list_all_pk_each_frame,list_best_score_each_frame,list_best_pk_each_frame
    else:
        list_negative_pk_sim_score,list_negative_pk,list_positive_pk_sim_score,list_positive_pk,list_all_score_each_frame,list_all_pk_each_frame,list_best_score_each_polarity,list_best_pk_each_polarity = [],[],[],[],[],[],[],[]
        for js in js_splitted_frame_id:
            list_negative_pk_sim_score.extend(js.get(f"negative_top_n_pk_sim_score"))
            list_negative_pk.extend(js.get(f"negative_top_n_pk"))
            list_positive_pk_sim_score.extend(js.get(f"positive_top_n_pk_sim_score"))
            list_positive_pk.extend(js.get(f"positive_top_n_pk"))
        list_all_score_each_frame.extend(list_negative_pk_sim_score)
        list_all_score_each_frame.extend(list_positive_pk_sim_score)
        list_all_pk_each_frame.extend(list_negative_pk)
        list_all_pk_each_frame.extend(list_positive_pk)
        tuple_best_negative = sorted(zip(list_negative_pk_sim_score, list_negative_pk), reverse=True)[:1]
        tuple_best_positive = sorted(zip(list_positive_pk_sim_score, list_positive_pk), reverse=True)[:1]
        list_best_score_each_polarity.append(tuple_best_negative[0][0])
        list_best_score_each_polarity.append(tuple_best_positive[0][0])
        list_best_pk_each_polarity.append(tuple_best_negative[0][1])
        list_best_pk_each_polarity.append(tuple_best_positive[0][1])
        return list_all_score_each_frame,list_all_pk_each_frame,list_best_score_each_polarity,list_best_pk_each_polarity
    
# Function to get pk from all polarity
def get_pk_stc(ssa_input_jsonl_path,ssa_output_jsonl_path,splitted_frame_jsonl_path,negative_sim_score_path,negative_pk_jsonl_path,positive_sim_score_path,positive_pk_jsonl_path,splitted_st="yes",pk_attribute_name="cleaned_answer",return_result="yes"):
    # Load ssa and splitted_frame data
    js_ssa = read_jsonl(ssa_input_jsonl_path)
    js_splitted_frame = read_jsonl(splitted_frame_jsonl_path)
    # Get pk for each polarity each frame
    print("Processing to get pk of negative polarity for each frame.")
    negative_sim_score = get_list_sim_score_from_txt(negative_sim_score_path)
    negative_pk = get_list_pk(negative_pk_jsonl_path,pk_attribute_name)
    js_splitted_frame = get_top_n_pk_for_each_frame(js_splitted_frame,negative_pk,negative_sim_score,"negative")
    print("Processing to get pk of positive polarity for each frame.")
    positive_sim_score = get_list_sim_score_from_txt(positive_sim_score_path)
    positive_pk = get_list_pk(positive_pk_jsonl_path,pk_attribute_name)
    js_splitted_frame = get_top_n_pk_for_each_frame(js_splitted_frame,positive_pk,positive_sim_score,"positive")
    # Add pk to the ssa dataset
    for i in range(len(js_ssa)):
        if splitted_st == "yes":
            js_id = str(js_ssa[i].get("st_id"))
        else:
            js_id = str(js_ssa[i].get("id"))
        list_all_score_each_frame,list_all_pk_each_frame,list_best_score_each_polarity,list_best_pk_each_polarity = get_top_n_pk_for_each_id_each_polarity(js_id,js_splitted_frame)
        js_ssa[i]["list_all_score_each_frame"] = list_all_score_each_frame
        js_ssa[i]["list_all_pk_each_frame"] = list_all_pk_each_frame
        js_ssa[i]["list_best_score_each_polarity"] = list_best_score_each_polarity
        js_ssa[i]["list_best_pk_each_polarity"] = list_best_pk_each_polarity
    # Write to jsonl
    write_jsonl(js_ssa,ssa_output_jsonl_path)
    # Return the js_ssa if needed
    if return_result == "yes":
        return js_ssa