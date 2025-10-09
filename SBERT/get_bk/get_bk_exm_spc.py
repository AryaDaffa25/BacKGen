# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import library
from converter.io import read_jsonl, write_jsonl

##################################
# Function collections to get bk #
##################################
def get_list_sim_score_from_txt(txt_file_path):
    list_sim_score = []
    with open(txt_file_path,"r") as txt_file:
        for line in txt_file:
            sim_score = line.split()
            sim_score = [float(score) for score in sim_score]
            list_sim_score.append(sim_score)
    return list_sim_score

def get_list_bk(bk_jsonl_path,bk_attribute_name="cleaned_answer"):
    js_bk = read_jsonl(bk_jsonl_path)
    return [js.get(bk_attribute_name) for js in js_bk]

def get_top_n_bk_for_each_frame(js_splitted_frame,list_bk,list_sim_score,bk_polarity,top_n=1):
    for i in range(len(js_splitted_frame)):
        list_sim_score_i = list_sim_score[i]
        top_n_candidate = sorted(zip(list_sim_score_i, list_bk), reverse=True)[:top_n]
        js_splitted_frame[i][f"{bk_polarity}_top_n_bk_sim_score"] = [t[0] for t in top_n_candidate]
        js_splitted_frame[i][f"{bk_polarity}_top_n_bk"] = [t[1] for t in top_n_candidate]
    return js_splitted_frame

def get_top_n_bk_for_each_id_each_polarity(js_id,js_splitted_frame,splitted_st="yes"):
    if splitted_st == "yes":
        js_splitted_frame_id = [js for js in js_splitted_frame if str(js.get("st_id")) == js_id]
    else:
        js_splitted_frame_id = [js for js in js_splitted_frame if str(js.get("id")) == js_id]
    if js_splitted_frame_id == []:
        list_all_score_each_frame = "No background knowledge can be provided."
        list_all_bk_each_frame = "No background knowledge can be provided."
        list_best_score_each_frame = "No background knowledge can be provided."
        list_best_bk_each_frame = "No background knowledge can be provided."
        return list_all_score_each_frame,list_all_bk_each_frame,list_best_score_each_frame,list_best_bk_each_frame
    else:
        list_negative_bk_sim_score,list_negative_bk,list_positive_bk_sim_score,list_positive_bk,list_all_score_each_frame,list_all_bk_each_frame,list_best_score_each_polarity,list_best_bk_each_polarity = [],[],[],[],[],[],[],[]
        for js in js_splitted_frame_id:
            list_negative_bk_sim_score.extend(js.get(f"negative_top_n_bk_sim_score"))
            list_negative_bk.extend(js.get(f"negative_top_n_bk"))
            list_positive_bk_sim_score.extend(js.get(f"positive_top_n_bk_sim_score"))
            list_positive_bk.extend(js.get(f"positive_top_n_bk"))
        list_all_score_each_frame.extend(list_negative_bk_sim_score)
        list_all_score_each_frame.extend(list_positive_bk_sim_score)
        list_all_bk_each_frame.extend(list_negative_bk)
        list_all_bk_each_frame.extend(list_positive_bk)
        tuple_best_negative = sorted(zip(list_negative_bk_sim_score, list_negative_bk), reverse=True)[:1]
        tuple_best_positive = sorted(zip(list_positive_bk_sim_score, list_positive_bk), reverse=True)[:1]
        list_best_score_each_polarity.append(tuple_best_negative[0][0])
        list_best_score_each_polarity.append(tuple_best_positive[0][0])
        list_best_bk_each_polarity.append(tuple_best_negative[0][1])
        list_best_bk_each_polarity.append(tuple_best_positive[0][1])
        return list_all_score_each_frame,list_all_bk_each_frame,list_best_score_each_polarity,list_best_bk_each_polarity
#########################################
# End of function collections to get bk #
#########################################

########################################################
# Function collections to get example form selected bk #
########################################################
def get_list_bk_mti(bk_jsonl_path,bk_mti_attribute_name="medoid_text_id"):
    js_bk = read_jsonl(bk_jsonl_path)
    return [js.get(bk_mti_attribute_name) for js in js_bk]

def get_top_n_bk_mti_for_each_frame(js_splitted_frame,list_bk_mti,list_sim_score,bk_polarity,top_n=1):
    for i in range(len(js_splitted_frame)):
        list_sim_score_i = list_sim_score[i]
        top_n_candidate = sorted(zip(list_sim_score_i, list_bk_mti), reverse=True)[:top_n]
        js_splitted_frame[i][f"{bk_polarity}_top_n_bk_mti"] = [t[1] for t in top_n_candidate]
    return js_splitted_frame

def get_list_example(list_bk_mti,js_ex):
    list_selected_ex = []
    for mti in list_bk_mti:
        js_selected_ex = [js for js in js_ex if str(js.get("id")) == mti][0]
        list_selected_ex.append((js_selected_ex.get("text"),js_selected_ex.get("st_span"),js_selected_ex.get("st_polarity")))
    return list_selected_ex

def get_top_n_bk_mti_for_each_id_each_polarity(js_id,js_splitted_frame,js_negative_ex,js_positve_ex,splitted_st="yes"):
    if splitted_st == "yes":
        js_splitted_frame_id = [js for js in js_splitted_frame if str(js.get("st_id")) == js_id]
    else:
        js_splitted_frame_id = [js for js in js_splitted_frame if str(js.get("id")) == js_id]
    if js_splitted_frame_id == []:
        list_negative_mti = "No example can be provided."
        list_positive_mti = "No example can be provided." 
        list_negative_example_mti = "No example can be provided."
        list_positive_example_mti = "No example can be provided."
        return list_negative_mti,list_positive_mti,list_negative_example_mti,list_positive_example_mti
    else:
        list_negative_mti,list_positive_mti = [],[]
        for js in js_splitted_frame_id:
            list_negative_mti.extend(js.get(f"negative_top_n_bk_mti"))
            list_positive_mti.extend(js.get(f"positive_top_n_bk_mti"))
        list_negative_example_mti = get_list_example(list_negative_mti,js_negative_ex)
        list_positive_example_mti = get_list_example(list_positive_mti,js_positve_ex)
        return list_negative_mti,list_positive_mti,list_negative_example_mti,list_positive_example_mti
###############################################################
# End of function collections to get example form selected bk #
###############################################################
    
# Function to get bk and example from all polarity
def get_bk_exm_spc(ssa_input_jsonl_path,ssa_output_jsonl_path,splitted_frame_jsonl_path,negative_sim_score_path,negative_bk_jsonl_path,positive_sim_score_path,positive_bk_jsonl_path,splitted_st_negative_ex_jsonl_path,splitted_st_positive_ex_jsonl_path,splitted_st="yes",bk_attribute_name="cleaned_answer",bk_mti_attribute_name="medoid_text_id",return_result="yes"):
    # Load ssa and splitted_frame data
    js_ssa = read_jsonl(ssa_input_jsonl_path)
    js_splitted_frame = read_jsonl(splitted_frame_jsonl_path)

    # Get bk for each polarity each frame
    # Get bk of negative polarity
    print("Processing to get bk of negative polarity for each frame.")
    negative_sim_score = get_list_sim_score_from_txt(negative_sim_score_path)
    negative_bk = get_list_bk(negative_bk_jsonl_path,bk_attribute_name)
    js_splitted_frame = get_top_n_bk_for_each_frame(js_splitted_frame,negative_bk,negative_sim_score,"negative")
    # Get bk of positive polarity
    print("Processing to get bk of positive polarity for each frame.")
    positive_sim_score = get_list_sim_score_from_txt(positive_sim_score_path)
    positive_bk = get_list_bk(positive_bk_jsonl_path,bk_attribute_name)
    js_splitted_frame = get_top_n_bk_for_each_frame(js_splitted_frame,positive_bk,positive_sim_score,"positive")

    # Get example for each polarity each frame
    # Load example from trainset for each polarity
    js_negative_ex = read_jsonl(splitted_st_negative_ex_jsonl_path)
    js_positve_ex = read_jsonl(splitted_st_positive_ex_jsonl_path)
    # Get example of negative polarity
    print("Processing to get example of negative polarity for each frame.")
    negative_bk_mti = get_list_bk_mti(negative_bk_jsonl_path,bk_mti_attribute_name)
    js_splitted_frame = get_top_n_bk_mti_for_each_frame(js_splitted_frame,negative_bk_mti,negative_sim_score,"negative")
    # Get example of positive polarity
    print("Processing to get example of positive polarity for each frame.")
    positive_bk_mti = get_list_bk_mti(positive_bk_jsonl_path,bk_mti_attribute_name)
    js_splitted_frame = get_top_n_bk_mti_for_each_frame(js_splitted_frame,positive_bk_mti,positive_sim_score,"positive")

    # Add bk and example to the spc dataset
    for i in range(len(js_ssa)):
        if splitted_st == "yes":
            js_id = str(js_ssa[i].get("st_id"))
        else:
            js_id = str(js_ssa[i].get("id"))
        # Get bk and add the spc dataset
        list_all_score_each_frame,list_all_bk_each_frame,list_best_score_each_polarity,list_best_bk_each_polarity = get_top_n_bk_for_each_id_each_polarity(js_id,js_splitted_frame)
        js_ssa[i]["list_all_score_each_frame"] = list_all_score_each_frame
        js_ssa[i]["list_all_bk_each_frame"] = list_all_bk_each_frame
        js_ssa[i]["list_best_score_each_polarity"] = list_best_score_each_polarity
        js_ssa[i]["list_best_bk_each_polarity"] = list_best_bk_each_polarity
        # Get example and add the spc dataset
        list_negative_mti,list_positive_mti,list_negative_example_mti,list_positive_example_mti = get_top_n_bk_mti_for_each_id_each_polarity(js_id,js_splitted_frame,js_negative_ex,js_positve_ex)
        js_ssa[i]["list_negative_mti"] = list_negative_mti
        js_ssa[i]["list_positive_mti"] = list_positive_mti
        js_ssa[i]["list_negative_example_mti"] = list_negative_example_mti
        js_ssa[i]["list_positive_example_mti"] = list_positive_example_mti
        
    # Write to jsonl
    write_jsonl(js_ssa,ssa_output_jsonl_path)
    # Return the js_ssa if needed
    if return_result == "yes":
        return js_ssa