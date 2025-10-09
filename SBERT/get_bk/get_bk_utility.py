# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from clusterer.calculate_similarity import calculate_similarity
import pandas as pd

# Function collections to get a specific attribute from string .klp format
def get_list_klp(klp_path):
    with open(klp_path,"r") as file:
        return [line for line in file]

def get_id_from_str_klp(str_klp):
    return str_klp.split('|BS:id| ')[1].split('|ES|')[0].strip()

def get_bk_from_str_klp(str_klp):
    return str_klp.split('|BS:cleaned_bk| ')[1].split('|ES|')[0].strip()

# Function to get bk candidate per polarity
def get_bk_candidate_per_polarity(splitted_frame_klp_path,medoid_klp_path,lib_folder_path,top_n=1):
    list_splitted_frame_klp = get_list_klp(splitted_frame_klp_path)
    list_medoid_klp = get_list_klp(medoid_klp_path)
    list_medoid_bk = [get_bk_from_str_klp(str_klp) for str_klp in list_medoid_klp]
    list_bk_candidate = []
    for i in range(len(list_splitted_frame_klp)):
        id_i = get_id_from_str_klp(list_splitted_frame_klp[i])
        list_sim_score = [calculate_similarity(lib_folder_path,splitted_frame_klp_path,i,medoid_klp_path,j) for j in range(len(list_medoid_bk))]
        top_n_candidate = sorted(zip(list_sim_score, list_medoid_bk), reverse=True)[:top_n]
        top_n_candidate = [(id_i,t[0],t[1]) for t in top_n_candidate]
        list_bk_candidate.extend(top_n_candidate)
    return list_bk_candidate

# Function to get list top_n bk per polarity
def get_bk_per_polarity(list_bk_candidate,bk_polarity,top_n=1):
    # Get list id
    list_id = [i[0] for i in list_bk_candidate]
    list_id = sorted(set(list_id), key=list_id.index)
    # Initial the result container
    list_bk = []
    # Get top_n bk
    for id_text in list_id:
        list_bk_candidate_per_id = [i for i in list_bk_candidate if i[0] == id_text]
        df_bk_candidate_per_id = pd.DataFrame(list_bk_candidate_per_id, columns=["id","sim_score","bk"])
        df_bk_candidate_per_id = df_bk_candidate_per_id.sort_values("sim_score",ascending=False).drop_duplicates("bk",keep="first")
        list_score = list(df_bk_candidate_per_id["sim_score"][:top_n])
        top_n_bk = list(df_bk_candidate_per_id["bk"][:top_n])
        list_bk.append((id_text,bk_polarity,list_score,top_n_bk))
    return list_bk