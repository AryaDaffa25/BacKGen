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

def get_pk_from_str_klp(str_klp):
    return str_klp.split('|BS:cleaned_pk| ')[1].split('|ES|')[0].strip()

# Function to get pk candidate per polarity
def get_pk_candidate_per_polarity(splitted_frame_klp_path,medoid_klp_path,lib_folder_path,top_n=1):
    list_splitted_frame_klp = get_list_klp(splitted_frame_klp_path)
    list_medoid_klp = get_list_klp(medoid_klp_path)
    list_medoid_pk = [get_pk_from_str_klp(str_klp) for str_klp in list_medoid_klp]
    list_pk_candidate = []
    for i in range(len(list_splitted_frame_klp)):
        id_i = get_id_from_str_klp(list_splitted_frame_klp[i])
        list_sim_score = [calculate_similarity(lib_folder_path,splitted_frame_klp_path,i,medoid_klp_path,j) for j in range(len(list_medoid_pk))]
        top_n_candidate = sorted(zip(list_sim_score, list_medoid_pk), reverse=True)[:top_n]
        top_n_candidate = [(id_i,t[0],t[1]) for t in top_n_candidate]
        list_pk_candidate.extend(top_n_candidate)
    return list_pk_candidate

# Function to get list top_n pk per polarity
def get_pk_per_polarity(list_pk_candidate,pk_polarity,top_n=1):
    # Get list id
    list_id = [i[0] for i in list_pk_candidate]
    list_id = sorted(set(list_id), key=list_id.index)
    # Initial the result container
    list_pk = []
    # Get top_n pk
    for id_text in list_id:
        list_pk_candidate_per_id = [i for i in list_pk_candidate if i[0] == id_text]
        df_pk_candidate_per_id = pd.DataFrame(list_pk_candidate_per_id, columns=["id","sim_score","pk"])
        df_pk_candidate_per_id = df_pk_candidate_per_id.sort_values("sim_score",ascending=False).drop_duplicates("pk",keep="first")
        list_score = list(df_pk_candidate_per_id["sim_score"][:top_n])
        top_n_pk = list(df_pk_candidate_per_id["pk"][:top_n])
        list_pk.append((id_text,pk_polarity,list_score,top_n_pk))
    return list_pk