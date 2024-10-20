# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from sentence_transformers import SentenceTransformer,SimilarityFunction
from converter.io import read_jsonl, write_jsonl
import pandas as pd

def get_sim_score(list_attribute_instance,list_attribute_bk,bert_model,sim_function):
    if sim_function == "cosine":
        model = SentenceTransformer(bert_model, similarity_fn_name=SimilarityFunction.COSINE)
    elif sim_function == "dot":
        model = SentenceTransformer(bert_model, similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
    elif sim_function == "euclidean":
        model = SentenceTransformer(bert_model, similarity_fn_name=SimilarityFunction.EUCLIDEAN)
    elif sim_function == "manhattan":
        model = SentenceTransformer(bert_model, similarity_fn_name=SimilarityFunction.MANHATTAN)
    else:
        raise ValueError("Wrong 'sim_function' parameter. Only 'cosine', 'dot', euclidean', or 'manhattan' is allowed.")
    embeddings1 = model.encode(list_attribute_instance)
    embeddings2 = model.encode(list_attribute_bk)
    # Compute cosine similarities
    return model.similarity(embeddings1, embeddings2).tolist()

def mean_sim_score(list_text_instance,list_st_span_instance,list_text_bk,list_st_span_bk,bert_model,sim_function):
    sim_score_1 = get_sim_score(list_text_instance,list_text_bk,bert_model,sim_function)
    sim_score_2 = get_sim_score(list_st_span_instance,list_st_span_bk,bert_model,sim_function)
    return [[(g + h) / 2 for g, h in zip(a, b)] for a,b in zip(sim_score_1,sim_score_2)]

def get_bk_attribute(bk_jsonl_path,ex_jsonl_path):
    js_bk,js_ex = read_jsonl(bk_jsonl_path),read_jsonl(ex_jsonl_path)
    list_cluster_id = [js.get('cluster_id') for js in js_bk]
    list_medoid_text_id = [js.get('medoid_text_id') for js in js_bk]
    list_bk = [js.get('cleaned_answer') for js in js_bk]
    list_bk_text,list_bk_st_span,list_bk_st_polarity = [],[],[]
    for mti in list_medoid_text_id:
        ex = [js for js in js_ex if str(js.get('id')) == mti][0]
        list_bk_text.append(ex.get('text'))
        list_bk_st_span.append(ex.get('st_span'))
        list_bk_st_polarity.append(ex.get('st_polarity'))
    bk_attribute =  pd.DataFrame({"list_cluster_id":list_cluster_id,
                         "list_medoid_text_id":list_medoid_text_id,
                         "list_bk":list_bk,
                         "list_bk_text":list_bk_text,
                         "list_bk_st_span":list_bk_st_span,
                         "list_bk_st_polarity":list_bk_st_polarity})
    return bk_attribute

def get_bk_ex(bk_attribute,instance_sim_score,top_n,drop_bad_bk,drop_duplicate_text):
    bk_attribute["instance_sim_score"] = instance_sim_score
    if drop_bad_bk == "yes":
        bk_attribute = bk_attribute.drop(bk_attribute[bk_attribute.list_bk == "1."].index)
    bk_attribute = bk_attribute.sort_values(by=['instance_sim_score'], ascending=[False])
    if drop_duplicate_text == "yes":
        bk_attribute = bk_attribute.drop_duplicates(subset=['list_bk_text'], keep="first")
    bk_attribute = bk_attribute.reset_index(drop=True)
    selected_bk_score,selected_bk,selected_id,selected_ex = [],[],[],[]
    for i in range(0,top_n):
        selected_bk_score.append(bk_attribute.instance_sim_score[i])
        selected_bk.append(bk_attribute.list_bk[i])
        selected_id.append(bk_attribute.list_medoid_text_id[i])
        selected_ex.append((bk_attribute.list_bk_text[i],bk_attribute.list_bk_st_span[i],bk_attribute.list_bk_st_polarity[i]))
    return selected_bk_score,selected_bk,selected_id,selected_ex

def get_bk_exmbert_spc(ssa_input_jsonl_path,ssa_output_jsonl_path,negative_bk_jsonl_path,positive_bk_jsonl_path,splitted_st_negative_ex_jsonl_path,splitted_st_positive_ex_jsonl_path,sim_attribute,bert_model="all-MiniLM-L6-v2",sim_function="cosine",top_n=2,drop_bad_bk="yes",drop_duplicate_text="yes",return_result="no"):
    # Get ssa attribute
    js_ssa = read_jsonl(ssa_input_jsonl_path)
    list_ssa_text = [js.get('text') for js in js_ssa]
    list_ssa_st_span = [js.get('st_span') for js in js_ssa]
    # Get bk attribute
    neg_bk_attribute = get_bk_attribute(negative_bk_jsonl_path,splitted_st_negative_ex_jsonl_path)
    pos_bk_attribute = get_bk_attribute(positive_bk_jsonl_path,splitted_st_positive_ex_jsonl_path)
    # Get sim score
    if sim_attribute == "text":
        neg_sim_score = get_sim_score(list_ssa_text,neg_bk_attribute.list_bk_text,bert_model,sim_function)
        pos_sim_score = get_sim_score(list_ssa_text,pos_bk_attribute.list_bk_text,bert_model,sim_function)
    elif sim_attribute == "st_span":
        neg_sim_score = get_sim_score(list_ssa_st_span,neg_bk_attribute.list_bk_st_span,bert_model,sim_function)
        pos_sim_score = get_sim_score(list_ssa_st_span,pos_bk_attribute.list_bk_st_span,bert_model,sim_function)
    elif sim_attribute == "average":
        neg_sim_score = mean_sim_score(list_ssa_text,list_ssa_st_span,neg_bk_attribute.list_bk_text,neg_bk_attribute.list_bk_st_span,bert_model,sim_function)
        pos_sim_score = mean_sim_score(list_ssa_text,list_ssa_st_span,pos_bk_attribute.list_bk_text,pos_bk_attribute.list_bk_st_span,bert_model,sim_function)
    else:
        raise ValueError("Wrong 'sim_attribute' parameter. Only 'text', 'st_span', or 'average' is allowed.")
    # Add bk and example to the spc dataset
    for i in range(len(js_ssa)):
        # Get bk and example for instance i
        neg_bk_score,neg_bk,neg_id,neg_ex = get_bk_ex(neg_bk_attribute,neg_sim_score[i],top_n,drop_bad_bk,drop_duplicate_text)
        pos_bk_score,pos_bk,pos_id,pos_ex = get_bk_ex(pos_bk_attribute,pos_sim_score[i],top_n,drop_bad_bk,drop_duplicate_text)
        js_ssa[i][f"bert_{sim_attribute}_list_all_score_each_frame"] = neg_bk_score+pos_bk_score
        js_ssa[i][f"bert_{sim_attribute}_list_all_bk_each_frame"] = neg_bk+pos_bk
        js_ssa[i][f"bert_{sim_attribute}_list_best_score_each_polarity"] = neg_bk_score[:1]+pos_bk_score[:1]
        js_ssa[i][f"bert_{sim_attribute}_list_best_bk_each_polarity"] = neg_bk[:1]+pos_bk[:1]
        js_ssa[i][f"bert_{sim_attribute}_list_negative_mti"] = neg_id
        js_ssa[i][f"bert_{sim_attribute}_list_positive_mti"] = pos_id
        js_ssa[i][f"bert_{sim_attribute}_list_negative_example_mti"] = neg_ex
        js_ssa[i][f"bert_{sim_attribute}_list_positive_example_mti"] = pos_ex
    # Write to jsonl
    write_jsonl(js_ssa,ssa_output_jsonl_path)
    # Return the js_ssa if needed
    if return_result == "yes":
        return js_ssa