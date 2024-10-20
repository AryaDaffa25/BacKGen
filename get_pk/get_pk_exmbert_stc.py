# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from sentence_transformers import SentenceTransformer,SimilarityFunction
from converter.io import read_jsonl, write_jsonl
import pandas as pd

def get_sim_score(list_attribute_instance,list_attribute_pk,bert_model,sim_function):
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
    embeddings2 = model.encode(list_attribute_pk)
    # Compute cosine similarities
    return model.similarity(embeddings1, embeddings2).tolist()

def mean_sim_score(list_text_instance,list_st_span_instance,list_text_pk,list_st_span_pk,bert_model,sim_function):
    sim_score_1 = get_sim_score(list_text_instance,list_text_pk,bert_model,sim_function)
    sim_score_2 = get_sim_score(list_st_span_instance,list_st_span_pk,bert_model,sim_function)
    return [[(g + h) / 2 for g, h in zip(a, b)] for a,b in zip(sim_score_1,sim_score_2)]

def get_pk_attribute(pk_jsonl_path,ex_jsonl_path):
    js_pk,js_ex = read_jsonl(pk_jsonl_path),read_jsonl(ex_jsonl_path)
    list_cluster_id = [js.get('cluster_id') for js in js_pk]
    list_medoid_text_id = [js.get('medoid_text_id') for js in js_pk]
    list_pk = [js.get('cleaned_answer') for js in js_pk]
    list_pk_text,list_pk_st_span,list_pk_st_polarity = [],[],[]
    for mti in list_medoid_text_id:
        ex = [js for js in js_ex if str(js.get('id')) == mti][0]
        list_pk_text.append(ex.get('text'))
        list_pk_st_span.append(ex.get('st_span'))
        list_pk_st_polarity.append(ex.get('st_polarity'))
    pk_attribute =  pd.DataFrame({"list_cluster_id":list_cluster_id,
                         "list_medoid_text_id":list_medoid_text_id,
                         "list_pk":list_pk,
                         "list_pk_text":list_pk_text,
                         "list_pk_st_span":list_pk_st_span,
                         "list_pk_st_polarity":list_pk_st_polarity})
    return pk_attribute

def get_pk_ex(pk_attribute,instance_sim_score,top_n,drop_bad_pk,drop_duplicate_text):
    pk_attribute["instance_sim_score"] = instance_sim_score
    if drop_bad_pk == "yes":
        pk_attribute = pk_attribute.drop(pk_attribute[pk_attribute.list_pk == "1."].index)
    pk_attribute = pk_attribute.sort_values(by=['instance_sim_score'], ascending=[False])
    if drop_duplicate_text == "yes":
        pk_attribute = pk_attribute.drop_duplicates(subset=['list_pk_text'], keep="first")
    pk_attribute = pk_attribute.reset_index(drop=True)
    selected_pk_score,selected_pk,selected_id,selected_ex = [],[],[],[]
    for i in range(0,top_n):
        selected_pk_score.append(pk_attribute.instance_sim_score[i])
        selected_pk.append(pk_attribute.list_pk[i])
        selected_id.append(pk_attribute.list_medoid_text_id[i])
        selected_ex.append((pk_attribute.list_pk_text[i],pk_attribute.list_pk_st_span[i],pk_attribute.list_pk_st_polarity[i]))
    return selected_pk_score,selected_pk,selected_id,selected_ex

def get_pk_exmbert_stc(ssa_input_jsonl_path,ssa_output_jsonl_path,negative_pk_jsonl_path,positive_pk_jsonl_path,splitted_st_negative_ex_jsonl_path,splitted_st_positive_ex_jsonl_path,sim_attribute,bert_model="all-MiniLM-L6-v2",sim_function="cosine",top_n=2,drop_bad_pk="yes",drop_duplicate_text="yes",return_result="no"):
    # Get ssa attribute
    js_ssa = read_jsonl(ssa_input_jsonl_path)
    list_ssa_text = [js.get('text') for js in js_ssa]
    list_ssa_st_span = [js.get('st_span') for js in js_ssa]
    # Get pk attribute
    neg_pk_attribute = get_pk_attribute(negative_pk_jsonl_path,splitted_st_negative_ex_jsonl_path)
    pos_pk_attribute = get_pk_attribute(positive_pk_jsonl_path,splitted_st_positive_ex_jsonl_path)
    # Get sim score
    if sim_attribute == "text":
        neg_sim_score = get_sim_score(list_ssa_text,neg_pk_attribute.list_pk_text,bert_model,sim_function)
        pos_sim_score = get_sim_score(list_ssa_text,pos_pk_attribute.list_pk_text,bert_model,sim_function)
    elif sim_attribute == "st_span":
        neg_sim_score = get_sim_score(list_ssa_st_span,neg_pk_attribute.list_pk_st_span,bert_model,sim_function)
        pos_sim_score = get_sim_score(list_ssa_st_span,pos_pk_attribute.list_pk_st_span,bert_model,sim_function)
    elif sim_attribute == "average":
        neg_sim_score = mean_sim_score(list_ssa_text,list_ssa_st_span,neg_pk_attribute.list_pk_text,neg_pk_attribute.list_pk_st_span,bert_model,sim_function)
        pos_sim_score = mean_sim_score(list_ssa_text,list_ssa_st_span,pos_pk_attribute.list_pk_text,pos_pk_attribute.list_pk_st_span,bert_model,sim_function)
    else:
        raise ValueError("Wrong 'sim_attribute' parameter. Only 'text', 'st_span', or 'average' is allowed.")
    # Add pk and example to the splitted_stc dataset
    for i in range(len(js_ssa)):
        # Get pk and example for instance i
        neg_pk_score,neg_pk,neg_id,neg_ex = get_pk_ex(neg_pk_attribute,neg_sim_score[i],top_n,drop_bad_pk,drop_duplicate_text)
        pos_pk_score,pos_pk,pos_id,pos_ex = get_pk_ex(pos_pk_attribute,pos_sim_score[i],top_n,drop_bad_pk,drop_duplicate_text)
        js_ssa[i][f"bert_{sim_attribute}_list_all_score_each_frame"] = neg_pk_score+pos_pk_score
        js_ssa[i][f"bert_{sim_attribute}_list_all_pk_each_frame"] = neg_pk+pos_pk
        js_ssa[i][f"bert_{sim_attribute}_list_best_score_each_polarity"] = neg_pk_score[:1]+pos_pk_score[:1]
        js_ssa[i][f"bert_{sim_attribute}_list_best_pk_each_polarity"] = neg_pk[:1]+pos_pk[:1]
        js_ssa[i][f"bert_{sim_attribute}_list_negative_mti"] = neg_id
        js_ssa[i][f"bert_{sim_attribute}_list_positive_mti"] = pos_id
        js_ssa[i][f"bert_{sim_attribute}_list_negative_example_mti"] = neg_ex
        js_ssa[i][f"bert_{sim_attribute}_list_positive_example_mti"] = pos_ex
    # Write to jsonl
    write_jsonl(js_ssa,ssa_output_jsonl_path)
    # Return the js_ssa if needed
    if return_result == "yes":
        return js_ssa