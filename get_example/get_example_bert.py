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

def get_ex_attribute(ex_jsonl_path):
    js_ex = read_jsonl(ex_jsonl_path)
    list_ex_text = [js.get('text') for js in js_ex]
    list_ex_st_span = [js.get('st_span') for js in js_ex]
    list_ex_st_polarity = [js.get('st_polarity') for js in js_ex]
    return pd.DataFrame({'list_ex_text':list_ex_text,
                         'list_ex_st_span':list_ex_st_span,
                         'list_ex_st_polarity':list_ex_st_polarity})

def get_ex(ex_attribute,instance_sim_score,top_n,drop_duplicate_text):
    ex_attribute["instance_sim_score"] = instance_sim_score
    ex_attribute = ex_attribute.sort_values(by=['instance_sim_score'], ascending=[False])
    if drop_duplicate_text == "yes":
        ex_attribute = ex_attribute.drop_duplicates(subset=['list_ex_text'], keep="first")
    ex_attribute = ex_attribute.reset_index(drop=True)
    selected_ex = []
    for i in range(0,top_n):
        selected_ex.append((ex_attribute.list_ex_text[i],ex_attribute.list_ex_st_span[i],ex_attribute.list_ex_st_polarity[i]))
    return selected_ex

def get_example_bert(input_file_path,output_file_path,negative_example_path,positive_example_path,sim_attribute,bert_model="all-MiniLM-L6-v2",sim_function="cosine",top_n=2,drop_duplicate_text="yes",return_result="no"):
    # Get ssa attribute
    js_ssa = read_jsonl(input_file_path)
    list_ssa_text = [js.get('text') for js in js_ssa]
    list_ssa_st_span = [js.get('st_span') for js in js_ssa]
    # Get pk attribute
    neg_ex_attribute = get_ex_attribute(negative_example_path)
    pos_ex_attribute = get_ex_attribute(positive_example_path)
    # Get sim score
    if sim_attribute == "text":
        neg_sim_score = get_sim_score(list_ssa_text,neg_ex_attribute.list_ex_text,bert_model,sim_function)
        pos_sim_score = get_sim_score(list_ssa_text,pos_ex_attribute.list_ex_text,bert_model,sim_function)
    elif sim_attribute == "st_span":
        neg_sim_score = get_sim_score(list_ssa_st_span,neg_ex_attribute.list_ex_st_span,bert_model,sim_function)
        pos_sim_score = get_sim_score(list_ssa_st_span,pos_ex_attribute.list_ex_st_span,bert_model,sim_function)
    elif sim_attribute == "average":
        neg_sim_score = mean_sim_score(list_ssa_text,list_ssa_st_span,neg_ex_attribute.list_ex_text,neg_ex_attribute.list_ex_st_span,bert_model,sim_function)
        pos_sim_score = mean_sim_score(list_ssa_text,list_ssa_st_span,pos_ex_attribute.list_ex_text,pos_ex_attribute.list_ex_st_span,bert_model,sim_function)
    else:
        raise ValueError("Wrong 'sim_attribute' parameter. Only 'text', 'st_span', or 'average' is allowed.")
    # Add pk and example to the splitted_stc dataset
    for i in range(len(js_ssa)):
        # Get pk and example for instance i
        neg_ex = get_ex(neg_ex_attribute,neg_sim_score[i],top_n,drop_duplicate_text)
        pos_ex = get_ex(pos_ex_attribute,pos_sim_score[i],top_n,drop_duplicate_text)
        js_ssa[i][f"bert_{sim_attribute}_list_negative_example"] = neg_ex
        js_ssa[i][f"bert_{sim_attribute}_list_positive_example"] = pos_ex
    # Write to jsonl
    write_jsonl(js_ssa,output_file_path)
    # Return the js_ssa if needed
    if return_result == "yes":
        return js_ssa