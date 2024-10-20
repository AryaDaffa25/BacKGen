from converter.converter_utility import spacy_tokenizer, token_spacy_tag
from converter.io import read_jsonl, write_jsonl
import spacy

nlp = spacy.load("en_core_web_sm")

def frame2syntaxtree(frame,bracket_type):
    text = frame.get("text")
    token_list = spacy_tokenizer(text)
    spacy_doc = nlp(text)
    if bracket_type == "round":
        open_bracket = "("
        close_bracket = ")"
    elif bracket_type == "square":
        open_bracket = "["
        close_bracket = "]"
    else:
        raise ValueError("Currently only support 'round' and 'square' bracket. Feel free to modify the code.")
    frame_label = frame.get("label")
    synt = f"SYNT##{frame_label}"
    idx_span_lu = frame.get("span")
    token_list_lu = token_list[idx_span_lu[0]:idx_span_lu[1]+1]
    lex_lu_list = []
    for token in token_list_lu:
        lex_lu_list.append(f"{open_bracket}LEX##{token}::{token_spacy_tag(token,spacy_doc,text)}{close_bracket}")
    string_token_lu = "".join(lex_lu_list)
    string_lu = f"{open_bracket}POS##LU{string_token_lu}{close_bracket}"
    frame_children = frame.get("children")
    sorted_frame_children = {}
    for child in frame_children:
        span_start = child.get("span")[0]
        sorted_frame_children[span_start] = child
    id_frame_children = sorted(sorted_frame_children)
    pos_list = []
    for child_id in id_frame_children:
        child = sorted_frame_children[child_id]
        arg_label = child.get("label")
        pos = f"POS##{arg_label}"
        span_idx = child.get("span")
        span_token = token_list[span_idx[0]:span_idx[1]+1]
        for span in span_token:
            lex = f"{open_bracket}LEX##{span}::{token_spacy_tag(span,spacy_doc,text)}{close_bracket}"
            pos = pos+lex
        pos_list.append(f"{open_bracket}{pos}{close_bracket}")
    id_frame_children.append(idx_span_lu[0])
    id_frame_children = sorted(id_frame_children)
    idx_insert_lu = id_frame_children.index(idx_span_lu[0])
    pos_list.insert(idx_insert_lu,string_lu)
    string_post_list = "".join(pos_list)
    syntaxtree = f"{open_bracket}{synt}{string_post_list}{close_bracket}"
    return syntaxtree

def frame2symbolic(frame,bracket_type):
    token_list = spacy_tokenizer(frame.get("text"))
    if bracket_type == "round":
        open_bracket = "("
        close_bracket = ")"
    elif bracket_type == "square":
        open_bracket = "["
        close_bracket = "]"
    else:
        raise ValueError("Currently only support 'round' and 'square' bracket. Feel free to modify the code.")
    frame_label = frame.get("label")
    idx_span_lu = frame.get("span")
    span_lu = " ".join(token_list[idx_span_lu[0]:idx_span_lu[1]+1])
    string_lu = f"LU{open_bracket}{span_lu}{close_bracket}"
    frame_children = frame.get("children")
    sorted_frame_children = {}
    for child in frame_children:
        span_start = child.get("span")[0]
        sorted_frame_children[span_start] = child
    id_frame_children = sorted(sorted_frame_children)
    arg_list = []
    for child_id in id_frame_children:
        child = sorted_frame_children[child_id]
        arg_label = child.get("label")
        arg_span_idx = child.get("span")
        arg_span = " ".join(token_list[arg_span_idx[0]:arg_span_idx[1]+1])
        arg_list.append(f"{arg_label}{open_bracket}{arg_span}{close_bracket}")
    arg_string = ",".join(arg_list)
    if arg_list == []:
        symbolic = f"{frame_label}{open_bracket}{string_lu}{close_bracket}"
    else:
        symbolic = f"{frame_label}{open_bracket}{string_lu},{arg_string}{close_bracket}"
    return symbolic

def get_span_lu(frame,token_list):
    idx_span_lu = frame.get("span")
    token_list_lu = token_list[idx_span_lu[0]:idx_span_lu[1]+1]
    return " ".join(token_list_lu)

def frame2klp(frame,bracket_type):
    frame_label = frame.get("label")
    frame_syntaxtree = frame2syntaxtree(frame,bracket_type)
    frame_symbolic = frame2symbolic(frame,bracket_type)
    text_ori = frame.get("text")
    span_lu = get_span_lu(frame,spacy_tokenizer(text_ori))
    text_id = frame.get("id")
    return f"{frame_label} |BT:frame_syntaxtree| {frame_syntaxtree} |ET| |BS:original_text| {text_ori} |ES| |BS:span_lu| {span_lu} |ES| |BS:frame_symbolic| {frame_symbolic} |ES| |BS:id| {text_id} |ES|"

def jsonl2klp_frame2klp(input_file_path,output_file_path,bracket_type):
    list_frame = read_jsonl(input_file_path)
    with open(output_file_path,"w",newline=None) as output:
        for frame in list_frame:
            klp = frame2klp(frame,bracket_type)
            output.write(f"{klp}\n")
        print(f"The .klp file has been saved in {output_file_path}")

def split_frame_jsonl2jsonl(input_file_path,output_file_path):
    js_data = read_jsonl(input_file_path)
    list_frame = []
    for js in js_data:
        id = js.get("sent_id")
        text = js.get("text")
        data_frame = js.get("frame_tree")
        for frame_dict in data_frame:
            frame_dict = {"id": id, "text":text, **frame_dict}
            list_frame.append(frame_dict)
    write_jsonl(list_frame,output_file_path)
    print(f"The splitted frame has been saved into {output_file_path}")

def split_frame_jsonl2klp(input_file_path,output_file_path,bracket_type):
    js_data = read_jsonl(input_file_path)
    list_frame = []
    for js in js_data:
        id = js.get("sent_id")
        text = js.get("text")
        data_frame = js.get("frame_tree")
        for frame_dict in data_frame:
            frame_dict = {"id": id, "text":text, **frame_dict}
            list_frame.append(frame_dict)
    with open(output_file_path,"w",newline=None) as output:
        for frame in list_frame:
            klp = frame2klp(frame,bracket_type)
            output.write(f"{klp}\n")
        print(f"The splitted frame .klp file has been saved in {output_file_path}")

def jsonlpk2klp(input_file_path,output_file_path):
    list_frame = read_jsonl(input_file_path)
    with open(output_file_path,"w",newline=None) as output:
        for frame in list_frame:
            list_frame_symbolic = frame.get("list_frame_symbolic")
            if len(list_frame_symbolic) != 1:
                medoid_frame_label = frame.get("medoid_symbolic_tree").split("(")[0]
                medoid_syntax_tree = frame.get("medoid_syntax_tree")
                cluster_id = frame.get("cluster_id")
                polarity_label = frame.get("polarity_label")
                cleaned_pk = frame.get("cleaned_answer")
                klp = f"{medoid_frame_label} |BT:frame_syntaxtree| {medoid_syntax_tree} |ET| |BS:cluster_id| {cluster_id} |ES| |BS:polarity_label| {polarity_label} |ES| |BS:list_frame_symbolic| {list_frame_symbolic} |ES| |BS:cleaned_pk| {cleaned_pk} |ES|"
                output.write(f"{klp}\n")
        print(f"The .klp file has been saved in {output_file_path}")