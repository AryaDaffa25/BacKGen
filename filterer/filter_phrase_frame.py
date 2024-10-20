from converter.io import read_jsonl, write_jsonl
from converter.converter_utility import spacy_tokenizer
from filterer.filterer_utility import all_indices

def filter_phrase_frame_utt(data_ssa,data_frame):
    filtered_frame = []
    id = data_ssa.get("id")
    text = data_frame.get("text").lower()
    data_frame = data_frame.get("frame_tree")
    ssa_label_list = data_ssa.get("oesc_tuple")
    text_token = spacy_tokenizer(text)
    for ssa_label in ssa_label_list:
        ssa_label_span = ssa_label[0].lower()
        ssa_label_span_token = spacy_tokenizer(ssa_label_span)
        idx_first_word = all_indices(text_token,ssa_label_span_token[0])
        idx_last_word = all_indices(text_token,ssa_label_span_token[-1])
        try:
            idx_ssa_label_span = [idx_first_word[-1],idx_last_word[0]]
        except:
            raise ValueError(f"Invalid SSA label span index, please check your document:\nid -> {id}\ntext_token -> {text_token}\nssa_label_span_token -> {ssa_label_span_token}")
        # Looking for intersection
        for frame_dict in data_frame:
            frame_dict = {"id": id, "text":text, **frame_dict}
            span_frame_dict = frame_dict.get("span")
            if (span_frame_dict[0] >= idx_ssa_label_span[0] and span_frame_dict[0] <= idx_ssa_label_span[1]) or (span_frame_dict[1] >= idx_ssa_label_span[0] and span_frame_dict[1] <= idx_ssa_label_span[1]):
                if frame_dict not in filtered_frame:
                    filtered_frame.append(frame_dict)
            frame_children = frame_dict.get("children")
            for child in frame_children:
                span_frame_dict = child.get("span")
                if (span_frame_dict[0] >= idx_ssa_label_span[0] and span_frame_dict[0] <= idx_ssa_label_span[1]) or (span_frame_dict[1] >= idx_ssa_label_span[0] and span_frame_dict[1] <= idx_ssa_label_span[1]):
                    if frame_dict not in filtered_frame:
                        filtered_frame.append(frame_dict)
    return filtered_frame

def filter_phrase_frame_list(data_ssa,data_frame):
    list_filtered_frame = []
    for i in range(len(data_ssa)):
        if data_ssa[i].get('oesc_tuple') == []:
            print(f"Data text with ID: {data_ssa[i].get('id')} has no frame list so we skip it.")
            continue
        else:
            filtered_frame = filter_phrase_frame_utt(data_ssa[i],data_frame[i])
            list_filtered_frame.extend(filtered_frame)
    return list_filtered_frame

def jsonl2jsonl_filter_phrase_frame(ssa_file_path,input_file_path,output_file_path,verbose="yes"):
    data_ssa = read_jsonl(ssa_file_path)
    data_frame = read_jsonl(input_file_path)
    list_filtered_frame = filter_phrase_frame_list(data_ssa,data_frame)
    write_jsonl(list_filtered_frame,output_file_path)
    if verbose == "yes":
        print(f"Filtered frame has been saved into {output_file_path}.")