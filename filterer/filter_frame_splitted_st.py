from converter.io import read_jsonl, write_jsonl
from converter.converter_utility import spacy_tokenizer
from filterer.filterer_utility import all_indices

def filter_phrase_frame_st(data_ssa,data_frame,relax_filter):
    filtered_frame = []
    id = data_ssa.get("id")
    st_id = data_ssa.get("st_id")
    text = data_frame.get("text").lower()
    data_frame = data_frame.get("frame_tree")
    text_token = spacy_tokenizer(text)
    ssa_label_span = data_ssa.get('st_span').lower()
    ssa_label_span_token = spacy_tokenizer(ssa_label_span)
    idx_first_word = all_indices(text_token,ssa_label_span_token[0])
    idx_last_word = all_indices(text_token,ssa_label_span_token[-1])
    try:
        idx_ssa_label_span = [idx_first_word[-1],idx_last_word[0]]
    except:
        raise ValueError(f"Invalid SSA label span index, please check your document:\nid -> {id}\ntext_token -> {text_token}\nssa_label_span_token -> {ssa_label_span_token}")
    # Looking for intersection
    for frame_dict in data_frame:
        frame_dict = {"id": id, "st_id":st_id,"text":text, **frame_dict}
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
    # Debug to check empty list
    if filtered_frame == [] and relax_filter == "yes":
        closed_frame_dict = data_frame[0]
        closed_frame_dict = {"id": id, "st_id":st_id,"text":text, **closed_frame_dict}
        span_frame_dict = closed_frame_dict.get('span')
        closed_distance = min([abs(span_frame_dict[0]-idx_ssa_label_span[0]),abs(span_frame_dict[0]-idx_ssa_label_span[1]),abs(span_frame_dict[1]-idx_ssa_label_span[0]),abs(span_frame_dict[1]-idx_ssa_label_span[1])])
        if len(data_frame) > 1:
            for i in range(1,len(data_frame)):
                frame_dict = {"id": id, "st_id":st_id,"text":text, **data_frame[i]}
                span_frame_dict = frame_dict.get('span')
                span_distance = min([abs(span_frame_dict[0]-idx_ssa_label_span[0]),abs(span_frame_dict[0]-idx_ssa_label_span[1]),abs(span_frame_dict[1]-idx_ssa_label_span[0]),abs(span_frame_dict[1]-idx_ssa_label_span[1])])
                if span_distance < closed_distance:
                    closed_frame_dict = frame_dict
                    closed_distance = span_distance
                else:
                    frame_children = frame_dict.get("children")
                    for child in frame_children:
                        span_frame_dict = child.get("span")
                        span_distance = min([abs(span_frame_dict[0]-idx_ssa_label_span[0]),abs(span_frame_dict[0]-idx_ssa_label_span[1]),abs(span_frame_dict[1]-idx_ssa_label_span[0]),abs(span_frame_dict[1]-idx_ssa_label_span[1])])
                    if span_distance < closed_distance:
                        closed_frame_dict = frame_dict
                        closed_distance = span_distance
                        break
        filtered_frame.append(closed_frame_dict)
    return filtered_frame

def filter_phrase_frame_st_list(data_ssa,data_frame,rilex_filter):
    list_filtered_frame = []
    for i in range(len(data_ssa)):
        data_frame_i = [js for js in data_frame if js.get('sent_id') == data_ssa[i].get('id')][0]
        if data_frame_i.get('frame_list') != []:
            filtered_frame = filter_phrase_frame_st(data_ssa[i],data_frame_i,rilex_filter)
            list_filtered_frame.extend(filtered_frame)
        else:
            print(f"Data with ID: {data_ssa[i].get('id')} do not has frame so that we skip this data.")
    return list_filtered_frame

def jsonl2jsonl_filter_phrase_frame_st(ssa_file_path,frame_file_path,output_file_path,rilex_filter="yes",verbose="yes"):
    data_ssa = read_jsonl(ssa_file_path)
    data_frame = read_jsonl(frame_file_path)
    list_filtered_frame = filter_phrase_frame_st_list(data_ssa,data_frame,rilex_filter)
    write_jsonl(list_filtered_frame,output_file_path)
    if verbose == "yes":
        print(f"Filtered frame has been saved into {output_file_path}.")