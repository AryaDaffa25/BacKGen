# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from converter.io import read_jsonl, write_jsonl
from converter.converter_utility import spacy_tokenizer
from filterer.filterer_utility import all_indices

def filter_sentiment_frame_utt(data_ssa,data_frame):
    frame_positive,frame_negative,frame_neutral = [],[],[]
    id = data_ssa.get("id")
    text = data_frame.get("text").lower()
    data_frame = data_frame.get("frame_tree")
    ssa_label_list = data_ssa.get("oesc_tuple")
    if ssa_label_list == []:
        for frame_dict in data_frame:
            frame_dict = {"id": id, "text":text, **frame_dict}
            frame_neutral.append(frame_dict)
    else:
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
                    if ssa_label[1] == "Exp_Positive":
                        if frame_dict not in frame_positive:
                            frame_positive.append(frame_dict)
                    elif ssa_label[1] == "Exp_Negative":
                        if frame_dict not in frame_negative:
                            frame_negative.append(frame_dict)
                    else:
                        raise ValueError("Wrong sentiment term polarity label. Check your dataset!")
                frame_children = frame_dict.get("children")
                for child in frame_children:
                    span_frame_dict = child.get("span")
                    if (span_frame_dict[0] >= idx_ssa_label_span[0] and span_frame_dict[0] <= idx_ssa_label_span[1]) or (span_frame_dict[1] >= idx_ssa_label_span[0] and span_frame_dict[1] <= idx_ssa_label_span[1]):
                        if ssa_label[1] == "Exp_Positive":
                            if frame_dict not in frame_positive:
                                frame_positive.append(frame_dict)
                        elif ssa_label[1] == "Exp_Negative":
                            if frame_dict not in frame_negative:
                                frame_negative.append(frame_dict)
                        else:
                            raise ValueError("Wrong sentiment term polarity label. Check your dataset!")
    return frame_positive,frame_negative,frame_neutral

def filter_sentiment_frame_list(data_ssa,data_frame):
    list_frame_positive,list_frame_negative,list_frame_neutral = [],[],[]
    for i in range(len(data_ssa)):
        frame_positive,frame_negative,frame_neutral = filter_sentiment_frame_utt(data_ssa[i],data_frame[i])
        list_frame_positive.extend(frame_positive)
        list_frame_negative.extend(frame_negative)
        list_frame_neutral.extend(frame_neutral)
    return list_frame_positive,list_frame_negative,list_frame_neutral

def jsonl2jsonl_filter_sentiment_frame(ssa_file_path,frame_file_path,output_folder_path="",verbose="yes"):
    data_ssa = read_jsonl(ssa_file_path)
    data_frame = read_jsonl(frame_file_path)
    list_frame_positive,list_frame_negative,list_frame_neutral = filter_sentiment_frame_list(data_ssa,data_frame)
    file_name = frame_file_path.split("/")[-1]
    if output_folder_path != "" and output_folder_path[-1] != "/":
        output_folder_path = f"{output_folder_path}/"
    positive_file_path = f"{output_folder_path}positive-{file_name}"
    negative_file_path = f"{output_folder_path}negative-{file_name}"
    neutral_file_path = f"{output_folder_path}neutral-{file_name}"
    write_jsonl(list_frame_positive,positive_file_path)
    if verbose == "yes":
        print(f"Positive frame has been saved into {positive_file_path}.")
    write_jsonl(list_frame_negative,negative_file_path)
    if verbose == "yes":
        print(f"Negative frame has been saved into {negative_file_path}.")
    write_jsonl(list_frame_neutral,neutral_file_path)
    if verbose == "yes":
        print(f"Neutral frame has been saved into {neutral_file_path}.")