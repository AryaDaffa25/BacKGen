# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from converter.converter_utility import spacy_sentencizer
from converter.io import read_jsonl, write_jsonl
import re

def bk_preprocessor_single_sentence(original_answer):
    original_answer = original_answer.replace("\n","")
    list_answer = spacy_sentencizer(original_answer)
    answer = ""
    if len(list_answer) == 1:
        answer = list_answer[0]
    else:
        for i in range(0,len(list_answer)):
            if "Answer:\n" in list_answer[i]:
                answer = list_answer[i].split("Answer:\n")[1]
            elif "Answer = " in list_answer[i]:
                answer = list_answer[i].split("Answer = ")[1]
            elif "Answer:" in list_answer[i]:
                try:
                    answer = list_answer[i].split("Answer:")[1]
                    if answer == "":
                        answer = list_answer[i+1]
                except:
                    # print(f"original answer -> {original_answer}")
                    # print(f"list answer -> {list_answer}")
                    try:
                        answer = list_answer[i+1].strip()
                    except:
                        answer = list_answer[0].strip()
        if answer == "":
            answer = list_answer[0]
    answer = answer.replace("</s>","")
    answer = answer.replace("\n","")
    answer = answer.strip()
    return answer

def bk_preprocessor_whole_answer(original_answer):
    answer = original_answer.replace("</s>","")
    answer = answer.replace("\n"," ")
    answer = answer.replace("\t"," ")
    answer = re.sub(' +',' ',answer)
    answer = answer.strip()
    return answer

def bk_preprocessor_bulk(input_file_path,clean_answer_mode):
    js_data = read_jsonl(input_file_path)
    for js_idx in range(len(js_data)):
        del js_data[js_idx]["list_subtoken_score"]
        del js_data[js_idx]["list_subtoken"]
        if clean_answer_mode == "single_sentence":
            js_data[js_idx]["cleaned_answer"] = bk_preprocessor_single_sentence(js_data[js_idx]["original_answer"])
        elif clean_answer_mode == "whole_answer":
            js_data[js_idx]["cleaned_answer"] = bk_preprocessor_whole_answer(js_data[js_idx]["original_answer"])
        else:
            raise ValueError("Wrong 'clean_answer_mode'. Only 'single_sentence' or 'whole_answer' that are currently available. Feel free to add your own 'clean_answer_mode'.")
    return js_data

def bk_preprocessor_file2file(input_file_path,output_file_path,clean_answer_mode):
    cleaned_js_data = bk_preprocessor_bulk(input_file_path,clean_answer_mode)
    write_jsonl(cleaned_js_data,output_file_path)
    print(f"Generalized BK data has been cleaned and saved into {output_file_path}")