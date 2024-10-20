# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import library
from converter.io import read_jsonl,write_jsonl

def split_st(input_file_path,output_file_path,return_result="yes"):
    js_ssa = read_jsonl(input_file_path)
    js_splitted = []
    print("Start the splitting process ....")
    for js in js_ssa:
        oesc_tuple = js.get('oesc_tuple')
        js_id = js.get('id')
        if oesc_tuple == []:
            print(f"Text with ID: {js.get('id')} has no oesc_tuple so that we do not used that for sentiment term classification task.")
            continue
        else:
            for i in range(len(oesc_tuple)):
                st_id = f"{js_id}_{str(i+1)}"
                text = js.get('text')
                st_span = oesc_tuple[i][0]
                if oesc_tuple[i][1] == "Exp_Negative":
                    st_polarity = "negative"
                elif oesc_tuple[i][1] == "Exp_Positive":
                    st_polarity = "positive"
                # Check label error
                else:
                    raise ValueError(f"There is polarity label error on text with ID: {js_id}. Please check it.")
                js_splitted.append({"id":js_id,
                                    "st_id":st_id,
                                    "text":text,
                                    "st_span":st_span,
                                    "st_polarity":st_polarity})
    print("Splitting proccess is done. Now we are writing to the destined output file path ...")
    write_jsonl(js_splitted,output_file_path)
    if return_result == "yes":
        return js_splitted