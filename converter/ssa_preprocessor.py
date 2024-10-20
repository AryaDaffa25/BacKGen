import json
from converter.io import read_jsonl
from converter.converter_utility import clean_preprocessing

def ssa_preprocessor(file_input_path,file_output_path,verbose="yes"):
    data_ssa = read_jsonl(file_input_path)
    with open(file_output_path, 'w') as file_output:
        if verbose == "yes":
            total_data = len(data_ssa)
            print(f"Preprocessing data is starting with {total_data} of total data.")
        for i in range(len(data_ssa)):
            if verbose == "yes":
                print(f"Preprocess data {i+1} of {total_data}")
            text = data_ssa[i].get("text")
            text = clean_preprocessing(text)
            oesc_tuple = data_ssa[i].get("oesc_tuple")
            preprocessed_oesc = []
            if oesc_tuple != []:
                for oesc in oesc_tuple:
                    preprocessed_oesc.append([clean_preprocessing(oesc[0]),oesc[1]])
            entry = {"id":data_ssa[i].get("id"),
                    "text":text,
                    "oesc_tuple":preprocessed_oesc}
            json.dump(entry, file_output)
            file_output.write('\n')
        if verbose == "yes":
            print(f"Preprocessing data is finished and saved in {file_output_path}")