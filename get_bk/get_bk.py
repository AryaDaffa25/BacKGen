# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from get_bk.get_bk_utility import get_bk_candidate_per_polarity, get_bk_per_polarity
from converter.io import write_jsonl
import argparse

# Function to get bk from all polarity
def get_bk(splitted_frame_klp_path,negative_bk_klp_path,neutral_bk_klp_path,positive_bk_klp_path,output_file_path,lib_folder_path,top_n=1,return_result="yes"):
    # Get bk for each polarity
    print("Processing to get bk candidate of negative polarity.")
    negative_bk = get_bk_per_polarity(get_bk_candidate_per_polarity(splitted_frame_klp_path,negative_bk_klp_path,lib_folder_path,top_n),"negative",top_n)
    print("Processing to get bk candidate of neutral polarity.")
    neutral_bk = get_bk_per_polarity(get_bk_candidate_per_polarity(splitted_frame_klp_path,neutral_bk_klp_path,lib_folder_path,top_n),"neutral",top_n)
    print("Processing to get bk candidate of positive polarity.")
    positive_bk = get_bk_per_polarity(get_bk_candidate_per_polarity(splitted_frame_klp_path,positive_bk_klp_path,lib_folder_path,top_n),"positive",top_n)
    print("Process for getting bk candidate of each polarity is finished. Now we are processing to get top_n bk.")
    # Check if the length of list of bk is different
    if len(negative_bk) != len(neutral_bk) or len(negative_bk) != len(positive_bk) or len(neutral_bk) != len(positive_bk):
        raise ValueError("There is difference of length of list of bk!")
    # Define result container
    jsonl_bk = []
    for idx in range(len(negative_bk)):
        # Get id
        id_text = negative_bk[idx][0]
        # Get similarity score of top_n bk for each polarity, just in case we needed it in the future ablation test
        list_neg_bk_sim_score = negative_bk[idx][2]
        list_neu_bk_sim_score = neutral_bk[idx][2]
        list_pos_bk_sim_score = positive_bk[idx][2]
        # Get top_n bk for each polarity
        list_neg_bk = negative_bk[idx][3]
        list_neu_bk = neutral_bk[idx][3]
        list_pos_bk = positive_bk[idx][3]
        jsonl_bk.append({"id":id_text,"list_neg_bk_sim_score":list_neg_bk_sim_score,"list_neg_bk":list_neg_bk,"list_neu_bk_sim_score":list_neu_bk_sim_score,"list_neu_bk":list_neu_bk,"list_pos_bk_sim_score":list_pos_bk_sim_score,"list_pos_bk":list_pos_bk})
    write_jsonl(jsonl_bk,output_file_path)
    if return_result == "yes":
        return jsonl_bk

def main(args):
    # Get all arguments
    splitted_frame_klp_path = args.splitted_frame_klp_path
    negative_bk_klp_path = args.negative_bk_klp_path
    neutral_bk_klp_path = args.neutral_bk_klp_path
    positive_bk_klp_path = args.positive_bk_klp_path
    output_file_path = args.output_file_path
    lib_folder_path = args.lib_folder_path
    top_n = args.top_n
    # Running the main process
    get_bk(splitted_frame_klp_path,negative_bk_klp_path,neutral_bk_klp_path,positive_bk_klp_path,output_file_path,lib_folder_path,top_n,return_result="no")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splitted_frame_klp_path", help="Your splitted frame klp path.",
        type=str
    )
    parser.add_argument(
        "--negative_bk_klp_path", help="Your medoid negative polarity klp path.",
        type=str
    )
    parser.add_argument(
        "--neutral_bk_klp_path", help="Your medoid neutral polarity klp path.",
        type=str
    )
    parser.add_argument(
        "--positive_bk_klp_path", help="Your medoid positive polarity klp path.",
        type=str
    )
    parser.add_argument(
        "--output_file_path", help="Your output file path to store the choosen bk.",
        type=str
    )
    parser.add_argument(
        "--lib_folder_path", help="Your java library path for function to calculate similarity using kernel based method.",
        type=str
    )
    parser.add_argument(
        "--top_n", help="The number of top bk that you want to retrieve.",
        type=int
    )
    args = parser.parse_args()
    main(args)