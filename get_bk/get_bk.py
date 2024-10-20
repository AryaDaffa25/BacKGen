# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from get_bk.get_bk_utility import get_bk_candidate_per_polarity, get_bk_per_polarity
from converter.io import write_jsonl
import argparse

# Function to get pk from all polarity
def get_pk(splitted_frame_klp_path,negative_pk_klp_path,neutral_pk_klp_path,positive_pk_klp_path,output_file_path,lib_folder_path,top_n=1,return_result="yes"):
    # Get pk for each polarity
    print("Processing to get pk candidate of negative polarity.")
    negative_pk = get_bk_per_polarity(get_bk_candidate_per_polarity(splitted_frame_klp_path,negative_pk_klp_path,lib_folder_path,top_n),"negative",top_n)
    print("Processing to get pk candidate of neutral polarity.")
    neutral_pk = get_bk_per_polarity(get_bk_candidate_per_polarity(splitted_frame_klp_path,neutral_pk_klp_path,lib_folder_path,top_n),"neutral",top_n)
    print("Processing to get pk candidate of positive polarity.")
    positive_pk = get_bk_per_polarity(get_bk_candidate_per_polarity(splitted_frame_klp_path,positive_pk_klp_path,lib_folder_path,top_n),"positive",top_n)
    print("Process for getting pk candidate of each polarity is finished. Now we are processing to get top_n pk.")
    # Check if the length of list of pk is different
    if len(negative_pk) != len(neutral_pk) or len(negative_pk) != len(positive_pk) or len(neutral_pk) != len(positive_pk):
        raise ValueError("There is difference of length of list of pk!")
    # Define result container
    jsonl_pk = []
    for idx in range(len(negative_pk)):
        # Get id
        id_text = negative_pk[idx][0]
        # Get similarity score of top_n pk for each polarity, just in case we needed it in the future ablation test
        list_neg_pk_sim_score = negative_pk[idx][2]
        list_neu_pk_sim_score = neutral_pk[idx][2]
        list_pos_pk_sim_score = positive_pk[idx][2]
        # Get top_n pk for each polarity
        list_neg_pk = negative_pk[idx][3]
        list_neu_pk = neutral_pk[idx][3]
        list_pos_pk = positive_pk[idx][3]
        jsonl_pk.append({"id":id_text,"list_neg_pk_sim_score":list_neg_pk_sim_score,"list_neg_pk":list_neg_pk,"list_neu_pk_sim_score":list_neu_pk_sim_score,"list_neu_pk":list_neu_pk,"list_pos_pk_sim_score":list_pos_pk_sim_score,"list_pos_pk":list_pos_pk})
    write_jsonl(jsonl_pk,output_file_path)
    if return_result == "yes":
        return jsonl_pk

def main(args):
    # Get all arguments
    splitted_frame_klp_path = args.splitted_frame_klp_path
    negative_pk_klp_path = args.negative_pk_klp_path
    neutral_pk_klp_path = args.neutral_pk_klp_path
    positive_pk_klp_path = args.positive_pk_klp_path
    output_file_path = args.output_file_path
    lib_folder_path = args.lib_folder_path
    top_n = args.top_n
    # Running the main process
    get_pk(splitted_frame_klp_path,negative_pk_klp_path,neutral_pk_klp_path,positive_pk_klp_path,output_file_path,lib_folder_path,top_n,return_result="no")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splitted_frame_klp_path", help="Your splitted frame klp path.",
        type=str
    )
    parser.add_argument(
        "--negative_pk_klp_path", help="Your medoid negative polarity klp path.",
        type=str
    )
    parser.add_argument(
        "--neutral_pk_klp_path", help="Your medoid neutral polarity klp path.",
        type=str
    )
    parser.add_argument(
        "--positive_pk_klp_path", help="Your medoid positive polarity klp path.",
        type=str
    )
    parser.add_argument(
        "--output_file_path", help="Your output file path to store the choosen pk.",
        type=str
    )
    parser.add_argument(
        "--lib_folder_path", help="Your java library path for function to calculate similarity using kernel based method.",
        type=str
    )
    parser.add_argument(
        "--top_n", help="The number of top pk that you want to retrieve.",
        type=int
    )
    args = parser.parse_args()
    main(args)