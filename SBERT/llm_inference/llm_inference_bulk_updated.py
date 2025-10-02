# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from converter.io import read_jsonl, write_jsonl
from llm_inference.get_prompt import get_prompt
from llm_inference.llm_inference_utility import *
import argparse

# Function for bulk inference
# Import di bagian atas file
import torch
import gc

def nuclear_cleanup():
    """Most aggressive memory cleanup possible"""
    # Multiple rounds of garbage collection
    for i in range(5):
        collected = gc.collect()
    
    if torch.cuda.is_available():
        # Clear all CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

def llm_inference_bulk(input_file_path,list_inference_attribute,prompt_task_type,prompt_variant,model_name,gpu_device="",hf_token="",max_new_tokens=250,return_mode="with_subtoken_score",verbose="yes"):
    # Load model and tokenizer
    if verbose == "yes":
        print(f"Loading model and tokenizer from pretrained model: {model_name}")
    model,tokenizer = load_model_tokenizer(model_name,hf_token)
    
    # Initial nuclear cleanup after model load
    nuclear_cleanup()
    
    # Read jsonl data
    jsonl_data = read_jsonl(input_file_path)
    if verbose == "yes":
        len_jsonl_data = len(jsonl_data)
    
    # Loop the inference process for all data in the jsonl file
    for js_idx in range(0,len(jsonl_data)):
        # Nuclear cleanup sebelum setiap inference
        nuclear_cleanup()
        
        # Extra aggressive cleanup every 5 iterations
        if js_idx % 5 == 0 and js_idx > 0:
            if verbose == "yes":
                print(f"Extra aggressive cleanup at iteration {js_idx+1}")
            nuclear_cleanup()
            # Force additional cleanup
            nuclear_cleanup()
        
        if verbose == "yes":
            print(f"Processing file {js_idx+1} of {len_jsonl_data} total texts.")
        
        # Get list of inference input to generate the prompt
        list_inference_input = []
        for attribute in list_inference_attribute:
            list_inference_input.append(jsonl_data[js_idx].get(attribute))
        
        try:
            # Normal inference dengan semua data
            prompt = get_prompt(list_inference_input, prompt_task_type, prompt_variant)
            if return_mode == "without_subtoken_score":
                original_answer = llm_inference_greedy_search(prompt,tokenizer,model,max_new_tokens,return_mode)
                jsonl_data[js_idx]["original_answer"] = original_answer
            elif return_mode == "with_subtoken_score":
                original_answer, list_subtoken, list_subtoken_score = llm_inference_greedy_search(prompt,tokenizer,model,gpu_device,max_new_tokens,return_mode)
                jsonl_data[js_idx]["original_answer"] = original_answer
                jsonl_data[js_idx]["list_subtoken"] = list_subtoken
                jsonl_data[js_idx]["list_subtoken_score"] = list_subtoken_score
        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                # Nuclear cleanup before retry
                nuclear_cleanup()
                
                try:
                    # First fallback: potong jadi 5 instances
                    list_inference_input_reduced = []
                    for i, attribute in enumerate(list_inference_attribute):
                        if attribute in ["list_frame_symbolic", "list_text_source"]:
                            original_list = list_inference_input[i]
                            reduced_list = original_list[:5] if isinstance(original_list, list) else original_list
                            list_inference_input_reduced.append(reduced_list)
                        else:
                            list_inference_input_reduced.append(list_inference_input[i])
                    
                    prompt_reduced = get_prompt(list_inference_input_reduced, prompt_task_type, prompt_variant)
                    if return_mode == "without_subtoken_score":
                        original_answer = llm_inference_greedy_search(prompt_reduced,tokenizer,model,max_new_tokens,return_mode)
                        jsonl_data[js_idx]["original_answer"] = original_answer
                    elif return_mode == "with_subtoken_score":
                        original_answer, list_subtoken, list_subtoken_score = llm_inference_greedy_search(prompt_reduced,tokenizer,model,gpu_device,max_new_tokens,return_mode)
                        jsonl_data[js_idx]["original_answer"] = original_answer
                        jsonl_data[js_idx]["list_subtoken"] = list_subtoken
                        jsonl_data[js_idx]["list_subtoken_score"] = list_subtoken_score
                    
                    print(f"OOM detected on line {js_idx+1}, successfully processed with reduced data (top 5 instances)")
                    
                except Exception as e2:
                    if "out of memory" in str(e2).lower() or "oom" in str(e2).lower():
                        # Nuclear cleanup before second retry
                        nuclear_cleanup()
                        
                        try:
                            # Second fallback: potong jadi 3 instances
                            list_inference_input_reduced_3 = []
                            for i, attribute in enumerate(list_inference_attribute):
                                if attribute in ["list_frame_symbolic", "list_text_source"]:
                                    original_list = list_inference_input[i]
                                    reduced_list = original_list[:3] if isinstance(original_list, list) else original_list
                                    list_inference_input_reduced_3.append(reduced_list)
                                else:
                                    list_inference_input_reduced_3.append(list_inference_input[i])
                            
                            prompt_reduced_3 = get_prompt(list_inference_input_reduced_3, prompt_task_type, prompt_variant)
                            if return_mode == "without_subtoken_score":
                                original_answer = llm_inference_greedy_search(prompt_reduced_3,tokenizer,model,max_new_tokens,return_mode)
                                jsonl_data[js_idx]["original_answer"] = original_answer
                            elif return_mode == "with_subtoken_score":
                                original_answer, list_subtoken, list_subtoken_score = llm_inference_greedy_search(prompt_reduced_3,tokenizer,model,gpu_device,max_new_tokens,return_mode)
                                jsonl_data[js_idx]["original_answer"] = original_answer
                                jsonl_data[js_idx]["list_subtoken"] = list_subtoken
                                jsonl_data[js_idx]["list_subtoken_score"] = list_subtoken_score
                            
                            print(f"Double OOM detected on line {js_idx+1}, successfully processed with further reduced data (top 3 instances)")
                            
                        except:
                            # Final nuclear cleanup attempt
                            nuclear_cleanup()
                            
                            jsonl_data[js_idx]["original_answer"] = "failed_to_get_inference_result_even_with_top_3_instances"
                            print(f"Failed to get inference result even with top 3 instances on line {js_idx+1}")
                    else:
                        jsonl_data[js_idx]["original_answer"] = "failed_to_get_inference_result_after_first_reduction"
                        print(f"Failed after first reduction due to non-OOM error on line {js_idx+1}: {str(e2)}")
            else:
                jsonl_data[js_idx]["original_answer"] = "failed_to_get_inference_result"
                print(f"Failed to get inference result due to non-OOM error on line {js_idx+1}: {str(e)}")
        
        # Nuclear cleanup after each inference (regardless of success/failure)
        nuclear_cleanup()
    
    print("Inference bulk process is done.")
    return jsonl_data

def llm_inference_bulk_file2file(input_file_path,output_file_path,list_inference_attribute,prompt_task_type,prompt_variant,model_name,gpu_device="",hf_token="",max_new_tokens=250,return_mode="with_subtoken_score",verbose="yes"):
    inference_result = llm_inference_bulk(input_file_path,list_inference_attribute,prompt_task_type,prompt_variant,model_name,gpu_device,hf_token,max_new_tokens,return_mode,verbose)
    if verbose =="yes":
        print("Process to save inference result into desired destination.")
    write_jsonl(inference_result,output_file_path)
    if verbose == "yes":
        print(f"The inference result has been saved into: {output_file_path}")

def main(args):
    # Get all arguments
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    list_inference_attribute = args.list_inference_attribute
    list_inference_attribute = list_inference_attribute.split(",")
    prompt_task_type = args.prompt_task_type
    prompt_variant = args.prompt_variant
    model_name = args.model_name
    gpu_device = args.gpu_device
    if gpu_device == "":
        print("We will run inference process on CPU device.")
    else:
        print(f"We will run inference process on GPU: {gpu_device} device.")
    hf_token = args.hf_token
    max_new_tokens = args.max_new_tokens
    return_mode = args.return_mode
    verbose = args.verbose
    # Bulk inference process
    llm_inference_bulk_file2file(input_file_path,output_file_path,list_inference_attribute,prompt_task_type,prompt_variant,model_name,gpu_device,hf_token,max_new_tokens,return_mode,verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path", help="Your .jsonl input file path.",
        type=str
    )
    parser.add_argument(
        "--output_file_path", help="Your .jsonl output file path.",
        type=str
    )
    parser.add_argument(
        "--list_inference_attribute", help="Your inference attribute list. Please define as string separated by coma (,). Example: text,frame,polarity",
        type=str, default="text"
    )
    parser.add_argument(
        "--prompt_task_type", help="Your prompt task type. You can define your own prompt task type in get_prompt.py file.",
        type=str
    )
    parser.add_argument(
        "--prompt_variant", help="Your prompt task variant. You can define your own prompt task variant in get_prompt.py file.",
        type=int, default=1
    )
    parser.add_argument(
        "--model_name", help="Your model_name path. It can be your own local model or HuggingFace model name.",
        type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument(
        "--gpu_device", help="The maximum number of new token generated by the LLM.",
        type=str, default=""
    )
    parser.add_argument(
        "--hf_token", help="Your HuggingFace token (optional only when you use model from gated repository).",
        type=str, default=""
    )
    parser.add_argument(
        "--max_new_tokens", help="The maximum number of new token generated by the LLM.",
        type=int, default=250
    )
    parser.add_argument(
        "--return_mode", help="Your inference return mode, whether you only want get generated token (choose `without_subtoken_score`) or with the sub token score (choose `with_subtoken_score`).",
        type=str, default="with_subtoken_score"
    )
    parser.add_argument(
        "--verbose", help="Option for showing progress. Chose `yes` for showing complete progress for each sentence, chose `no` if you completely do not want to show the progress.",
        type=str, default="yes"
    )
    args = parser.parse_args()
    main(args)