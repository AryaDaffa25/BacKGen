# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import libraries
from llm_inference.get_prompt_utility import *
from llm_inference.get_prompt_spc import *
from llm_inference.get_prompt_bk_ablation import get_prompt_bk_ablation, get_prompt_bk_direct

def get_prompt_bk_with_example(frame_list,text_list,frame_definition_list,polarity,prompt_variant=1):
    # Generate attribute
    prompt_frame_definition = write_frame_definition(frame_definition_list)
    prompt_input_text = write_input_text(frame_list,text_list)
    example_frame_definition,example_input_text,example_answer = get_bk_example(polarity)
    if prompt_variant == 1: # This is prompt bk-1a in the paper
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the definitions of the involved frame(s):\n{example_frame_definition}Here are the input texts:\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}Here are the input texts:\n{prompt_input_text}Answer: "
        return prompt
    elif prompt_variant == 2: # This is prompt bk-1b in the paper
        prompt = f"Write a short paragraph expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the definitions of the involved frame(s):\n{example_frame_definition}Here are the input texts:\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}Here are the input texts:\n{prompt_input_text}Answer: "
        return prompt
    elif prompt_variant == 3: # This is prompt bk-2a in the paper
        prompt = f"Task: Write one sentence expressing general background knowledge about the world, based on the input sentences provided.\n\nInstructions:\n- Ensure that the generated text conveys a {polarity} sentiment and the reason for the sentiment should be made explicit.\n- Consider all input sentences, which are grouped according to frames defined by the Frame Semantics Theory. In each sentence, the Lexical Units (evoking the frames) and the corresponding roles are made explicit.\n- Consider the frame’s definitions, provided to guide the generalization process.\n- Do not explicitly mention the input prompt as the user is not aware of it.\n\nExample:\nFrame(s) definition:\n{example_frame_definition}Input text(s):\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nFrame(s) definition:\n{prompt_frame_definition}Input text(s):\n{prompt_input_text}Answer: "
        return prompt
    elif prompt_variant == 4: # This is prompt bk-2b in the paper
        prompt = f"Task: Write a short paragraph expressing general background knowledge about the world, based on the input sentences provided.\n\nInstructions:\n- Ensure that the generated text conveys a {polarity} sentiment and the reason for the sentiment should be made explicit.\n- Consider all input sentences, which are grouped according to frames defined by the Frame Semantics Theory. In each sentence, the Lexical Units (evoking the frames) and the corresponding roles are made explicit.\n- Consider the frame’s definitions, provided to guide the generalization process.\n- Do not explicitly mention the input prompt as the user is not aware of it.\n\nExample:\nFrame(s) definition:\n{example_frame_definition}Input text(s):\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nFrame(s) definition:\n{prompt_frame_definition}Input text(s):\n{prompt_input_text}Answer: "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_bk_with_example function. Please define your variant in that function.")

def get_prompt(list_inference_input,prompt_task_type,prompt_variant=1):
    if prompt_task_type == "bk_with_example":
        return get_prompt_bk_with_example(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    elif prompt_task_type == "bk_ablation":
        return get_prompt_bk_ablation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    elif prompt_task_type == "bk_direct":
        return get_prompt_bk_direct(list_inference_input[0],list_inference_input[1],prompt_variant)
    elif prompt_task_type == "spc_zero_with_explanation":
        return get_prompt_spc_zero_with_explanation(list_inference_input[0],list_inference_input[1],prompt_variant)
    elif prompt_task_type == "spc_bk_with_explanation":
        return get_prompt_spc_bk_with_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],prompt_variant)
    elif prompt_task_type == "spc_few_with_explanation":
        return get_prompt_spc_few_with_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    else:
        raise ValueError("Wrong `prompt_task_type`. Please check on `get_prompt.py` for the correct params or define your own prompt there.")
