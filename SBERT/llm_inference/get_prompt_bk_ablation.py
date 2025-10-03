# Add BacKGen's libraries path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import library
from llm_inference.get_prompt_utility import *

def get_prompt_bk_ablation(frame_list,text_list,frame_definition_list,polarity,prompt_variant=1):
    prompt_frame_definition = write_frame_definition(frame_definition_list)
    prompt_input_text = write_input_text(frame_list,text_list)
    prompt_input_text_only = write_input_text_only(text_list)
    prompt_input_frame_only = write_input_frame_only(frame_list)
    example_input_text_only,example_input_frame_only = get_input_example_ablation(polarity)
    example_frame_definition,example_input_text,example_answer = get_bk_example(polarity)
    # Frame + Text + Definition
    if prompt_variant == 1:
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the definitions of the involved frame(s):\n{example_frame_definition}Here are the input texts:\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}Here are the input texts:\n{prompt_input_text}Answer: "
        return prompt
    # Frame + Text
    elif prompt_variant == 2:
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the input texts:\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nHere are the input texts:\n{prompt_input_text}Answer: "
        return prompt
    # Frame + Definition
    elif prompt_variant == 3:
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input frames provided. These sentences are grouped by shared frames modeled according to Frame Semantics Theory. Each input frame explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the definitions of the involved frame(s):\n{example_frame_definition}Here are the input frames:\n{example_input_frame_only}Answer: {example_answer}\n\nYour Turn:\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}Here are the input frames:\n{prompt_input_frame_only}Answer: "
        return prompt
    # Frame
    elif prompt_variant == 4:
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input frames provided. These sentences are grouped by frames modeled according to Frame Semantics Theory. Each input frame explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the input frames:\n{example_input_frame_only}Answer: {example_answer}\n\nYour Turn:\nHere are the input frames:\n{prompt_input_frame_only}Answer: "
        return prompt
    # Text
    elif prompt_variant == 5:
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the input texts:\n{example_input_text_only}Answer: {example_answer}\n\nYour Turn:\nHere are the input texts:\n{prompt_input_text_only}Answer: "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_bk_ablation function. Please define your variant in that function.")
    

def get_prompt_bk_direct(text_list,polarity,prompt_variant=1):
    prompt_input_text_only = write_input_text_only(text_list)
    example_input_text_only,example_input_frame_only = get_input_example_ablation(polarity)
    example_frame_definition,example_input_text,example_answer = get_bk_example(polarity)
    # Frame + Text + Definition
    if prompt_variant == 1:
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations modeled according to embedding similarity. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the input texts:\n{example_input_text_only}Answer: {example_answer}\n\nYour Turn:\nHere are the input texts:\n{prompt_input_text_only}Answer: "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_bk_ablation function. Please define your variant in that function.")