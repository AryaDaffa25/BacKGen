# Add library path
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path)

# Import library
from llm_inference.get_prompt_utility import *

def get_prompt_spc_zero_with_explanation(text,st_span,prompt_variant=1):
    if prompt_variant == 1: # This is prompt zero-shot in the paper
        prompt = f"Task: Determine the polarity (either 'positive' or 'negative') of the target phrase from the provided text. Then, provide a short explanation for your classification. The explanation should be clear and helpful for the user to understand the choice.\n\nInstructions:\n- The polarity output can only be 'positive' or 'negative'.\n- The first word of your answer should be your final polarity classification, then followed by your explanation.\n\nInput:\n- Text: '{text}'\n-Target Phrase: '{st_span}'\n\nAnswer: "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_spc_zero_with_explanation function. Please define your variant in that function.")

def get_prompt_spc_few_with_explanation(text,st_span,negative_example_list,positive_example_list,prompt_variant=11):
    if prompt_variant == 11: # This is prompt few-shot 1-shot in the paper
        prompt = f"Task: Determine the polarity (either 'positive' or 'negative') of the target phrase from the provided text. Then, provide a short explanation for your classification. You are also provided with some examples. The explanation should be clear and helpful for the user to understand the choice.\n\nInstructions:\n- Use the examples to help determine the polarity.\n- Note the sentiment of each example sentence as it may assist in your reasoning.\n- The polarity output can only be 'positive' or 'negative'.\n- The first word of your answer should be your final polarity classification, then followed by your explanation.\n- The user is not aware of the examples, so you cannot refer to them explicitly in your explanation.\n\nInput:\n- Text: '{text}'\n- Target Phrase: '{st_span}'\n\nExamples:\n1. {negative_example_list[0][0]}. Target Phrase: {negative_example_list[0][1]}. Sentiment: {negative_example_list[0][2]}\n2. {positive_example_list[0][0]}. Target Phrase: {positive_example_list[0][1]}. Sentiment: {positive_example_list[0][2]}\n\nAnswer: "
        return prompt
    elif prompt_variant == 12: # This is prompt few-shot 2-shot in the paper
        prompt = f"Task: Determine the polarity (either 'positive' or 'negative') of the target phrase from the provided text. Then, provide a short explanation for your classification. You are also provided with some examples. The explanation should be clear and helpful for the user to understand the choice.\n\nInstructions:\n- Use the examples to help determine the polarity.\n- Note the sentiment of each example sentence as it may assist in your reasoning.\n- The polarity output can only be 'positive' or 'negative'.\n- The first word of your answer should be your final polarity classification, then followed by your explanation.\n- The user is not aware of the examples, so you cannot refer to them explicitly in your explanation.\n\nInput:\n- Text: '{text}'\n- Target Phrase: '{st_span}'\n\nExamples:\n1. {negative_example_list[0][0]}. Target Phrase: {negative_example_list[0][1]}. Sentiment: {negative_example_list[0][2]}\n2. {negative_example_list[1][0]}. Target Phrase: {negative_example_list[1][1]}. Sentiment: {negative_example_list[1][2]}\n3. {positive_example_list[0][0]}. Target Phrase: {positive_example_list[0][1]}. Sentiment: {positive_example_list[0][2]}\n4. {positive_example_list[1][0]}. Target Phrase: {positive_example_list[1][1]}. Sentiment: {positive_example_list[1][2]}\n\nAnswer: "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_spc_pk_with_explanation function. Please define your variant in that function.")

def get_prompt_spc_bk_with_explanation(text,st_span,bk_list,prompt_variant=1):
    # For prompt_variant 2 and so on
    prompt_pk = write_bk(bk_list)
    if prompt_variant == 1: # This is prompt bk-shot in the paper
        prompt = f"Task: Determine the polarity (either 'positive' or 'negative') of the target phrase from the provided text. Then, provide a short explanation for your classification. You are also provided with potentially useful sentences reflecting background knowledge. The explanation should be clear and helpful for the user to understand the choice.\n\nInstructions:\n- Use the background knowledge to help determine the polarity.\n- Note the sentiment of each background sentence as it may assist in your reasoning.\n- The polarity output can only be 'positive' or 'negative'.\n- The first word of your answer should be your final polarity classification, then followed by your explanation.\n- The user is not aware of the background knowledge, so you cannot refer to it explicitly in your explanation.\n\nInput:\n- Text: '{text}'\n- Target Phrase: '{st_span}'\n\nBackground Knowledge:\n{prompt_pk}\nAnswer: "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_spc_pk_with_explanation function. Please define your variant in that function.")