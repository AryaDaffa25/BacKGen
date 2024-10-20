from llm_inference.get_prompt_utility import *
from llm_inference.get_prompt_splitted_stc import *
from llm_inference.get_prompt_pk_ablation import get_prompt_pk_ablation

def get_prompt_sentiment_zero_with_explanation(text,prompt_variant=1):
    if prompt_variant == 1:
        prompt = f"Given the text below, please classify whether the sentiment of the text is 'positive', 'neutral', or 'negative' and explain the reason in a single sentence.\nText = {text}\nAnswer = "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_ste_zero_without_pk function. Please define your variant in that function.")

def get_prompt_sentiment_pk_with_explanation(text,pk_list,prompt_variant=1):
    if prompt_variant == 1:
        prompt = f"Given the text and the prototypical knowledge list below, please please classify whether the sentiment of the text is 'positive', 'neutral', or 'negative' and explain the reason in a single sentence. You can use the prototypical knowledge as the general context when you classify the sentiment of the text and explain the reason for your classification result.\nText = {text}\nPrototypical Knowledge List = {str(pk_list)}\nAnswer = "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_ste_zero_without_pk function. Please define your variant in that function.")

def get_prompt_sentiment_zero_without_explanation(text,prompt_variant=1):
    if prompt_variant == 1:
        prompt = f"Given the text below, please classify whether the sentiment of the text is 'positive', 'neutral', or 'negative' without any explanation.\nText = {text}\nAnswer = "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_ste_zero_without_pk function. Please define your variant in that function.")

def get_prompt_stc_zero_with_explanation(text,ste_list,prompt_variant=1):
    # Get sentiment phrase list
    sentiment_phrase = [ste[0] for ste in ste_list]
    if prompt_variant == 1:
        prompt = f"Given the text and its list of sentiment phrases below,  please determine the sentiment subjectivity for each sentiment phrase whether it is 'positive' or 'negative' and explain in a single sentence for each of them.\nText = {text}\nSentiment Phrase List = {str(sentiment_phrase)}\nAnswer = "
        return prompt
    elif prompt_variant == 2:
        prompt = f"Given the text and its list of sentiment phrases below,  please determine the sentiment subjectivity for each sentiment phrase whether it is 'positive' or 'negative' and explain in a single sentence for each of them. Please answer by giving two lists i.e., a list of sentiment subjectivity classification and a list of your explanation.\nText = {text}\nSentiment Phrase List = {str(sentiment_phrase)}\nAnswer = "
        return prompt
    elif prompt_variant == 3:
        prompt = f"Given the text and its list of sentiment phrases below, determine the sentiment subjectivity for each sentiment phrase whether it is 'positive' or 'negative' and explain in a single sentence for each of them. Please answer with the format ['sentiment subjectivity', 'explanation'].\nText = {text}\nSentiment Phrase List = {str(sentiment_phrase)}\nAnswer = "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_ste_zero_without_pk function. Please define your variant in that function.")

def get_prompt_stc_pk_with_explanation(text,ste_list,pk_list,prompt_variant=1):
    # Get sentiment phrase list
    sentiment_phrase = [ste[0] for ste in ste_list]
    if prompt_variant == 1:
        prompt = f"Given the text and its list of sentiment phrases below, determine the sentiment subjectivity for each sentiment phrase whether it is 'positive' or 'negative' and explain in a single sentence for each of them. Please answer with the format ['sentiment subjectivity', 'explanation']. To help you do this task, we also provided a list of background knowledge about sentiment stereotypes that you can use as the context when you classify the sentiment subjectivity of the term given and explain the reason for your classification result.\nText = {text}\nSentiment Phrase List = {str(sentiment_phrase)}\nBackground Knowledge List = {str(pk_list)}\nAnswer = "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_ste_zero_without_pk function. Please define your variant in that function.")

def get_prompt_ste_zero_with_explanation(text,prompt_variant=1):
    if prompt_variant == 1:
        prompt = f"Given the text below, please extract all sentiment terms and its polarities with the format ['sentiment term', 'polarity'] where the polarity options is ['positive', 'negative']. Give your explanation for each sentiment term and polarity pair extracted in a single sentence. If no sentiment term exists, then only answer '[]' without explanation.\nText = {text}\nAnswer = "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_ste_zero_without_pk function. Please define your variant in that function.")

def get_prompt_pk_without_example(frame_list,text_list,frame_definition_list,polarity,prompt_variant=1):
    if prompt_variant == 1:
        prompt_frame_definition = write_frame_definition(frame_definition_list)
        prompt_input_text = write_input_text(frame_list,text_list)
        prompt = f"Write a short paragraph expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}\nHere are the input texts:\n{prompt_input_text}"
        return prompt
    elif prompt_variant == 2:
        prompt_frame_definition = write_frame_definition(frame_definition_list)
        prompt_input_text = write_input_text(frame_list,text_list)
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}\nHere are the input texts:\n{prompt_input_text}"
        return prompt
    elif prompt_variant == 3:
        prompt_frame_definition = write_frame_definition(frame_definition_list)
        prompt_input_text = write_input_text(frame_list,text_list)
        prompt = f"Task: Write a short paragraph expressing general background knowledge about the world, based on the input sentences provided.\n\nInstructions:\n- Ensure that the generated text conveys a {polarity} sentiment and the reason for the sentiment should be made explicit.\n- Consider all input sentences, which are grouped according to frames defined by the Frame Semantics Theory. In each sentence, the Lexical Units (evoking the frames) and the corresponding roles are made explicit.\n- Consider the frame’s definitions, provided to guide the generalization process.\n- Do not explicitly mention the input prompt as the user is not aware of it.\n\nFrame(s) definition:\n{prompt_frame_definition}\nInput text(s):\n{prompt_input_text}"
        return prompt
    elif prompt_variant == 4:
        prompt_frame_definition = write_frame_definition(frame_definition_list)
        prompt_input_text = write_input_text(frame_list,text_list)
        prompt = f"Task: Write one sentence expressing general background knowledge about the world, based on the input sentences provided.\n\nInstructions:\n- Ensure that the generated text conveys a {polarity} sentiment and the reason for the sentiment should be made explicit.\n- Consider all input sentences, which are grouped according to frames defined by the Frame Semantics Theory. In each sentence, the Lexical Units (evoking the frames) and the corresponding roles are made explicit.\n- Consider the frame’s definitions, provided to guide the generalization process.\n- Do not explicitly mention the input prompt as the user is not aware of it.\n\nFrame(s) definition:\n{prompt_frame_definition}\nInput text(s):\n{prompt_input_text}"
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_pk_without_example function. Please define your variant in that function.")

def get_prompt_pk_with_example(frame_list,text_list,frame_definition_list,polarity,prompt_variant=1):
    # Generate pairs of frame semantics and their text sources for variant 2
    for i in range(len(frame_list)):
        if i == 0:
            frame_text_without_and = f"'{frame_list[i]}' from text '{text_list[i]}';"
        elif i == len(frame_list) - 1:
            frame_text_with_and = f"{frame_text_without_and} and '{frame_list[i]}' from text '{text_list[i]}'."
            frame_text_without_and = f"{frame_text_without_and} '{frame_list[i]}' from text '{text_list[i]}'."
        else:
            frame_text_without_and = f"{frame_text_without_and} '{frame_list[i]}' from text '{text_list[i]}';"
    # Generate attribute for variant 3-6
    prompt_frame_definition = write_frame_definition(frame_definition_list)
    prompt_input_text = write_input_text(frame_list,text_list)
    example_frame_definition,example_input_text,example_answer = get_pk_example(polarity)
    # The prompt variant below is baseline for PK Generation by only giving the frame semantic list where the example taken from dummy manual cluster
    if prompt_variant == 1:
        prompt = f"Given the frame semantic list and the sentiment polarity below, please generate general prototypical knowledge in a single sentence without explanation that generelize the relation of the frame semantic list (the list of frame label and its lexical unit (LU) and other arguments) with sentiment polarity given.\n\nExample 1:\nFrame Semantic List = [Cause_change_of_position_on_a_scale(LU(reduce),Attribute(energybills)),Cause_change_of_position_on_a_scale(LU(reduce),Attribute(plastic waste),Place(in the pacific)),Cause_change_of_position_on_a_scale(LU(reducing),Attribute(its footprint)),Cause_change_of_position_on_a_scale(LU(reduce),Agent(the ball),Attribute(ghgs bipartisanclimate)),Cause_change_of_position_on_a_scale(LU(reduces),Attribute(costs),Attribute(the deficit))]\nSentiment Polarity = positive\nAnswer = Reducing material that bad for the environment is a positive act.\n\nExample 2:\nFrame Semantic List = [Cause_change_of_position_on_a_scale(LU(tripling),Agent(party),Attribute(the carbon tax)),Cause_change_of_position_on_a_scale(LU(doubling),Agent(party),Attribute(the gst),Means(while tripling the carbon tax)),Cause_change_of_position_on_a_scale(LU(increasing),Agent(we),Attribute(methane leakage),Means(with fracking)),Cause_change_of_position_on_a_scale(LU(increases),Cause(carbon taxes),Cause(that),Attribute(the cost of everything)),Cause_change_of_position_on_a_scale(LU(increase),Item(the methane released),Attribute(denmarks climate impact),Time(this year))]\nSentiment Polarity = negative\nAnswer = Increasing material that bad for the environment is a negative act.\n\nExample 3:\nFrame Semantic List = [Questioning(LU(asked),Speaker(we),Message(what they want the nation to know)),Questioning(LU(history),Topic(history))]\nSentiment Polarity = neutral\nAnswer = Questioning about something is a neutral statement.\n\nYour turn:\nFrame Semantic List = {str(frame_list)}\nSentiment Polarity = {polarity}\nAnswer = "
        return prompt
    # The prompt variant below is PK Generation by giving the frame semantics and their text source where the example taken from real clustering result of NormPTK
    elif prompt_variant == 2:
        prompt = f"Given several pairs of frame semantics and their source text where the frame semantics come from, and their sentiment subjectivity, please write a general background knowledge that represents sentiment subjectivity stereotypes from that information in a single sentence without explanation. To help you do this task, we give you three examples.\nExamples 1:\nFrame semantics and their text sources = 'Cause_change_of_position_on_a_scale(LU(reducing),Attribute(its footprint))' from the text 'if the tourism sector is serious about reducing its footprint they should choose real emission reductions and biodiversity protection even airlines are starting to move away from offsets fornature 4'; 'Cause_change_of_position_on_a_scale(LU(reducing),Attribute(consumerism))' from the text 'moving away from capitalism green washing is not easy under the current systems political allegiances we live within so i commend for being bold enough to try but let us not forget that redistributing wealth and reducing consumerism must remain 1 priorities'; and 'Cause_change_of_position_on_a_scale(LU(reduced),Agent(india),Attribute(emission intensity of its gdp),Difference(by 24 per cent),Speed(in 11 yrs),Time(through 2016),Means(un via official pollution))' from the text 'india reduced emission intensity of its gdp by 24 per cent in 11 yrs through 2016 un via official pollution'.\nSentiment subjectivity = positive\nAnswer = Reducing material that is bad for the environment is a positive act.\nExample 2:\nFrame semantics and their text sources = 'Causation(LU(putting),Cause(water pollution),Effect(our health),Cause(at risk unsafe water kills more people each year than war and all other forms of violence combined))' from the text 'water pollution is putting our health at risk unsafe water kills more people each year than war and all other forms of violence combined here are six causes of water pollution as well as what we can do to reduce it'; 'Destroying(LU(destroying),Cause(pollution in all it s forms),Cause(which),Patient(our environment and health))' from the text 'i hope izzy one day understands that we can be against pollution in all it s forms which truly is destroying our environment and health but also be smart enough to see through the carbon emissions global warming shenanigans'; 'Cause_to_end(LU(end),Cause(technology),State(pollution of the air water soil))' from the text 'extinction is forever amp for all we know we have lost what we will need to fix things when it becomes obvious we have to do something technology will not end pollution of the air water soil or the contamination of our food earth cycles themselves will be the only way out of it';  and 'Cause_to_amalgamate(LU(combined),Parts(all other forms of violence))' from the text 'water pollution is putting our health at risk unsafe water kills more people each year than war and all other forms of violence combined here are six causes of water pollution as well as what we can do to reduce it'.\nSentiment subjectivity = negative\nAnswer = The existence of pollution and other materials that cause damage and destroy our environment is very negative.\nExample 3:\nFrame semantics and their text sources = 'Ordinal_numbers(LU(2nd),Type(edition))' from the text 'an expanded version of a recent lecture that traces the history of the primary secondary and tertiary climate change amp health effect framework from 1992 until 2nd edition climate change and global health eds butler and higgs to be published in 2023'; 'Ordinal_numbers(LU(tertiary),Type(climate change))' from the text 'an expanded version of a recent lecture that traces the history of the primary secondary and tertiary climate change amp health effect framework from 1992 until 2nd edition climate change and global health eds butler and higgs to be published in 2023'; and 'Ordinal_numbers(LU(last),Comparison_set(trip))' from the text 'yeah you re right when 15 year old goes on holiday to place that people shouldn t actually go to due to saving the environment yet the last trip she takes is to that same location that s perfectly normal no irony in it just a minor oversight'\nSentiment subjectivity = neutral\nAnswer = Stating information about a number in a fact or opinion is a neutral statement.\nYour Turn:\nFrame semantics and their text sources = {frame_text_with_and}\nSentiment subjectivity = {polarity}\nAnswer = "
        return prompt
    elif prompt_variant == 3:
        prompt = f"Write a short paragraph expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the definitions of the involved frame(s):\n{example_frame_definition}Here are the input texts:\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}Here are the input texts:\n{prompt_input_text}Answer: "
        return prompt
    elif prompt_variant == 4:
        prompt = f"Write one sentence expressing general background knowledge that reflects stereotypical information, based on the input sentences provided. These sentences are grouped by shared situations (or frames) modeled according to Frame Semantics Theory. Each input sentence explicitly indicates the Lexical Unit (evoking the frames) and the corresponding role. Definitions of the frames will also be provided to guide the generation. Ensure that the generated text conveys a {polarity} sentiment and the reason of the sentiment should be made explicit.\n\nExample:\nHere are the definitions of the involved frame(s):\n{example_frame_definition}Here are the input texts:\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nHere are the definitions of the involved frame(s):\n{prompt_frame_definition}Here are the input texts:\n{prompt_input_text}Answer: "
        return prompt
    elif prompt_variant == 5:
        prompt = f"Task: Write a short paragraph expressing general background knowledge about the world, based on the input sentences provided.\n\nInstructions:\n- Ensure that the generated text conveys a {polarity} sentiment and the reason for the sentiment should be made explicit.\n- Consider all input sentences, which are grouped according to frames defined by the Frame Semantics Theory. In each sentence, the Lexical Units (evoking the frames) and the corresponding roles are made explicit.\n- Consider the frame’s definitions, provided to guide the generalization process.\n- Do not explicitly mention the input prompt as the user is not aware of it.\n\nExample:\nFrame(s) definition:\n{example_frame_definition}Input text(s):\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nFrame(s) definition:\n{prompt_frame_definition}Input text(s):\n{prompt_input_text}Answer: "
        return prompt
    elif prompt_variant == 6:
        prompt = f"Task: Write one sentence expressing general background knowledge about the world, based on the input sentences provided.\n\nInstructions:\n- Ensure that the generated text conveys a {polarity} sentiment and the reason for the sentiment should be made explicit.\n- Consider all input sentences, which are grouped according to frames defined by the Frame Semantics Theory. In each sentence, the Lexical Units (evoking the frames) and the corresponding roles are made explicit.\n- Consider the frame’s definitions, provided to guide the generalization process.\n- Do not explicitly mention the input prompt as the user is not aware of it.\n\nExample:\nFrame(s) definition:\n{example_frame_definition}Input text(s):\n{example_input_text}Answer: {example_answer}\n\nYour Turn:\nFrame(s) definition:\n{prompt_frame_definition}Input text(s):\n{prompt_input_text}Answer: "
        return prompt
    else:
        raise ValueError(f"You choose variant {prompt_variant} but we do not have it in get_prompt_pk_with_example function. Please define your variant in that function.")

def get_prompt(list_inference_input,prompt_task_type,prompt_variant=1):
    if prompt_task_type == "sentiment_zero_with_explanation":
        return get_prompt_sentiment_zero_with_explanation(list_inference_input[0],prompt_variant)
    elif prompt_task_type == "sentiment_pk_with_explanation":
        return get_prompt_sentiment_pk_with_explanation(list_inference_input[0],list_inference_input[1],prompt_variant)
    elif prompt_task_type == "sentiment_zero_without_explanation":
        return get_prompt_sentiment_zero_without_explanation(list_inference_input[0],prompt_variant)
    elif prompt_task_type == "splitted_stc_zero_without_explanation":
        return get_prompt_splitted_stc_zero_without_explanation(list_inference_input[0],list_inference_input[1],prompt_variant)
    elif prompt_task_type == "splitted_stc_pk_without_explanation":
        return get_prompt_splitted_stc_pk_without_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],prompt_variant)
    elif prompt_task_type == "splitted_stc_ex_without_explanation":
        return get_prompt_splitted_stc_ex_without_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    elif prompt_task_type == "splitted_stc_pk_ex_without_explanation":
        return get_prompt_splitted_stc_pk_ex_without_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],list_inference_input[4],prompt_variant)
    elif prompt_task_type == "splitted_stc_ex_pk_without_explanation":
        return get_prompt_splitted_stc_ex_pk_without_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],list_inference_input[4],prompt_variant)
    elif prompt_task_type == "splitted_stc_zero_with_explanation":
        return get_prompt_splitted_stc_zero_with_explanation(list_inference_input[0],list_inference_input[1],prompt_variant)
    elif prompt_task_type == "splitted_stc_pk_with_explanation":
        return get_prompt_splitted_stc_pk_with_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],prompt_variant)
    elif prompt_task_type == "splitted_stc_ex_with_explanation":
        return get_prompt_splitted_stc_ex_with_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    elif prompt_task_type == "splitted_stc_pk_ex_with_explanation":
        return get_prompt_splitted_stc_pk_ex_with_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],list_inference_input[4],prompt_variant)
    elif prompt_task_type == "stc_zero_with_explanation":
        return get_prompt_stc_zero_with_explanation(list_inference_input[0],list_inference_input[1],prompt_variant)
    elif prompt_task_type == "stc_pk_with_explanation":
        return get_prompt_stc_pk_with_explanation(list_inference_input[0],list_inference_input[1],list_inference_input[2],prompt_variant)
    elif prompt_task_type == "ste_zero_with_explanation":
        return get_prompt_ste_zero_with_explanation(list_inference_input[0],prompt_variant)
    elif prompt_task_type == "pk_with_example":
        return get_prompt_pk_with_example(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    elif prompt_task_type == "pk_without_example":
        return get_prompt_pk_without_example(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    elif prompt_task_type == "pk_ablation":
        return get_prompt_pk_ablation(list_inference_input[0],list_inference_input[1],list_inference_input[2],list_inference_input[3],prompt_variant)
    else:
        raise ValueError("Wrong `prompt_task_type`. Please check on `get_prompt.py` for the correct params or define your own prompt there.")
