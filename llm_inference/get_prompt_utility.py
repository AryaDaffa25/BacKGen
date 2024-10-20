def write_pk(pk_list):
    if pk_list == "No background knowledge can be provided.":
        return pk_list
    else:
        prompt_pk = ""
        for i in range(len(pk_list)):
            if i < len(pk_list)/2:
                polarity = "Negative"
            else:
                polarity = "Positive"
            prompt_pk = f"{prompt_pk}{i+1}. '{pk_list[i]}' (Sentiment: {polarity})\n"
        return prompt_pk

def write_pk_polarity_style(pk_list):
    if pk_list == "No background knowledge can be provided.":
        return "Background Knowledge: No background knowledge can be provided."
    else:
        prompt_pk = ""
        for i in range(len(pk_list)):
            if i < len(pk_list)/2:
                polarity = "Negative"
            else:
                polarity = "Positive"
            prompt_pk = f"{prompt_pk}Background Knowledge {i+1}:\n- Text: '{pk_list[i]}'\nPolarity: {polarity}\n"
        return prompt_pk

def write_example(negative_example_list,positive_example_list,promp_variant):
    if negative_example_list == "No example can be provided.":
        return negative_example_list
    else:
        examples = ""
        if promp_variant == 41:
            example_list = negative_example_list[:1]+positive_example_list[:1]
        else:
            example_list = negative_example_list+positive_example_list
        if promp_variant == 1 or promp_variant == 2:
            phrase_term_name = "Sentiment Phrase = "
            polarity_name = "Sentiment Subjectivity = "
        elif promp_variant == 3 or promp_variant == 41 or promp_variant == 42:
            phrase_term_name = "Target Phrase: "
            polarity_name = "Sentiment: "
        for i in range(len(example_list)):
            examples = f"{examples}{i+1}. {example_list[i][0]}. {phrase_term_name}{example_list[i][1]}. {polarity_name}{example_list[i][2]}\n"
        return examples

def write_example_polarity_style(negative_example_list,positive_example_list):
    if negative_example_list == "No example can be provided.":
        return "Example: No example can be provided."
    else:
        examples = ""
        example_list = negative_example_list+positive_example_list
        for i in range(len(example_list)):
            examples = f"{examples}Example {i+1}:\nText: '{example_list[i][0]}'\nTarget Phrase: '{example_list[i][1]}'\nPolarity: {example_list[i][2]}\n"
        return examples

def write_frame_definition(frame_definition_list):
    prompt_frame_definition = ""
    for fd in frame_definition_list:
        prompt_frame_definition = f"{prompt_frame_definition}- {fd[0]}: {fd[1]}\n"
    return prompt_frame_definition

def write_input_text(frame_list,text_list):
    prompt_input_text = ""
    for i in range(len(text_list)):
        frame_list_i = str(frame_list[i])
        frame_label = frame_list_i.split('(')[0]
        lu_span = frame_list_i.split('(LU(')[1].split(')')[0]
        try:
            roles = frame_list_i.split(f"(LU({lu_span}),")[1].replace('))',')')
            prompt_input_text = f"{prompt_input_text}{i+1}. {text_list[i]}\n\t- {frame_label}\n\t\t- Lexical Unit (LU): {lu_span}\n\t\t- Roles: {roles}\n"
        except:
            prompt_input_text = f"{prompt_input_text}{i+1}. {text_list[i]}\n\t- {frame_label}\n\t\t- Lexical Unit (LU): {lu_span}\n"
    return prompt_input_text

def write_input_text_only(text_list):
    prompt_input_text_only = ""
    for i in range(len(text_list)):
        prompt_input_text_only = f"{prompt_input_text_only}{i+1}. {text_list[i]}\n"
    return prompt_input_text_only

def write_input_frame_only(frame_list):
    prompt_input_frame = ""
    for i in range(len(frame_list)):
        frame_list_i = str(frame_list[i])
        frame_label = frame_list_i.split('(')[0]
        lu_span = frame_list_i.split('(LU(')[1].split(')')[0]
        try:
            roles = frame_list_i.split(f"(LU({lu_span}),")[1].replace('))',')')
            prompt_input_frame = f"{prompt_input_frame}{i+1}. {frame_label}\n\t- Lexical Unit (LU): {lu_span}\n\t- Roles: {roles}\n"
        except:
            prompt_input_frame = f"{prompt_input_frame}{i+1}. {frame_label}\n\t- Lexical Unit (LU): {lu_span}\n"
    return prompt_input_frame

def get_pk_example(polarity):
    if polarity == "negative":
        frame_definition_list = [['Causation','A Cause causes an Effect.'],['Destroying','A Destroyer (a conscious entity) or Cause (an event, or an entity involved in such an event) affects the Patient negatively so that the Patient no longer exists.'],['Cause_to_end','An Agent or Cause causes a Process or State to end.'],['Cause_to_amalgamate','These words refer to an Agent joining Parts to form a Whole.']]
        frame_list = ['Causation(LU(putting),Cause(water pollution),Effect(our health),Cause(at risk unsafe water kills more people each year than war and all other forms of violence combined))','Destroying(LU(destroying),Cause(pollution in all it s forms),Cause(which),Patient(our environment and health))','Cause_to_end(LU(end),Cause(technology),State(pollution of the air water soil))','Cause_to_amalgamate(LU(combined),Parts(all other forms of violence))']
        text_list = ['water pollution is putting our health at risk unsafe water kills more people each year than war and all other forms of violence combined here are six causes of water pollution as well as what we can do to reduce it','i hope izzy one day understands that we can be against pollution in all it s forms which truly is destroying our environment and health but also be smart enough to see through the carbon emissions global warming shenanigans','extinction is forever amp for all we know we have lost what we will need to fix things when it becomes obvious we have to do something technology will not end pollution of the air water soil or the contamination of our food earth cycles themselves will be the only way out of it','water pollution is putting our health at risk unsafe water kills more people each year than war and all other forms of violence combined here are six causes of water pollution as well as what we can do to reduce it']
        example_answer = 'The existence of pollution and other materials that cause damage and destroy our environment is very negative.'
    elif polarity == "positive":
        frame_definition_list = [['Cause_change_of_position_on_a_scale','This frame consists of words that indicate that an Agent or a Cause affects the position of an Item on some scale (the Attribute) to change it from an initial value (Value_1) to an end value (Value_2).']]
        frame_list = ['Cause_change_of_position_on_a_scale(LU(reducing),Attribute(its footprint))','Cause_change_of_position_on_a_scale(LU(reducing),Attribute(consumerism))','Cause_change_of_position_on_a_scale(LU(reduced),Agent(india),Attribute(emission intensity of its gdp),Difference(by 24 per cent),Speed(in 11 yrs),Time(through 2016),Means(un via official pollution))']
        text_list = ['if the tourism sector is serious about reducing its footprint they should choose real emission reductions and biodiversity protection even airlines are starting to move away from offsets fornature 4','moving away from capitalism green washing is not easy under the current systems political allegiances we live within so i commend for being bold enough to try but let us not forget that redistributing wealth and reducing consumerism must remain 1 priorities','india reduced emission intensity of its gdp by 24 per cent in 11 yrs through 2016 un via official pollution']
        example_answer = 'Reducing material that is bad for the environment is a positive act.'
    elif polarity == "neutral":
        frame_definition_list = [['Ordinal_numbers','An Item is picked out either by the order in which the members of a set would be encountered by an implicit cognizer, or by the order in which the members of a set participate in an event or state which serves as the Basis_of_order.']]
        frame_list = ['Ordinal_numbers(LU(2nd),Type(edition))','Ordinal_numbers(LU(tertiary),Type(climate change))','Ordinal_numbers(LU(last),Comparison_set(trip))']
        text_list = ['an expanded version of a recent lecture that traces the history of the primary secondary and tertiary climate change amp health effect framework from 1992 until 2nd edition climate change and global health eds butler and higgs to be published in 2023','an expanded version of a recent lecture that traces the history of the primary secondary and tertiary climate change amp health effect framework from 1992 until 2nd edition climate change and global health eds butler and higgs to be published in 2023','yeah you re right when 15 year old goes on holiday to place that people shouldn t actually go to due to saving the environment yet the last trip she takes is to that same location that s perfectly normal no irony in it just a minor oversight']
        example_answer = 'Stating information about a number in a fact or opinion is a neutral statement.'
    else:
        raise ValueError("Wrong polarity input. Only 'negative', 'positive', 'neutral' are allowed.")
    example_frame_definition = write_frame_definition(frame_definition_list)
    example_input_text = write_input_text(frame_list,text_list)
    return example_frame_definition,example_input_text,example_answer

def get_input_example_ablation(polarity):
    if polarity == "negative":
        frame_list = ['Causation(LU(putting),Cause(water pollution),Effect(our health),Cause(at risk unsafe water kills more people each year than war and all other forms of violence combined))','Destroying(LU(destroying),Cause(pollution in all it s forms),Cause(which),Patient(our environment and health))','Cause_to_end(LU(end),Cause(technology),State(pollution of the air water soil))','Cause_to_amalgamate(LU(combined),Parts(all other forms of violence))']
        text_list = ['water pollution is putting our health at risk unsafe water kills more people each year than war and all other forms of violence combined here are six causes of water pollution as well as what we can do to reduce it','i hope izzy one day understands that we can be against pollution in all it s forms which truly is destroying our environment and health but also be smart enough to see through the carbon emissions global warming shenanigans','extinction is forever amp for all we know we have lost what we will need to fix things when it becomes obvious we have to do something technology will not end pollution of the air water soil or the contamination of our food earth cycles themselves will be the only way out of it','water pollution is putting our health at risk unsafe water kills more people each year than war and all other forms of violence combined here are six causes of water pollution as well as what we can do to reduce it']
    elif polarity == "positive":
        frame_list = ['Cause_change_of_position_on_a_scale(LU(reducing),Attribute(its footprint))','Cause_change_of_position_on_a_scale(LU(reducing),Attribute(consumerism))','Cause_change_of_position_on_a_scale(LU(reduced),Agent(india),Attribute(emission intensity of its gdp),Difference(by 24 per cent),Speed(in 11 yrs),Time(through 2016),Means(un via official pollution))']
        text_list = ['if the tourism sector is serious about reducing its footprint they should choose real emission reductions and biodiversity protection even airlines are starting to move away from offsets fornature 4','moving away from capitalism green washing is not easy under the current systems political allegiances we live within so i commend for being bold enough to try but let us not forget that redistributing wealth and reducing consumerism must remain 1 priorities','india reduced emission intensity of its gdp by 24 per cent in 11 yrs through 2016 un via official pollution']
    elif polarity == "neutral":
        frame_list = ['Ordinal_numbers(LU(2nd),Type(edition))','Ordinal_numbers(LU(tertiary),Type(climate change))','Ordinal_numbers(LU(last),Comparison_set(trip))']
        text_list = ['an expanded version of a recent lecture that traces the history of the primary secondary and tertiary climate change amp health effect framework from 1992 until 2nd edition climate change and global health eds butler and higgs to be published in 2023','an expanded version of a recent lecture that traces the history of the primary secondary and tertiary climate change amp health effect framework from 1992 until 2nd edition climate change and global health eds butler and higgs to be published in 2023','yeah you re right when 15 year old goes on holiday to place that people shouldn t actually go to due to saving the environment yet the last trip she takes is to that same location that s perfectly normal no irony in it just a minor oversight']
    else:
        raise ValueError("Wrong polarity input. Only 'negative', 'positive', 'neutral' are allowed.")
    return write_input_text_only(text_list),write_input_frame_only(frame_list)