# BaKGen
This repository contains the code and dataset for the paper **Modeling Background Knowledge with Frame Semantics for Fine-grained
Sentiment Classification** published in **Analogy-Angle II** workshop by *Muhammad Okky Ibrohim* (University of Turin), *Valerio Basile* (University of Turin), *Danilo Croce* (University of Rome "Tor Vergata"), *Cristina Bosco* (University of Turin), and *Roberto Basili* (University of Rome "Tor Vergata"). The paper will be available soon.

# What is BacKGen?
## Introduction
The problem of sample selection in few-shot prompting introduces knowledge and chain-of-taught prompting as the more robust alternative. Here, we propose BacKGen, a framework to generate background knowledge (BK) based on frame semantic theory that can be used for knowledge prompting. We tested the generated knowledge for knowledge prompting in solving Sentiment Phrase Classification (SPC), a task where the goal is to determine the sentiment polarity of a target phrase in a given text. The example below illustrates why knowledge prompting using BacKGen is particularly important for this task (these are abbreviated versions for illustration; full prompts are explained in the paper).
> Example of `zero-shot` SPC prompt:
>> Task: Determine the polarity (either 'positive' or 'negative') of the target phrase.
>> 
>> Input:
>> - Text: *"The government phases out fossil fuels."*
>> - Target Phrase: *"phases out fossil fuels"*
>> 
>> Model Output: negative

> Example of `bk-shot` SPC prompt with injected background knowledge:
>> Task: Determine the polarity (either 'positive' or 'negative') of the target phrase, using background knowledge if helpful.
>>
>> Input:
>> - Text: *"The government phases out fossil fuels."*
>> - Target Phrase: *"phases out fossil fuels"*
>>
>> Background Knowledge:
>> - The fact that a public entity wants to remove something related to green initiatives is perceived negatively.
>> - Public entities’ intention to reduce non-renewable energy sources is seen as a positive step.
>>
>> Model Output: positive

From the example above, we can see that SPC becomes especially challenging when the sentiment of a phrase is context-dependent or ambiguous. Here, the target phrase *"phases out fossil fuels"* might be misclassified as negative in a zero-shot setting, as *"phases out"* often conveys abandonment. However, in the context of environmental policy, the action of phasing out fossil fuels is typically seen in a positive light. In this case, injecting BK into the prompt can guide the model toward the correct interpretation. Statements such as *"public entities’ intention to reduce non-renewable energy sources is seen as a positive step"* help contextualize the sentiment, enabling the model to move beyond surface-level heuristics. This example demonstrates how BK can resolve subtle ambiguities in sentiment interpretation and reinforces our motivation for replacing concrete examples with structured, generalizable knowledge. The experiment results demonstrate that BK-based prompting consistently outperforms standard few-shot approaches, achieving up to 29.94% error reduction.

## How BacKGen Works
Using BacKGen for knowledge prompting consists of two main processes, namely **BK Generation** (the process that is done to generate BK) and **BK Selection** (the process that is done to select BK that will be used for knowledge prompting). The **BK Generation** can be seen in the figure below:
![alt text](https://github.com/crux82/BacKGen/blob/main/readme_images/bkgenerationflow.png)

Given a set of annotated examples, the first step of **BK Generation** is to perform **Frame-based Parsing**. In this case, we extract the [frame semantics](https://www.cs.toronto.edu/~gpenn/csc2519/fillmore85.pdf) that consists of *Lexical Unit* (LU) and its associated *Frame Elements* (FEs) using a particular frame parser (in our paper, we use [LOME frame parser](https://aclanthology.org/2021.eacl-demos.19/)). The next step is to perform **Frame-based Clustering**, which aims to group similar frames to identify shared conceptual structures. Note that measuring the similarity between these structured representations requires a metric sensitive to both tree structure and semantic similarity of role fillers. In our paper, we employ the [Smoothed Partial Tree Kernel (SPTK)](https://aclanthology.org/D11-1096/), a method that evaluates the similarity of two trees by counting the number of shared substructures, while also weighting the contribution of lexically different but semantically related elements. For the clustering process, we use [Kernel-based k-means](https://dl.acm.org/doi/10.1145/1014052.1014118) that is implemented in [KELP](https://jmlr.org/papers/v18/16-087.html). Lastly, we perform **Background Knowledge Generation** by utilizing a generative model to verbalize the common information in each cluster into reusable BK.

To perform knowledge prompting using the generated BK, we need to retrieve a set of BK that need to be injected to the prompt, like we give example in classical few-shot. Therefore, an efficient retrieval strategy is needed that allows selecting representative knowledge from the BK collection. In our paper, we use two methods:
- Frame-based Similarity (`Kernel`): Utilize the [SPTK](https://aclanthology.org/D11-1096/) to calculate kernel similarity of frame semantic structure between the text instance and the medoid.
- Text-similarity via Sentence-BERT (`TSim`): Calculate cosine similarity of [Sentence-BERT](https://aclanthology.org/D19-1410/) embedding between the text instance and the medoid.

## Experiment Result
We evaluate the effectiveness of BacKGen using [Mistral-7B]\(https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and [Llama3-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Each model is employed both for generating background knowledge (BK) and for performing SPC in the environmental sustainability domain, ensuring a consistent evaluation across the entire pipeline. The evaluation follows a 5-fold cross-validation setup. For each fold, BacKGen is applied to 4/5 of the dataset (training set) to generate a BK database, while the remaining 1/5 is used for testing. The models are tested under different prompting conditions, including `0-shot`, `few-shot` with random example selection (`Rand`), `few-shot` with `Tsim` example selection, `bk-shot` (our proposed BK-based prompting) with `Kernel` example selection, and `bk-shot` with `Tsim` example selection. 

# Minimum Working Example (MWE)
In this MWE, we demonstrate how to use BacKGen to generate BK and use it for SPC task. Given a set of texts, our goal in this demonstration is to generate two BK databases i.e. BK for positive and negative polarity.

## Requirements
BacKGen is implemented in `Python 3` (tested on `Python 3.8.12` and `Python 3.10.2`) and requires a number of packages. To install the packages, simply run `$ pip install -r requirements.txt` in your virtual environment. For the `torch` version, you may need a different one following your `cuda` version. For frame clustering, you need `java` installed in your environment, where in our case we use `java 22`.

## BK Generation process
In general, generating BK using BacKGen consists of four major steps:
1. `Frame Extraction`: Given a set of texts, the first step is to extract the frame semantics of the frame. In this MWE, we use [LOME frame parser](https://aclanthology.org/2021.eacl-demos.19/) to extract the frame from text in `$ data/mwe/0_spc_data/mwe-train.jsonl` and store the result in `$ data/mwe/backgen_1_extracted_frame/mwe-train-frame.jsonl`.
2. `Frame Filtering`: In this MWE, we want to generate BK for SPC, we filter the `frame_file` (`$ data/mwe/backgen_1_extracted_frame/mwe-train-frame.jsonl`) to only use the frame that the frame element intersects with the target (sentiment) phrase in the `train_file` (`$ data/mwe/0_spc_data/mwe-train.jsonl`) by running the function `jsonl2jsonl_filter_sentiment_frame(train_file,frame_file,output_folder)` that written in `$ filterer/filter_sentiment_frame.py` file script. In this MWE, the filtered frames stored in the `output_folder` `$ data/mwe/backgen_2_filtered_frame`. As the frame clustering process needs `.klp` format, we convert the `.jsonl` filtered frame into the desired `.klp` format by running the function `jsonl2klp_frame2klp(input_file_path,output_file_path,bracket_type="round")` in the file script `$ converter/frame_converter.py`.
3. `Frame Clustering`: For this process, we use kernel-based k-Means clustering provided by [KELP](https://jmlr.org/papers/v18/16-087.html). Suppose that we want to cluster the `positive_frame` in file `$ mwe/backgen_2_filtered_frame/mwe-train-frame_positive.klp` into 195 clusters with 10 of k-Means iteration process and then store the `clustering_result` in file `$ mwe/backgen_3_clustered_frame/mwe-train-clustered_positive.csv`, we move our working directory to `$ cd clusterer` folder and run the script `$ java -cp .:./lib/kelp-additional-algorithms-2.2.2.jar:lib/kelp-core-2.2.2.jar:lib/kelp-full2.0.2.jar KernelBasedFrameClustering positive_frame 195 10 > clustering_result`. As we do not need the singleton and need `.jsonl` format for the BK generation process, we filter and convert the clustered file into the desired `.jsonl` format by running the function `framecluster2jsonl(input_file_path,output_file_path,cluster_name,cluster_polarity)` in file script `$ converter/framecluster2jsonl.py`. This process is repeated for each label data, which in this MWE is repeated for the filtered negative polarity frame.
4. `BK Generation`: Using the file script `$ llm_inference/llm_inference_bulk.py`, we generate the BK for each cluster polarity. For custom prompts, we can define our prompt in the file script `$ llm_inference/get_prompt.py`. As the LLM output may contain noise, we clean the BK using the function `bk_preprocessor_file2file(input_file_path,output_file_path,clean_answer_mode)` in the file script `$ converter/pk_preprocessor.py`. Our generated BK in this MWE is stored in the folder `$ data/mwe/backgen_4_generated_bk`. If we later want to perform BK selection using frame filtering, we need to convert the generated BK to `.klp` format using the function `jsonlbk2klp(input_file_path,output_file_path)` written in the file script `$ converter/frame_converter.py`.
## BK Selection and Injection Process
BacKGen provides two approaches for BK selection i.e. based on text similarity and frame similarity. The BK selection based on text similarity is pretty direct, where we simply calculate the text-similarity of text instance with each original text of medoid and take the top-n BK based on the similarity score rank. Please explore and use the file script `$ get_bk/get_bk_exmbert_spc.py`. In this MWE, we demonstrate BK selection and injection based on frame similarity, which in general consists of five major steps:
1. `Frame Extraction`: Using [LOME frame parser](https://aclanthology.org/2021.eacl-demos.19/), we extract the text instance in file `$ data/mwe/0_spc_data/mwe-val.jsonl` and stored it in file `$ data/mwe/spc_1_extracted_frame/mwe-val-frame.jsonl`.
2. `Frame Filtering`: We filter the `frame_file` (`$ data/mwe/spc_1_extracted_frame/mwe-val-frame.jsonl`) based on the `target_phrase_file` (`$ data/mwe/0_spc_data/mwe-val-spc.jsonl`) using the function `jsonl2jsonl_filter_phrase_frame_st(target_phrase_file,frame_file,output_file)` written in the file script `$ filterer/filter_frame_splitted_st.py` and will produce the filtered frame `$ data/mwe/spc_2_filtered_frame/mwe-val-frame_filtered.jsonl`. As frame similarity scoring needs a `.klp` format, we convert the `.jsonl` filtered frame into the desired `.klp` format using the function `split_frame_jsonl2klp(input_file_path,output_file_path,bracket_type="round")` written in the file script `$ converter/frame_converter.py`.
3. `Frame Similarity Scoring`: We calculate the frame similarity of the filtered frame with each medoid frame using tree-kernel-based similarity provided by [KELP](https://jmlr.org/papers/v18/16-087.html). Suppose we want to calculate the frame similarity between the `filtered_frame` in the file `$ data/mwe/spc_2_filtered_frame/mwe-val-frame_filtered.jsonl` with the medoid frame of the `positive_cluster` in the file `$ data/mwe/backgen_4_generated_bk/mwe-bk_positive.klp` and store the `similarity_score` in the file `$ data/mwe/spc_3_sim_score/mwe-sim_score_positive.txt`, we move our working directory to `$ cd clusterer` folder and run the script `$ java -cp .:./lib/kelp-additional-algorithms-2.2.2.jar:lib/kelp-core-2.2.2.jar:lib/kelp-full2.0.2.jar KernelSimilarity filtered_frame positive_cluster similarity_score frame_syntaxtree`. This process is repeated for the `negative_cluster` so that we obtain two similarity score files as seen in the folder `$ data/mwe/spc_3_sim_score`.
4. `BK Selection`: From the obtained similarity score files, we can retrieve top-n BK for each BK's polarity database and add it to the SPC file instance `$ data/mwe/0_spc_data/mwe-val-spc.jsonl` by using the function `get_bk_spc(args)` written in the file `$ get_bk/get_bk_spc.py`. The result of this process is the original SPC file with the extra BK attribute as can be seen in the file `$ data/mwe/spc_4_selected_bk/mwe-val-bk_instance.jsonl`.
5. `BK Injection`: Using the file script `$ llm_inference/llm_inference_bulk.py` we inject the selected BK and perform bk-shot prompting using a particular LLM.  For custom prompts, we can define our prompt in the file `$ llm_inference/get_prompt_spc.py`. The output example of this process can be seen in the file `$ data/mwe/spc_5_bk_shot_result/mwe-val-bk_shot_result.jsonl`. Finally, we can post-process (to get the final answer) and evaluate the output using a particular post-processing and evaluation metric as we need.

# Citation
To cite the paper, please use the following:
```
@inproceedings{ibrohim-etal-2025-backgen,
    title = "Modeling Background Knowledge with Frame Semantics for Fine-grained Sentiment Classification",
    author = "Muhammad Okky Ibrohim and Valerio Basile and Danilo Croce and Cristina Bosco and Roberto Basili",
    editor = "Filip Ilievski and Giulia Rambelli and Marianna Bolognesi and Ute Schmid and Pia Sommerauer",
    booktitle = "Proceedings of The Second Workshop on Analogical Abstraction in Cognition, Perception, and Language (Analogy-Angle II)",
    month = july,
    year = "2025",
    address = "Vienna, Austria",
}
```
