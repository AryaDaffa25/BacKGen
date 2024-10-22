# BaKGen
A framework to generate background knowledge (BK) for knowledge prompting for a particular NLP downstream task. In this repository, we demonstrate how to generate BK using BacKGen and implement it for sentiment phrase classification (SPC) task.
## Brief Introduction
The problem of sample selection in few-shot prompting introduces knowledge and chain-of-taught prompting as the more robust alternative. Here, we propose BacKGen, a framework to generate background knowledge (BK) based on frame semantic theory that can be used for knowledge prompting. We tested the generated knowledge for knowledge prompting in solving SPC. The experiment results showed that the BK added in the prompt increased the performance with a significant impact on error reduction rate.
## Main Use
Using BacKGen for knowledge prompting consists of two main processes: <br />
1. `BK Generation`: This process is done to generate BK. Suppose that we have a collection of frames extracted from texts using a particular parser, we use BacKGen to cluster filter and cluster them. Then, we generate BK from the clustered frame using a particular LLM with a particular prompt template.
2. `BK Selection and Injection`: This process is done for bk injection shot (bk-shot). Suppose that we already obtain the generated BK, we select the top-n BK for each label based on a particual similarity function (frame or text similarity), then inject it to the prompt.
## About Dataset
In folder `data`, we provided three sub-folders: <br />
1. `5_fold_bk`: This folder contains the BK database that can be directly used for sentiment analysis on environmental sustainability (ES) issues using bk-shot.
2. `5_fold_frame_spc`: This folder contains our 5_fold data for reproducibility and development needs in sentiment phrase classification (SPC) on ES issues.
3. `mwe`: This folder contains the dataset for the minimum working example (MWE) of our BK generation, selection, and injection process. 
## Requirements
BacKGen is implemented in `Python 3` (tested on `Python 3.8.12` and `Python 3.10.2`) and requires a number of packages. To install the packages, simply run `$ pip install -r requirements.txt` in your virtual environment. For the `torch` version, you may need a different one following your `cuda` version. For frame clustering, you need `java` installed in your environment, where in our case we use `java 22`.