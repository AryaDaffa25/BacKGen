# About BacKGen Dataset

This `data` folder contains four subfolders that can be used as follows:
* `original_annotation`: This folder contains the original annotation result where each tweet is annotated by three different annotators. You can use a particular aggregation method (e.g. follow [Rodrigues et.al. (2014)](https://link.springer.com/article/10.1007/s10994-013-5411-2) as we did) or use your own aggregation method. You can also take the benefit of the disagreement as we provide all annotation results.
* `5_fold_frame_spc`: This folder contains the 5-fold dataset of aggregated sentiment phrase (named as `*spc.jsonl`) that used for the main sentiment phrase classification (SPC) task and generating background knowledge (BK) and frame extracted from the instance (named as `*frame.jsonl`) that used for generating BK.
* `5_fold_bk`: This folder contains the 5-fold generated BK for the BK-injection to LLMs purpose.
* `mwe`: This folder contains the minimum working example output for the end-to-end process of the SPC with BK-injection.
