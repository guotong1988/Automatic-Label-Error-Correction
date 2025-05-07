# Automatic Label Error Correction Without Human Labor

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

# Doc

https://guotong1988.github.io/core_research/2024/02/01/auto-re-label/

# Run

Step-1, Train the model on origin training dataset, `train.py`

Step-2, Predict the training/dev datasets, `predict.py`

Step-3, Prepare the candidate training datasets, `get_dataset_list.py`

Step-4, Find the best dataset by dev accuracy, `explore_train.py`


# Requirement

transformers            4.38.2 or 4.26.1

torch                   2.2.1 or 1.11.0

scikit-learn            1.3.2

datasets                2.18.0

accelerate              0.27.2

# Experiment Results

![table1](https://guotong1988.github.io/assets/png/auto-relabel/table1.png)

![table1](https://guotong1988.github.io/assets/png/auto-relabel/table2.png)

# Related Work

Label Error Correction With Human Labor: [The Re-Label Method For Data-Centric Machine Learning](https://arxiv.org/abs/2302.04391) 

Controllable Label Error Fixing: [Re-Label By Data Pattern For Controllable Deep Learning](https://www.techrxiv.org/users/679328/articles/679640)

Apply To LLMs: [Simple Self-Eval-Drop for Leveraging LLMs](https://www.techrxiv.org/users/679328/articles/1221820), [Drop Noise For Cleaning LLMs Data](https://www.techrxiv.org/users/679328/articles/1256798)

# More Info

The methods proposed in this project (and its related works) can be applied to all manually annotated (or dataset by LLMs) machine learning / deep learning tasks. 

Not only NLP tasks, but can also be efficiently extended to CV(computer vision) tasks, ASR(speech recognition) tasks, TTS(text-to-speech) tasks, and more.


