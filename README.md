# Automatic Label Error Correction Without Human Labor

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

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
