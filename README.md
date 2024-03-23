# Automatic Label Error Correction Without Human Labor
https://guotong1988.github.io/core_research/2024/02/01/auto-re-label/

# Run

Step-1, Train the model, `train.py`

Step-2, Predict the train/dev datasets, `predict.py`

Step-3, Prepare the candidate datasets, `get_dataset_list.py`

Step-4, Find the best dataset, `explore_train.py`


# Requirement

transformers            4.38.2 or 4.26.1

torch                   2.2.1 or 1.11.0

scikit-learn            1.3.2

datasets                2.18.0

accelerate              0.27.2

