import transformers
import numpy as np
import os

from datasets import load_metric

print(transformers.__version__)

from datasets import load_dataset

from transformers import AutoConfig, AutoModelForSequenceClassification

f_result = open("explore_result.txt", encoding="utf-8", mode="w")

seed_count = 50

for the_seed in range(seed_count):
    checkpoint_local = "bert-base-uncased/"
    # 从本地读取config
    config = AutoConfig.from_pretrained(checkpoint_local)
    label2id = {}
    f = open("data/label.data", encoding="utf-8", mode="r")
    for i, line in enumerate(f):
        label2id[line.strip()] = i

    config.num_labels = len(label2id)  # 很重要

    model = AutoModelForSequenceClassification.from_config(config)

    input_data_train = load_dataset("data/explore_data", data_files="train" + str(the_seed) + ".txt")

    input_data_dev = load_dataset("data/explore_data", data_files="test" + str(the_seed) + ".txt")

    from transformers import AutoTokenizer

    if os.path.exists(checkpoint_local + "tokenizer.json"):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=checkpoint_local,
            tokenize_chinese_chars=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            tokenize_chinese_chars=True)
        tokenizer.save_pretrained(checkpoint_local)

    max_len = 24


    def preprocess_function(examples):
        inputs = [one.split(":")[1] for one in examples["text"]]
        targets = [one.split(":")[0] for one in examples["text"]]
        model_inputs = tokenizer(inputs,
                                 max_length=max_len,
                                 padding="max_length",
                                 truncation=True,
                                 add_special_tokens=False,  # 指的是首尾的
                                 return_token_type_ids=False)

        model_inputs["label"] = [label2id[one] for one in targets]  # label2id 是一个dict
        return model_inputs


    tokenized_datasets_train = input_data_train.map(preprocess_function, batched=True, num_proc=4, batch_size=100,
                                                    remove_columns=["text"])
    tokenized_datasets_dev = input_data_dev.map(preprocess_function, batched=True, num_proc=4, batch_size=100,
                                                remove_columns=["text"])


    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=checkpoint_local,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=False,
        num_train_epochs=1
    )

    train_dataset = tokenized_datasets_train["train"]

    dev_dataset = tokenized_datasets_dev["train"]

    num_train_steps = len(train_dataset) * int(training_args.num_train_epochs)
    num_warmup_steps = 0

    from transformers import EvalPrediction, Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=checkpoint_local,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=False,
        num_train_epochs=1
    )

    metric = load_metric("common/my_accuracy.py")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer)

    train_result = trainer.train()

    metrics = trainer.evaluate(eval_dataset=dev_dataset)

    print("index", the_seed)
    print("eval_accuracy", metrics["eval_accuracy"])
    print("eval_loss", metrics["eval_loss"])
    f_result.write(str(the_seed) + "\t" + str(metrics["eval_accuracy"]) + "\t" + str(metrics["eval_loss"]) + "\n")
