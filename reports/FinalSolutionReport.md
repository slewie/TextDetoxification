# Final Solution Report

## Introduction

The goal of this project was to develop a text detoxification model that can reduce toxicity in text while preserving meaning. Several models and metrics were explored to identify the best approach.

## Data Analysis

### Toxicity of the sentences

Firstly, I checked the distribution of `ref_tox` and `trn_tox`:

![image](./figures/data_analysis/first_hist.png)

We can see that `trn_tox` and `ref_tox` both are in the range from 0 to 1. I transformed the dataset so that one column contains only toxic phrases, and the second contains neutral phrases:

![image](./figures/data_analysis/second_hist.png)

### Toxicity difference

Secondly, I examined the difference between `tox_tox` and `detox_tox`. My idea was that can be case when the difference is low, and it means that quality of detoxification is low, and I removed all cases where df['detox_tox'] > 0.35 or df['tox_diff'] < 0.6

### Similarity and length difference

Finaly, I reviewed the `length_diff` and `similarity`, because I had hypothesis, that can be result, where toxic words was just removed and the meaning of sentence is changed.

![image](./figures/data_analysis/1pairplot.png)

But after exploring data, I concluded that often the pairs with high level of `lenght_diff` just small sentences, where toxic words was removed and meaning doesn't change. I dropped only the values with high level of `lenght_diff`, small level of `similarity` and where `toxic_len` > `detox_len` in order to prevent my hypothesis.


## Model Specification

Four models were evaluated: T5-small, T5-base-detox, Bart-base-detox, and Flan-T5-base. The T5-base-detox model from HuggingFace was selected as it achieved the best performance on validation metrics.

Metrics:

![image](./figures/model_selection/val_loss.png)
Loss on validation set

![image](./figures/model_selection/meteor.png)
Meteor score

![image](./figures/model_selection/sta.png)
Toxicity score

![image](./figures/model_selection/sim.png)
Similarity score

## Training Process

Training is performed on 20000 sample from the dataset on 15 epochs with starting learning_rate = 2e-5.

Training results:
![image](./figures/final_solution/eval_loss.png)
Loss on validation set

![image](./figures/final_solution/train_loss.png)
Loss on train set

![image](./figures/final_solution/meteor.png)
Meteor score

![image](./figures/final_solution/sta.png)
Toxicity score

![image](./figures/final_solution/sim.png)
Similarity score

We can see slight overfitting because the validation loss increases while the training loss decreases. Also the meteor and the similarity become lower.

## Evaluation

Several metrics were used for evaluation:

* Toxicity score (STA): Provided by a RoBERTa toxicity classifier trained on the three datasets by Jigsaw ([Jigsaw 2018](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), [Jigsaw 2019](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), [Jigsaw 2020](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification))
* METEOR: For assessing meaning preservation
* Similarity score (SIM): Using FastText embeddings to compare input and output similarity

I think STA is a key metric because it shows the average toxicity of a sentence. And my model in the detoxified version shows about 0.32.

## Results

You can see the detoxification results by finetuned T5-base-detox model:

![image](./figures/final_solution/predictions.png)

Also, you can test the model using [link](https://huggingface.co/slewie/t5-ultradetox-finetuned)
- Toxicity score 0.32 on average
- METEOR score of 0.61 indicating good meaning retention
- Similarity score of 0.91 showing high semantic similarity

In conclusion, the T5-base-detox model was successful in reducing toxicity while maintaining meaning as measured by both automatic metrics and human evaluation.