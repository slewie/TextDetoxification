# TextDetoxification Assignment
Vladislav Kulikov v.kulikov@innopolis.university BS21-DS-02


## Installation

1. Download repository
```
git clone https://github.com/slewie/TextDetoxification
```

2. Install libraries
```
pip install -r requirements.txt
```

3. Download and prepare dataset:

```
python ./src/data/make_dataset.py
```

## Train model

```
python src/models/train_model.py
```

with such parameters:

* --model_name | what is model will be trained. Now supported only ['toxicity_identifier'] from pytorch and all models with tokenizer from transformers

* --library | parameter that corresponds to which library the model is from: `pytorch` or `transformers`

* --num_epochs | number of model training epochs
* --random_seed | parameter responsible for reproducible results
* --device |  `cuda` or `cpu`
* --learning_rate | optimizer learning rate
* --data_path | path to the .csv file with data
* --vocab_size | vocabulary size for the model
* --save_model | save model or not


## Predict model

```
python src/models/predict_model.py
```

with such parameters:

* --model_name | what is model will be trained.
* --library | parameter that corresponds to which library the model is from: `pytorch` or `transformers`

After running the command, you need to write a text sequence that needs to be predicted or generated.