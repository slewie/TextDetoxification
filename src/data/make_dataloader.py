import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def make_dataloader_transformers(data_path, model_name, test_size: float = 0.1, max_length: int = 128,
                                 sample_size: int = 50000):

    print('Creating dataloader...')
    df = pd.read_csv(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prefix = "Make this sentence non-toxic: "

    def preprocess_function(examples):
        inputs = [prefix + example for example in examples["toxic"]]
        targets = examples["detoxified"]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs

    dataset = Dataset.from_pandas(df[['toxic', 'detoxified']]).select(range(sample_size))
    train_dataset, validation_dataset = dataset.train_test_split(test_size=test_size).values()

    dd = DatasetDict({"train": train_dataset, "test": validation_dataset})
    tokenized_dataset = dd.map(preprocess_function, batched=True)

    print('Dataloader created')
    return tokenized_dataset
