import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils


def _make_dataloader_transformers(data_path, model_name, test_size: float = 0.1, max_length: int = 128,
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


def _make_dataloader_pytorch(data_path, model_name, test_size: float = 0.1, batch_size: int = 512,
                            random_seed: int | None = 0, device: str = 'cpu'):
    print('Creating dataloader...')
    df = pd.read_csv(data_path)
    threshold = 0.5

    df['tox_level'] = df['tox_level'].apply(lambda x: 1 if x > threshold else 0)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')  # TODO: read the tokenizer name from console

    def preprocessing_stage(sample):
        # in the preprocessing phase, I convert the input text to the list of tokens
        model_inputs = tokenizer(sample['text'], padding='max_length', max_length=256, truncation=True)
        return model_inputs['input_ids']

    df['input_ids'] = df.apply(lambda x: preprocessing_stage(x), axis=1)
    df.drop(columns=['text'], inplace=True)
    train, val = train_test_split(
        df, stratify=df['tox_level'], test_size=test_size, random_state=random_seed
    )

    def collate_batch(batch):
        text_list, toxicity_list = [], []
        for _toxicity, _text in batch:
            text_list.append(_text)
            toxicity_list.append(_toxicity)
        return torch.LongTensor(text_list).to(device), torch.FloatTensor(toxicity_list).to(device)

    train_dataloader = data_utils.DataLoader(
        train.to_numpy(), batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    val_dataloader = data_utils.DataLoader(
        val.to_numpy(), batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    print('Dataloader created')
    return train_dataloader, val_dataloader
