from tqdm import tqdm
import torch
import random
import numpy as np
import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    AutoTokenizer


class Trainer:
    """
    Class is responsible for training model. It supports only models from pytorch and transformers libraries.
    """

    def __init__(self, model, library: str, num_epochs: int = 10, random_seed: int | None = 0, device: str = 'cpu'):
        """
        :param model: training model
        :param library: parameter that corresponds to which library the model is from: `pytorch` or `transformers`
        :param num_epochs: number of model training epochs
        :param random_seed: parameter responsible for reproducible results
        """
        self.num_epochs = num_epochs
        self.device = device
        if isinstance(model, str):
            self.model = model
        else:
            self.model = model.to(self.device)
        if random_seed is not None:
            self._set_seed(random_seed)
        self.library = library
        self._check_library()

    def _check_library(self):
        if self.library not in ['pytorch', 'transformers']:
            raise NameError(
                f"'{self.library}' is not supported. "
                f"You can use only 'pytorch' and 'transformers' as parameter 'library'")

    @staticmethod
    def _set_seed(random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        transformers.set_seed(random_seed)

    def _train_pytorch(self, optimizer, loss_fn, train_dataloader, val_dataloader=None, use_validation=False, **kwargs):

        for epoch in range(1, self.num_epochs):
            self._train_one_epoch_pytorch(train_dataloader, optimizer, loss_fn, epoch_num=epoch)
            if use_validation:
                self._val_one_epoch_pytorch(val_dataloader, loss_fn, epoch)

    def _train_one_epoch_pytorch(self, dataloader, optimizer, loss_fn, epoch_num):
        loop = tqdm(
            enumerate(dataloader, 1),
            total=len(dataloader),
            desc=f"Epoch {epoch_num}: train",
            leave=True,
        )

        self.model.train()
        train_loss = 0.0
        for i, batch in loop:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            outputs = self.model(X).squeeze(1)
            loss = loss_fn(outputs, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix({"loss": train_loss / (i * len(y))})

    def _val_one_epoch_pytorch(self, dataloader, loss_fn, epoch_num=-1):
        loop = tqdm(
            enumerate(dataloader, 1),
            total=len(dataloader),
            desc=f"Epoch {epoch_num}: val",
            leave=True,
        )
        val_loss = 0.0
        total = 0
        with torch.no_grad():
            self.model.eval()
            for i, batch in loop:
                X, y = batch

                X, y = X.to(self.device), y.to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = loss_fn(outputs, y)

                total += len(y)

                val_loss += loss.item()
                loop.set_postfix({"loss": val_loss / total})

        torch.cuda.empty_cache()
        return val_loss / total

    def _train_transformers(self, tokenized_dataset, learning_rate: float = 2e-5, batch_size: int = 32, **kwargs):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model)
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model_name = self.model.split("/")[-1]
        args = Seq2SeqTrainingArguments(
            f"{model_name}-finetuned",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.num_epochs,
            predict_with_generate=True,
            # fp16=True,
            report_to=['none'],
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=self.model)
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        print('Start training..')
        trainer.train()
        trainer.save_model(f'../models/{model_name}')

    def train(self, num_epochs: int | None = None, **kwargs):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        match self.library:
            case 'pytorch':
                self._train_pytorch(**kwargs)
            case 'transformers':
                self._train_transformers(**kwargs)
