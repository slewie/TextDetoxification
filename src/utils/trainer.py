from tqdm import tqdm
import torch
import random
import numpy as np
import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    AutoTokenizer
from src.evaluation.evaluator import Evaluator
from src.utils.make_embeddings import make_embeddings
import os


class Trainer:
    """
    Class is responsible for training model. It supports only models from pytorch and transformers libraries.
    """

    def __init__(self, model, library: str, num_epochs: int = 10, random_seed: int | None = 0, device: str = 'cpu',
                 sim_model_path: str = '../models/fasttext.bin'):
        """
        :param model: training model
        :param library: parameter that corresponds to which library the model is from: `pytorch` or `transformers`
        :param num_epochs: number of model training epochs
        :param random_seed: parameter responsible for reproducible results
        :param device: on which device train model
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
        self.sim_model_path = sim_model_path

    def _check_library(self):
        """
        The function checks whether the 'library' parameter is supported or not.
        """
        if self.library not in ['pytorch', 'transformers']:
            raise NameError(
                f"'{self.library}' is not supported. "
                f"You can use only 'pytorch' and 'transformers' as parameter 'library'")

    def _set_seed(self, random_seed):
        """
        Function fixes the random seed in all libraries
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(random_seed)
        transformers.set_seed(random_seed)

    def _train_pytorch(self, optimizer, loss_fn, train_dataloader, val_dataloader=None, use_validation=False,
                       save_model=False, **kwargs):
        """
        The function runs the training process for pytorch model
        :param optimizer: model optimizer. Adam by default
        :param loss_fn: loss function. Binary Cross Entropy by default
        :param train_dataloader: dataloader with train data
        :param val_dataloader: dataloader with validation data
        :param use_validation: whether to use verification or not
        :param save_model: whether to save model or not. Model is saved into ../models directory with class name
        """
        print('Start training..')
        for epoch in range(1, self.num_epochs):
            self._train_one_epoch_pytorch(train_dataloader, optimizer, loss_fn, epoch_num=epoch)
            if use_validation:
                self._val_one_epoch_pytorch(val_dataloader, loss_fn, epoch)
        print('Training is finished')
        if save_model:
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(f'../models/{str(self.model).split("(")[0]}.pt')

    def _train_one_epoch_pytorch(self, dataloader, optimizer, loss_fn, epoch_num):
        """
        Function for training one epoch. Also, prints training information using tqdm module
        :param dataloader: dataloader with train data
        :param optimizer: model optimizer
        :param loss_fn: loss function
        :param epoch_num: shows at what epoch the training was used
        """
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
        """
        Function for validating one epoch. Also, prints training information using tqdm module
        :param dataloader: dataloader with train data
        :param loss_fn: loss function
        :param epoch_num: shows at what epoch the training was used
        """
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

    def _train_transformers(self, tokenized_dataset, learning_rate: float = 2e-5, batch_size: int = 32,
                            save_model: bool = False, model_path='../models/',
                            data_embeddings_path='../data/interim/toxicity_levels.csv', **kwargs):
        """
        The function runs the training process for transformers model. Function uses trainers from transformers library
        :param tokenized_dataset: dataset converted to the tokens for language model
        :param learning_rate: starting learning rate
        :param batch_size: batch size for the training
        :param save_model: whether to save model or not. Model is saved into ../models directory with model name
        """
        if not os.path.exists(self.sim_model_path):
            make_embeddings(embedding_path=self.sim_model_path, data_embeddings_path=data_embeddings_path)

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model)
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model_name = self.model.split("/")[-1]
        evaluator = Evaluator(tokenizer, sim_model_path=self.sim_model_path)
        args = Seq2SeqTrainingArguments(
            f"./{model_name}-finetuned",
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
            tokenizer=tokenizer,
            compute_metrics=evaluator.compute_metric
        )
        print('Start training..')
        trainer.train()
        print('Training is finished')
        if save_model:
            trainer.save_model(f'{model_path}{model_name}-finetuned')

    def train(self, num_epochs: int | None = None, **kwargs):
        """
        Runs training procedure based on model library
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs
        match self.library:
            case 'pytorch':
                self._train_pytorch(**kwargs)
            case 'transformers':
                self._train_transformers(**kwargs)
