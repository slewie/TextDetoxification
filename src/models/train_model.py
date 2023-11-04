import sys

sys.path.append("./")
import argparse
from src.utils.trainer import Trainer
import src.models.architectures as architectures
from src.data.make_dataloader import make_dataloader_transformers, make_dataloader_pytorch
import torch


def train(config):
    if config.library == 'transformers':
        print('Creating model...')
        trainer = Trainer(config.model_name, 'transformers')
        trainer.train(**vars(config),
                      tokenized_dataset=make_dataloader_transformers(config.data_path, config.model_name))
    elif config.library == 'pytorch':
        match config.model_name:
            case 'toxicity_identifier':
                if config.vocab_size is None:
                    raise AttributeError(f'for the {config.model_name} model vocab_size is required')
                model = architectures.toxicity_classification_model.ToxicityClassificationModel(config.vocab_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                loss_fn = torch.nn.BCELoss()
                trainer = Trainer(model, 'pytorch', device=config.device)
                train_dataloader, val_dataloader = make_dataloader_pytorch(config.data_path, config.model_name)
                trainer.train(20, optimizer=optimizer, loss_fn=loss_fn, train_dataloader=train_dataloader,
                              use_validation=True, val_dataloader=val_dataloader)
            case _:
                print("This model isn't supported")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, help="what is model will be trained. "
                                                       "Now supported only ['toxicity_identifier'] from pytorch and "
                                                       "all models with tokenizer from transformers")
    parser.add_argument('--library', type=str, help='parameter that corresponds to which '
                                                    'library the model is from: `pytorch` or `transformers`')
    parser.add_argument('--num_epochs', default=10, type=int, help='number of model training epochs')
    parser.add_argument('--random_seed', default=0, required=False, type=int,
                        help='parameter responsible for reproducible results')
    parser.add_argument('--device', default='cpu', type=str, required=False, help="'cuda' or 'cpu'")
    parser.add_argument('--learning_rate', default=1e-3, type=float, required=False, help="optimizer learning rate")
    parser.add_argument('--data_path', type=str, help="path to the .csv file with data")
    parser.add_argument('--vocab_size', type=int, required=False, help="vocabulary size for the model")
    config = parser.parse_args()

    train(config)
