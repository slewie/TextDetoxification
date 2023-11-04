import sys

sys.path.append("./")
import argparse
from src.utils.predictor import Predictor


def predict(config, request):
    if config.library == 'transformers':
        print('Creating model...')
        trainer = Predictor(config.model_name, 'transformers')
        print(trainer.predict(request))
    elif config.library == 'pytorch':
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, help="what is model will be trained.")
    parser.add_argument('--library', type=str, help='parameter that corresponds to which '
                                                    'library the model is from: `pytorch` or `transformers`')
    config = parser.parse_args()
    print('Write a text sequence to make it non-toxic')
    request = input()

    predict(config, request)
