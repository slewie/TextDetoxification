import fasttext
import pandas as pd


def make_embeddings(data_embeddings_path='../data/interim/toxicity_levels.csv',
                    embedding_path="../models/fasttext.bin"):
    df_train = pd.read_csv(data_embeddings_path)
    df_train['text'].to_csv('data.txt', index=False)
    print('Creating embeddings...')
    model = fasttext.train_unsupervised('data.txt')
    print('Embeddings created')
    model.save_model(embedding_path)