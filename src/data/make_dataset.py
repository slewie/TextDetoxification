import argparse
import os
import requests
import zipfile
import io
import pandas as pd


def download_read_data():
    """
    Function downloads zipfile with raw dataset and extract it
    """
    print("Downloading file...")
    result = requests.get(
        'https://raw.githubusercontent.com/slewie/TextDetoxification/main/data/raw/filtered_paranmt.zip')
    zip_file = zipfile.ZipFile(io.BytesIO(result.content))
    print('Extracting zipfile...')
    zip_file.extractall('./data/raw')


def prepare_data():
    """
    Function prepares dataset for future work: divides all phrases on toxic and normal version and drops all numeric columns
    """
    data_path = "./data/raw/filtered.tsv"
    df = pd.read_csv(data_path, sep='\t', index_col=0)
    print('Preparing .csv file...')
    df['toxic'] = df.apply(lambda x: x['reference'] if x['ref_tox'] > x['trn_tox'] else x['translation'], axis=1)
    df['detoxified'] = df.apply(lambda x: x["translation"] if x['ref_tox'] > x['trn_tox'] else x['reference'], axis=1)
    df['tox_tox'] = df.apply(lambda x: x["ref_tox"] if x['ref_tox'] > x['trn_tox'] else x['trn_tox'], axis=1)
    df['detox_tox'] = df.apply(lambda x: x["trn_tox"] if x['ref_tox'] > x['trn_tox'] else x['ref_tox'], axis=1)
    df['tox_diff'] = df['tox_tox'] - df['detox_tox']
    df.drop(columns=['reference', 'translation', 'trn_tox', 'ref_tox'], inplace=True)
    df = df[(df['detox_tox'] < 0.35) & (df['tox_diff'] > 0.6)]
    detox_df = df[['toxic', 'detoxified']]
    print('Converting it to .csv...')
    detox_df.to_csv('./data/interim/tox_sentences.csv', index=None)
    print('File was downloaded')


if __name__ == "__main__":
    download_read_data()
    prepare_data()
