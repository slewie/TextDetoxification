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

    # convert the original sentences into toxic and detoxified based on toxicity levels
    df['toxic'] = df.apply(lambda x: x['reference'] if x['ref_tox'] > x['trn_tox'] else x['translation'], axis=1)
    df['detoxified'] = df.apply(lambda x: x["translation"] if x['ref_tox'] > x['trn_tox'] else x['reference'], axis=1)
    df['tox_tox'] = df.apply(lambda x: x["ref_tox"] if x['ref_tox'] > x['trn_tox'] else x['trn_tox'], axis=1)
    df['detox_tox'] = df.apply(lambda x: x["trn_tox"] if x['ref_tox'] > x['trn_tox'] else x['ref_tox'], axis=1)
    df['tox_diff'] = df['tox_tox'] - df['detox_tox']
    df.drop(columns=['reference', 'translation', 'trn_tox', 'ref_tox'], inplace=True)

    # remove columns where detoxified sentence has high level of toxicity and
    # columns where toxic sentence has smale level of toxicity
    df = df[(df['detox_tox'] < 0.35) & (df['tox_diff'] > 0.6)]
    # remove samples where the toxic sentence and detoxified sentence are different (in sense of phrase meaning)
    df = df.query('similarity >= 0.7')
    # remove cases where the model can simply throw out toxic words from a sentence and lose the meaning of the phrase
    length_df = df.copy()
    length_df['toxic_len'] = df['toxic'].map(len)
    length_df['detox_len'] = df['detoxified'].map(len)
    df = df.drop(length_df.query('lenght_diff > 0.35 & toxic_len > detox_len & similarity < 0.77').index)

    detox_df = df[['toxic', 'detoxified']]
    print('Converting it to .csv...')
    detox_df.to_csv('./data/interim/tox_sentences.csv', index=None)
    print('File was downloaded')


if __name__ == "__main__":
    download_read_data()
    prepare_data()
