import requests
from urllib.parse import urlencode
import os


def make_embeddings(embedding_path="fasttext.bin"):
    print('Downloading embeddings...')
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/4KOFyzhKjvYUtg'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_response = requests.get(download_url, stream=True)
    with open(os.path.join(os.path.dirname(__file__), '../../models/') + embedding_path, 'wb') as f:
        for chunk in download_response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print('Embeddings created')
