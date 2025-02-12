from pathlib import Path
import tarfile
import urllib.request
import pandas as pd

def load_data():
    rel_path = Path('datasets/housing.tgz')
    if(not rel_path.is_file()):
        Path("datasets").mkdir(parents=True,exist_ok=True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url,rel_path)
        with tarfile.open(rel_path) as zip:
            zip.extractall(path='data')
    return pd.read_csv('datasets/housing/housing.csv')        

