import pandas as pd
import urllib
import os
from tqdm import tqdm

DATA_DIR = "../data/"
URL_DIR = "../data/3massiv_urls"
PRETRN_LABEL_FILE = "pretrain_labels.csv"
PRETRN_DATA_DIR = "../data/pretrain"
pretrn_labels = pd.read_csv(os.path.join(DATA_DIR, PRETRN_LABEL_FILE))

URL_FILE_TRN = "3massiv_train_urls.csv"
URL_FILE_VAL = "3massiv_val_urls.csv"
url_file_trn = pd.read_csv(os.path.join(URL_DIR, URL_FILE_TRN))
url_file_val = pd.read_csv(os.path.join(URL_DIR, URL_FILE_VAL))

urls = pd.concat([url_file_trn, url_file_val])

pretrn_dloaded, pretrn_missed = 0, 0

# Download pretrain train data handling errors
for post_index in tqdm(pretrn_labels["PostIndex"]):
    url = urls[urls["PostIndex"]==post_index]["PostURL"].values[0]
    try:
        urllib.request.urlretrieve(url, os.path.join(PRETRN_DATA_DIR,
                                                     str(post_index)+".mp4"))
        pretrn_dloaded += 1
    except:
        print("Error downloading", post_index)
        pretrn_missed += 1

print("Pretrain downloaded:{}/{}".format(pretrn_dloaded,
                                         len(pretrn_labels)))
print("Pretrain missed:{}/{}".format(pretrn_missed,
                                    len(pretrn_labels)))

