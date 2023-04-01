import pandas as pd
import urllib
import os
from tqdm import tqdm

DATA_DIR = "../data/"
URL_DIR = "../data/3massiv_urls"
FEWSHOT_TRN_LABEL_FILE = "fewshot_trn_labels.csv"
FEWSHOT_VAL_LABEL_FILE = "fewshot_val_labels.csv"
URL_FILE_TRN = "3massiv_train_urls.csv"
URL_FILE_VAL = "3massiv_val_urls.csv"

FEWSHOT_TRN_DATA_DIR = "../data/fewshot_trn"
FEWSHOT_VAL_DATA_DIR = "../data/fewshot_val"

fewshot_trn_labels = pd.read_csv(os.path.join(DATA_DIR, FEWSHOT_TRN_LABEL_FILE))
fewshot_val_labels = pd.read_csv(os.path.join(DATA_DIR, FEWSHOT_VAL_LABEL_FILE))
url_file_trn = pd.read_csv(os.path.join(URL_DIR, URL_FILE_TRN))
url_file_val = pd.read_csv(os.path.join(URL_DIR, URL_FILE_VAL))

urls = pd.concat([url_file_trn, url_file_val])

trn_dloaded, trn_missed = 0, 0

# Download fewshot train data handling errors
for post_index in fewshot_trn_labels["PostIndex"]:
    url = urls[urls["PostIndex"]==post_index]["PostURL"].values[0]
    try:
        urllib.request.urlretrieve(url, os.path.join(FEWSHOT_TRN_DATA_DIR,
                                                     str(post_index)+".mp4"))
        trn_dloaded += 1
    except:
        print("Error downloading", post_index)
        trn_missed += 1

print("Fewshot train downloaded:{}/{}".format(trn_dloaded,
                                              len(fewshot_trn_labels)))
print("Fewshot train missed:{}/{}".format(trn_missed,
                                          len(fewshot_trn_labels)))

val_dloaded, val_missed = 0, 0

# Download fewshot val data handling errors
for post_index in fewshot_val_labels["PostIndex"]:
    url = urls[urls["PostIndex"]==post_index]["PostURL"].values[0]
    try:
        urllib.request.urlretrieve(url, os.path.join(FEWSHOT_VAL_DATA_DIR,
                                                     str(post_index)+".mp4"))
        val_dloaded += 1
    except:
        print("Error downloading", post_index)
        val_missed += 1

print("Fewshot val downloaded:{}/{}".format(val_dloaded,
                                             len(fewshot_val_labels)))
print("Fewshot val missed:{}/{}".format(val_missed,
                                        len(fewshot_val_labels)))

