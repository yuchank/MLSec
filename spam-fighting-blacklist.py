import os
import pickle
import email_read_util

DATA_DIR = './datasets/trec07p/data/'
LABELS_FILE = './datasets/trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}
spam_words = set()
ham_words = set()

# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# Split corpus into train and test sets
filelist = os.listdir(DATA_DIR)
