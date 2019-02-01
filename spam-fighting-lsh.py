import os
import pickle
import email_read_util
from datasketch import MinHash, MinHashLSH
from nltk.corpus import words
from IPython.display import HTML, display

DATA_DIR = './datasets/trec07p/data/'
LABELS_FILE = './datasets/trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}  # dict

# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# Split corpus into training and test sets
file_list = os.listdir(DATA_DIR)
X_train = file_list[:int(len(file_list)*TRAINING_SET_RATIO)]  # 52793 out of 75419
X_test = file_list[int(len(file_list)*TRAINING_SET_RATIO):]   # 22626 out of 75419

# Extract only spam files for inserting into the LSH matcher
spam_files = [x for x in X_train if labels[x] == 0]

fp = 0
tp = 0
fn = 0
tn = 0

for filename in X_test:
    path = os.path.join(DATA_DIR, filename)
    if filename in labels:
        label = labels[filename]
        stems = email_read_util.load(path)
        if not stems:
            continue
        stems_set = set(stems)
        if stems_set & blacklist:   # it's positive(spam)
            if label == 1:
                fp = fp + 1         # it's false(ham)
            else:
                tp = tp + 1         # it's true(spam)
        else:                       # it's negative(ham)
            if label == 1:
                tn = tn + 1         # it's true(ham)
            else:
                fn = fn + 1         # it's false(spam)

conf_matrix = [[tn, fp],
               [fn, tp]]
display(HTML('<table><tr>{}</tr></table>'.format(
    '</tr><tr>'.join('<td>{}</td>'.format(
        '<td></td>'.join(str(_) for _ in row)) for row in conf_matrix))))

count = tn + tp + fn + fp
percent_matrix = [["{:.1%}".format(tn/count), "{:.1%}".format(fp/count)],
                  ["{:.1%}".format(fn/count), "{:.1%}".format(tp/count)]]
display(HTML('<table><tr>{}</tr></table>'.format(
    '</tr><tr>'.join('<td>{}</td>'.format(
        '</td><td>'.join(str(_) for _ in row)) for row in percent_matrix))))

print("Classification accuracy: {}".format("{:.1%}".format((tp + tn)/count)))
