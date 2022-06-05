import string
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
import json
import nltk
import matplotlib.pyplot as plt
import argparse


# Build a Linear model and train it with the Perceptron Algorithm
class LinearModel():
    def __init__(self, feature_size):
        self.feature_size = feature_size
        self.positive_weights = np.zeros(feature_size)
        self.negative_weights = np.zeros(feature_size)

    def update_by_word(self, feature, tag):
        feature = np.asarray(feature.todense()).reshape(-1)
        positive_score = np.sum(self.positive_weights * feature)
        negative_score = np.sum(self.negative_weights * feature)
        if tag == 1 and positive_score <= negative_score:
            self.positive_weights += feature
            self.negative_weights -= feature
        if tag == 0 and negative_score <= positive_score:
            self.positive_weights -= feature
            self.negative_weights += feature

    def update_by_batch(self, features, tags):
        for f, t in zip(features, tags):
            self.update_by_word(f, t)

    def pred_by_word(self, feature):
        feature = np.asarray(feature.todense()).reshape(-1)
        positive_score = np.sum(self.positive_weights * feature)
        negative_score = np.sum(self.negative_weights * feature)
        return 1 if positive_score > negative_score else 0

    def pred_by_batch(self, features):
        return [self.pred_by_word(f) for f in features]


# Given the sentence and a token, form a feature dictionary for the token.
def get_feature(token, token_idx, sent, sent_pos):
    last_token = '' if token_idx == 0 else sent[token_idx - 1]
    next_token = '' if token_idx == len(sent) - 1 else sent[token_idx + 1]
    punctuation = set(string.punctuation)
    token_feature = {
        'token': token,
        # 'last_token': last_token,
        # 'next_token': next_token,
        'position': token_idx,
        'is_first': token_idx == 0,
        'is_last': token_idx == len(sent) - 1,

        'is_capitalized': token[0].upper() == token[0],
        '-1is_capitalized': 0 if token_idx == 0 else last_token[0].upper() == last_token,
        '+1is_capitalized': 0 if token_idx == len(sent) - 1 else next_token[0].upper() == next_token[0],

        'is_numeric': token.isdigit(),
        '-1is_numeric': last_token.isdigit(),
        '+1is_numeric': next_token.isdigit(),

        'is_punctuation': token in punctuation,
        '+1is_punctuation': last_token and last_token in punctuation,
        '-1is_punctuation': next_token and next_token in punctuation,

        'pos': sent_pos[token_idx],
        '+1pos': '' if token_idx == 0 else sent_pos[token_idx - 1],
        '-1pos': '' if token_idx == len(sent) - 1 else sent_pos[token_idx + 1],
    }
    return token_feature


# Turn texts into feature dictionaries and flatten the sentences.
def form_data(filepath):
    with open(filepath, encoding='utf8') as f:
        data = json.load(f)
    all_features = []
    all_tags = []
    for sent in data:
        sent_features = []
        text = sent['words']
        tag = sent['triggers']
        sent_pos = nltk.pos_tag(text)
        sent_pos = [p[1] for p in sent_pos]
        for token_idx, token in enumerate(text):
            sent_features.append(get_feature(token, token_idx, text, sent_pos))
        all_features += sent_features
        all_tags += tag

    return all_features, all_tags


# Turn feature dictionaries into vectors.
def train_vectorizer():
    train_features, train_tags = form_data('data/train.json')
    vectorizer = DictVectorizer()
    vectorizer.fit(train_features)
    return vectorizer


def vectorize_data(features, vectorizer):
    vec_features = []
    for feature in features:
        vec_features.append(vectorizer.transform(feature))
    return vec_features


def evaluate(model, features, tags, test=False):
    pred = model.pred_by_batch(features)
    pred = np.array(pred)
    tags = np.array(tags)
    if test:
        FP_idx = np.argwhere(pred - tags == 1).reshape(-1)
        FN_idx = np.argwhere(pred - tags < 0).reshape(-1)
        np.save("linear_fp.npy", FP_idx)
        np.save("linear_fn.npy", FN_idx)
    TP = np.count_nonzero(pred + tags > 1)
    FP = np.count_nonzero(pred - tags == 1)
    TN = np.count_nonzero(pred + tags == 0)
    FN = np.count_nonzero(pred - tags < 0)
    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP + 1e-4)
    recall = TP / (TP + FN + 1e-4)
    F1 = 2 * precision * recall / (precision + recall + 1e-4)
    return acc, precision, recall, F1


def train(model, train_features, train_tags, eval_features, eval_tags, num_epoch, f):
    acc_log = []
    F1_log = []
    for epoch in tqdm(range(num_epoch)):
        model.update_by_batch(train_features, train_tags)
        acc, p, r, f1 = evaluate(model, eval_features, eval_tags)
        acc_log.append(acc)
        F1_log.append(f1)
        print(f"Epoch {epoch}. Eval: acc {acc}, precision {p}, recall {r}, F1 {f1}.")
        f.write(f"Epoch {epoch}. Eval: acc {acc}, precision {p}, recall {r}, F1 {f1}.\n")

    plt.figure()
    plt.plot(acc_log, '-x', label='acc')
    plt.plot(F1_log, '-o', label='F1')
    plt.legend()
    plt.savefig('eval_LinearModel.pdf')
    plt.show()


# Read data, extract features and vectorize features.
vectorizer = train_vectorizer()
train_features, train_tags = form_data('data/train.json')
train_features = vectorize_data(train_features, vectorizer)
eval_features, eval_tags = form_data('data/valid.json')
eval_features = vectorize_data(eval_features, vectorizer)
test_features, test_tags = form_data('data/test.json')
test_features = vectorize_data(test_features, vectorizer)

# Build the model.
n_feature = train_features[0].shape[1]
model = LinearModel(n_feature)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', default=40)
args = parser.parse_args()
n_epoch = int(args.n_epoch)

# Train and test.
f = open('LinearModel_log.txt', 'w', encoding='utf8')
train(model, train_features, train_tags, eval_features, eval_tags, n_epoch, f)
acc, p, r, f1 = evaluate(model, test_features, test_tags, test=True)
print(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.")
f.write(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.\n")
f.close()
