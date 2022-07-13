import random
import string
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
import json
import nltk
from utils import trigger_tag


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

    def save(self, filepath):
        tmp = np.stack((self.positive_weights, self.negative_weights))
        np.save(filepath, tmp)

    def load(self, filepath):
        tmp = np.load(filepath)
        self.positive_weights = tmp[0]
        self.negative_weights = tmp[1]


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
        data = json.loads(f.read())
    all_features = []
    all_tags = []

    for sent in data:
        sent_features = []
        text = sent['words']
        tag = trigger_tag(sent)
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


def evaluate(model, features, tags):
    pred = model.pred_by_batch(features)
    pred = np.array(pred)
    tags = np.array(tags)
    TP = np.count_nonzero(pred + tags > 1)
    FP = np.count_nonzero(pred - tags == 1)
    TN = np.count_nonzero(pred + tags == 0)
    FN = np.count_nonzero(pred - tags < 0)
    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP + 1e-4)
    recall = TP / (TP + FN + 1e-4)
    F1 = 2 * precision * recall / (precision + recall + 1e-4)
    return acc, precision, recall, F1, pred.tolist()


def train(model, train_features, train_tags, valid_features, valid_tags, num_epoch):
    acc_log = []
    F1_log = []
    for epoch in tqdm(range(num_epoch)):
        tmp = [(train_features[i], train_tags[i]) for i in range(len(train_features))]
        random.shuffle(tmp)
        train_features = [t[0] for t in tmp]
        train_tags = [t[1] for t in tmp]
        model.update_by_batch(train_features, train_tags)
        acc, p, r, f1, preds = evaluate(model, valid_features, valid_tags)
        acc_log.append(acc)
        F1_log.append(f1)
        print(f"Epoch {epoch}. Eval: acc {acc}, precision {p}, recall {r}, F1 {f1}.")

    '''
    plt.figure()
    plt.plot(acc_log, '-x', label='acc')
    plt.plot(F1_log, '-o', label='F1')
    plt.legend()
    plt.savefig('valid_LinearModel.pdf')
    plt.show()
    '''


def main():
    print('\n\n')
    print('Train Trigger Detection Model')
    # Read data, extract features and vectorize features.
    vectorizer = train_vectorizer()
    train_features, train_tags = form_data('data/train.json')
    train_features = vectorize_data(train_features, vectorizer)
    valid_features, valid_tags = form_data('data/valid.json')
    valid_features = vectorize_data(valid_features, vectorizer)
    test_features, test_tags = form_data('data/test.json')
    test_features = vectorize_data(test_features, vectorizer)

    # Build the model.
    n_feature = train_features[0].shape[1]
    print(f'Feature size: {n_feature}')
    model = LinearModel(n_feature)

    n_epoch = 40

    # Train and test.
    train(model, train_features, train_tags, valid_features, valid_tags, n_epoch)
    model.save('trigger_detection.npy')
    acc, p, r, f1, preds = evaluate(model, test_features, test_tags)
    print(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.")


if __name__ == '__main__':
    main()
