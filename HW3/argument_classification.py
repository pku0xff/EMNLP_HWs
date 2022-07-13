import json
import numpy as np
from tqdm import tqdm
import spacy
from argument_identification import read_arg_data, build_feature_for_id

role2idx = {'Defendant': 0, 'Instrument': 1, 'Time-At-End': 2, 'Time-After': 3, 'Vehicle': 4, 'Adjudicator': 5,
            'Artifact': 6, 'Price': 7, 'Origin': 8, 'Place': 9, 'Time-Starting': 10, 'Recipient': 11, 'Giver': 12,
            'Position': 13, 'Org': 14, 'Agent': 15, 'Time-Holds': 16, 'Destination': 17, 'Seller': 18, 'Buyer': 19,
            'Person': 20, 'Time-At-Beginning': 21, 'Target': 22, 'Time-Before': 23, 'Attacker': 24, 'Beneficiary': 25,
            'Crime': 26, 'Time-Within': 27, 'Time-Ending': 28, 'Prosecutor': 29, 'Sentence': 30, 'Money': 31,
            'Entity': 32, 'Victim': 33, 'Plaintiff': 34}

idx2role = {i: s for s, i in role2idx.items()}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction import DictVectorizer
from argument_identification import get_arg_feature


class LinearModel:
    def __init__(self,
                 train_data,
                 train_labels,
                 valid_data,
                 valid_labels,
                 test_data,
                 test_labels,
                 n_features,
                 n_classes):
        self.train_data = train_data  # array (N, n_features)
        self.train_labels = train_labels  # List
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.n_features = n_features
        self.n_classes = n_classes
        self.weight = np.random.rand(n_classes, n_features) / 10  # (n_features, n_labels)

    def predict(self, x):
        score = (self.weight @ x.reshape(-1, 1)).reshape(-1)
        return int(np.argmax(score))

    def train(self, n_epoch):
        for epoch in tqdm(range(n_epoch)):
            for i in range(len(self.train_labels)):
                feature = self.train_data[i]
                gt = self.train_labels[i]
                pred = self.predict(feature)
                if gt != pred:
                    self.weight[gt] += feature
                    self.weight[pred] -= feature
            acc, p, r, f1 = self.evaluate()
            print(f"Epoch {epoch}. Eval: acc {acc}, precision {p}, recall {r}, F1 {f1}.")

    def evaluate(self):
        preds = []
        for i in range(len(self.valid_labels)):
            feature = self.valid_data[i]
            pred = self.predict(feature)
            preds.append(int(pred))
        acc = accuracy_score(self.valid_labels, preds)
        p = precision_score(self.valid_labels, preds, average='macro')
        r = recall_score(self.valid_labels, preds, average='macro')
        f1 = f1_score(self.valid_labels, preds, average='macro')
        return acc, p, r, f1

    def test(self):
        preds = []
        for i in range(len(self.test_labels)):
            feature = self.test_data[i]
            pred = self.predict(feature)
            preds.append(int(pred))
        acc = accuracy_score(self.test_labels, preds)
        p = precision_score(self.test_labels, preds, average='macro')
        r = recall_score(self.test_labels, preds, average='macro')
        f1 = f1_score(self.test_labels, preds, average='macro')
        return acc, p, r, f1

    def save(self, filepath):
        np.save(filepath, self.weight)

    def load(self, filepath):
        self.weight = np.load(filepath)


# turn raw data into features and labels
def form_arg_data(data):
    features = []
    labels = []
    for sent in data:
        text = sent['words']
        for event in sent['events']:
            trigger = event['trigger']
            args = event['arguments']
            for arg in args:
                feature = get_arg_feature(text, trigger, arg)
                label = role2idx[arg['role']]
                features.append(feature)
                labels.append(label)
    return features, labels


def main():
    print('\n\n')
    print('Train Argument Classification Model')
    with open('data/train.json') as f:
        train_data = json.loads(f.read())
    with open('data/valid.json') as f:
        valid_data = json.loads(f.read())
    with open('data/test.json') as f:
        test_data = json.loads(f.read())

    nlp = spacy.load('en_core_web_sm')
    train_instances = read_arg_data(nlp, 'data/train.json')
    train_features, _ = build_feature_for_id(train_instances)
    vectorizer = DictVectorizer()
    vectorizer.fit(train_features)

    train_features, train_labels = form_arg_data(train_data)
    valid_features, valid_labels = form_arg_data(valid_data)
    test_features, test_labels = form_arg_data(test_data)

    train_features = [vectorizer.transform(i) for i in train_features]
    valid_features = [vectorizer.transform(i) for i in valid_features]
    test_features = [vectorizer.transform(i) for i in test_features]
    n_features = test_features[0].shape[1]

    model = LinearModel(train_features, train_labels, valid_features, valid_labels, test_features, test_labels,
                        n_features, len(role2idx))

    model.train(8)
    acc, p, r, f1 = model.test()
    model.save('argument_classification.npy')
    print(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.")


if __name__ == '__main__':
    main()
