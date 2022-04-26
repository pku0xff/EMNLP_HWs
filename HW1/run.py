import time
import argparse
from collections import defaultdict
from tqdm.std import trange
import numpy as np
from preprocess import load_data, unique

'''
reference: 
Zhang and Oles (2010), Text Categorization Based on Regularized Linear Classification Methods
Yang Y and Pedersen J (1997), A comparative study on feature selection in text categorization
'''


def cal_IG(term, texts, labels, constant):
    '''
    :param term: the term to calculate IG
    :param texts: a list of lists
    :param labels: a list of labels(int)
    :param constant: Sigma(P(c)*logP(c))
    :return: information gain of term
            G(t) = P(t)*Sigma(P(c|t)*logP(c|t)) + P(^t)*Sigma(P(c|^t)*logP(c|^t)) - constant
    '''
    term_cnt = 0
    cond_cnt = defaultdict(int)
    cond_cnt_wo_t = defaultdict(int)
    N = len(texts)

    # Count the numbers first
    for i in range(N):
        if term in texts[i]:
            term_cnt += 1
            cond_cnt[labels[i]] += 1
        else:
            cond_cnt_wo_t[labels[i]] += 1

    cond_prob = np.array(list(cond_cnt.values())) / term_cnt
    sum1 = np.sum(cond_prob * np.log(cond_prob))
    cond_prob_wo_t = np.array(list(cond_cnt_wo_t.values())) / (N - term_cnt)
    sum2 = np.sum(cond_prob_wo_t * np.log(cond_prob_wo_t))
    IG = sum1 * (term_cnt / N) + sum2 * (1 - term_cnt / N) - constant
    return IG


def feature_extraction(texts, labels, classes, n_features, method):
    vocab = set()
    for sent in texts:
        for word in sent:
            vocab.add(word)
    print(f'Length of original vocab: {len(vocab)}')  # sst 10955, yelp 567558

    vocab = list(vocab)

    def filter_by_DF():
        all_words = []
        for sent in texts:
            all_words += sent
        cnt_dict = defaultdict(int)
        for w in all_words:
            cnt_dict[w] += 1
        new_vocab = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=True)
        new_vocab = [i[0] for i in new_vocab[0:n_features]]
        return new_vocab

    def filter_by_IG():
        ig = {}
        cato_dict = {}
        N = len(texts)
        for cato in classes:
            cato_dict[cato] = labels.count(cato) / N
        tmp = np.array(list(cato_dict.values()))
        constant = np.sum(tmp * np.log(tmp))
        for term in vocab:
            ig[term] = cal_IG(term, texts, labels, constant)
        ig = sorted(ig.items(), key=lambda x: x[1], reverse=True)
        features = [p[0] for p in ig[0:n_features]]
        return features

    if method == 'DF':
        vocab = filter_by_DF()
        print(f'Length of vocab after document frequency filtering: {len(vocab)}')
    else:
        vocab = filter_by_IG()
        print(f'Length of vocab after IG filtering: {len(vocab)}')
    return vocab


def text2vec(texts, vocab):
    vec = np.zeros((len(texts), len(vocab)))
    for i in range(len(texts)):
        words = set(texts[i])
        for j in range(len(vocab)):
            if vocab[j] in words:
                vec[i][j] = 1
    return vec


# I try to use numpy instead of for/while loop in my model to improve efficiency.
class LLModel:
    def __init__(self,
                 dataset,
                 train_data,
                 train_labels,
                 valid_data,
                 valid_labels,
                 test_data,
                 test_labels,
                 vocab,
                 classes,
                 alpha,
                 save_path):
        self.dataset = dataset
        self.train_data = train_data  # (N, n_features)
        self.train_labels = train_labels
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.vocab = vocab
        self.classes = classes
        self.weight = np.random.rand(self.train_data.shape[1], len(classes)) / 10  # (n_features, n_labels)
        self.alpha = alpha
        self.save_path = save_path

    def score(self, x):
        '''
        :param x: f(x,y), ndarray in shape (n_features, )
        :return: scores, ndarray in shape (n_labels, )
        '''
        return np.sum(x.reshape(1, -1) @ self.weight, axis=0)

    def prob(self, x, y):
        s = np.exp(self.score(x))
        return s[y] / np.sum(s)

    def loss(self):
        loss = 0
        for i in range(self.train_data.shape[0]):
            x = train_data[i]
            loss += x.dot(self.weight[:, self.train_labels[i]].reshape(-1))
            loss -= np.log(np.sum(np.exp(self.score(x))))
        loss -= self.alpha / 2 * (np.sum(self.weight))
        loss = -loss
        return loss

    def grad(self, batch_data, batch_label):
        grad = np.zeros(self.weight.shape)  # (n_features, n_labels)
        for i in range(batch_data.shape[0]):
            grad[:, batch_label[i]] += batch_data[i]
            score = np.exp(self.score(batch_data[i]))
            grad[np.where(batch_data[i] > 0), :] -= score / np.sum(score)
        grad -= self.alpha * self.weight
        return grad

    def train(self, batch_size, lr, epoch):
        f = open(self.save_path, 'a', encoding='utf-8')
        steps = int(np.ceil(self.train_data.shape[0] / batch_size))
        for i in range(epoch):
            self.train_data = np.random.permutation(self.train_data)
            for s in trange(steps):
                batch_data = train_data[s * batch_size:(s + 1) * batch_size]
                batch_label = train_labels[s * batch_size:(s + 1) * batch_size]
                grad = self.grad(batch_data, batch_label)
                self.weight += lr * grad
                if s % 100 == 0 and s > 0:
                    print(f'step {s}, loss {self.loss()}, valid acc {self.validate()}')
                    f.write(f'step {s}, loss {self.loss()}\n')
                # lr = lr * (0.9 ** ((i * steps) / 1))
            print(f'epoch {i} finished with loss {self.loss()}, valid acc {self.validate()}')
            f.write(f'epoch {i} finished with loss {self.loss()}, valid acc {self.validate()}\n')
        f.write('\n')
        f.close()

    def validate(self):
        N = len(valid_labels)
        corr = 0
        for i in range(N):
            score = self.score(self.valid_data[i])
            pred = np.argmax(score)
            if pred == self.valid_labels[i]:
                corr += 1
        return corr / N

    def test(self):
        N = len(test_labels)
        pred = []
        for i in range(N):
            score = self.score(self.test_data[i])
            pred.append(np.argmax(score))
        # accuracy
        acc = np.where(np.array(pred) == test_labels, 1, 0)
        acc = np.sum(acc) / acc.shape[0]
        # Macro-F1
        case = np.zeros((len(classes), 3))  # 4 int in dimension 1: TP, FP, FN
        for i in range(N):
            if (pred[i] == test_labels[i]):  # TP
                case[pred[i], 0] += 1
            else:
                case[pred[i], 1] += 1
                case[test_labels[i], 2] += 1
        precision = case[:, 0] / (case[:, 0] + case[:, 1])
        recall = case[:, 0] / (case[:, 0] + case[:, 2])
        F1 = np.average((2 * precision * recall) / (precision + recall))
        return acc, F1


class NaiveBayesModel:
    def __init__(self,
                 dataset,
                 train_data,
                 train_labels,
                 test_data,
                 test_labels,
                 vocab,
                 classes,
                 ):
        self.dataset = dataset
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.vocab = vocab
        self.classes = classes
        self.prob = np.zeros(len(classes))
        self.cond_prob = np.zeros((len(vocab), len(classes)))

    def train(self):
        cnt_class = np.zeros(len(classes))
        cnt_coappear = np.zeros((len(vocab), len(classes)))
        for i in trange(self.train_data.shape[0]):
            data = self.train_data[i]
            label = self.train_labels[i]
            cnt_class[label] += 1
            cnt_coappear[:, label] += data
        self.prob = cnt_class / np.sum(cnt_class)
        self.cond_prob = cnt_coappear / cnt_class

    def predict(self, x):
        indices = np.where(x > 0)
        prob = np.prod(self.cond_prob[indices], axis=0) * self.prob
        return np.argmax(prob)

    def test(self):
        N = len(test_labels)
        pred = []
        for i in range(N):
            pred.append(self.predict(self.train_data[i]))
        # accuracy
        acc = np.where(np.array(pred) == test_labels, 1, 0)
        print(acc)
        acc = np.sum(acc) / acc.shape[0]
        # Macro-F1
        case = np.zeros((len(classes), 3))  # 4 int in dimension 1: TP, FP, FN
        for i in range(N):
            if (pred[i] == test_labels[i]):  # TP
                case[pred[i], 0] += 1
            else:
                case[pred[i], 1] += 1
                case[test_labels[i], 2] += 1
        precision = case[:, 0] / (case[:, 0] + case[:, 1])
        recall = case[:, 0] / (case[:, 0] + case[:, 2])
        F1 = np.average((2 * precision * recall) / (precision + recall))
        return acc, F1


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(description='Run Log-Linear Model for Classification')
    parser.add_argument('--model', default='LogLinear')
    parser.add_argument('--dataset', default='yelp')
    parser.add_argument('--method', default='DF')
    parser.add_argument('--n_features', default=2000)
    parser.add_argument('--batch_size', default=1000)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--alpha', default=1e-4)
    parser.add_argument('--epoch', default=1)
    parser.add_argument('--save_path')
    args = parser.parse_args()
    '''
    sst:
        n_features = 5000
        batch_size = 10000
        learning_rate = 0.01
        alpha = 1e-4
        epoch = 20
    yelp:
        n_features = 2000 (5000: runtime error, out of memory)
        batch_size = 1000
        learning_rate = 0.001
        alpha = 1e-4
        epoch = 1
    '''
    model_name = args.model
    dataset = args.dataset
    method = args.method
    n_features = int(args.n_features)
    batch_size = int(args.batch_size)
    learning_rate = float(args.lr)
    alpha = float(args.alpha)  # regularization
    epoch = int(args.epoch)
    save_path = args.save_path
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f'model_name: {model_name}\n')
        f.write(f'dataset: {dataset}\n')
        f.write(f'method: {method}\n')
        f.write(f'n_features: {n_features}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'learning_rate: {learning_rate}\n')
        f.write(f'epoch: {epoch}\n')
        f.write('\n')

    print("Load preprocessed data from files")
    train_labels = load_data(f'data/{dataset}_train.csv')[1]
    valid_labels = train_labels[-int(len(train_labels) / 5):]
    train_labels = train_labels[:-int(len(train_labels) / 5)]
    test_labels = load_data(f'data/{dataset}_test.csv')[1]
    if dataset == 'yelp':
        train_labels = (np.array(train_labels) - 1).tolist()
        valid_labels = (np.array(valid_labels) - 1).tolist()
        test_labels = (np.array(test_labels) - 1).tolist()
    train_texts = open(f'data/{dataset}/train.txt').read().strip().split('\n')
    train_texts = [sent.split(' ') for sent in train_texts]
    valid_texts = open(f'data/{dataset}/valid.txt').read().strip().split('\n')
    valid_texts = [sent.split(' ') for sent in valid_texts]
    test_texts = open(f'data/{dataset}/test.txt').read().strip().split('\n')
    test_texts = [sent.split(' ') for sent in test_texts]
    classes = unique(train_labels)

    # Extract features from the preprocessed texts
    # Native feature space: the unique terms (words or phrases) that occur in documents. That's too many!
    # Try to avoid manual definition or construction here.
    # removal of non-informative terms & construction of new features
    # - Document frequency(DF)
    # - Information gain(IG)
    print('Start feature extraction')
    t0 = time.time()
    vocab = feature_extraction(train_texts, train_labels, classes, n_features, method)
    t1 = time.time()
    print(f'Extracting features costs {t1 - t0} seconds')

    if model_name == 'LogLinear':
        print('Turn texts into numpy arrays')
        train_data = text2vec(train_texts, vocab)  # ndarray, shape (N, n_features)
        valid_data = text2vec(valid_texts, vocab)
        test_data = text2vec(test_texts, vocab)
        print('Build the Log-Linear model')
        model = LLModel(dataset, train_data, train_labels, valid_data, valid_labels, test_data, test_labels,
                        vocab, classes, alpha, save_path)
        print('Start training')
        t2 = time.time()
        model.train(batch_size, learning_rate, epoch)
        t3 = time.time()
        print(f'Training costs {t3 - t2} seconds')
    else:
        print('Turn texts into numpy arrays')
        train_data = text2vec(train_texts + valid_texts, vocab)
        test_data = text2vec(test_texts, vocab)
        train_labels = train_labels + valid_labels
        print('Build the Naive Bayes model')
        model = NaiveBayesModel(dataset, train_data, train_labels, test_data, test_labels, vocab, classes)
        print('Start training')
        t2 = time.time()
        model.train()
        t3 = time.time()
        print(f'Training costs {t3 - t2} seconds')

    print('Start testing')
    acc, F1 = model.test()
    print(f"test result: acc {acc}, F1 {F1}")
    with open(save_path, 'a') as f:
        f.write(f"test result: acc {acc}, F1 {F1}\n")

    print("==============================================")
    print("\n")
