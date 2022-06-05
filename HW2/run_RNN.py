import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchtext.vocab import GloVe, vocab
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import json
import argparse
import time


class RNN(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_size=128, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings, freeze=False, sparse=True)
        self.i2h = nn.Linear(pretrained_embeddings.shape[1] + hidden_size, hidden_size)
        self.i2o = nn.Linear(pretrained_embeddings.shape[1] + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_, hidden):
        embedded = self.embedding(input_, offset_input)
        combined = torch.cat((embedded, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def train_step(model, data, optimizer, criterion):
    optimizer.zero_grad()
    batch_loss = 0
    text = data['words']
    label = data['triggers']
    hidden = model.init_hidden()
    loss = torch.tensor(0.0)
    for i in range(len(text)):
        word = text[i]
        token = tokenizer(word)
        indice = glove_vocab(token)
        input_ = torch.tensor(indice).int()
        output, hidden = model(input_, hidden)
        loss += criterion(output.view(-1), torch.tensor(label[i]).view(-1).float())
    loss.backward(retain_graph=True)

    batch_loss += loss.item()
    optimizer.step()

    return batch_loss


def prob2metrics(probs, labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(labels)):
        p = np.array(probs[i])
        p = np.where(p > 0.5, 1, 0)
        l = np.array(labels[i])
        TP += np.count_nonzero(p + l > 1)
        FP += np.count_nonzero(p - l == 1)
        TN += np.count_nonzero(p + l == 0)
        FN += np.count_nonzero(p - l < 0)
    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP + 1e-4)
    recall = TP / (TP + FN + 1e-4)
    F1 = 2 * precision * recall / (precision + recall + 1e-4)
    return acc, precision, recall, F1


def evaluate(model, eval_data, test=False):
    labels = []
    probs = []
    f = open('False_RNN.txt', 'w', encoding='utf8')
    for i, d in tqdm(enumerate(eval_data)):
        text = d['words']
        label = d['triggers']
        hidden = model.init_hidden()
        pred_label = []
        for j, word in enumerate(text):
            token = tokenizer(word)
            indice = glove_vocab(token)
            input_ = torch.tensor(indice).int()
            output, hidden = model(input_, hidden)
            o = output.item()
            # Record tokens that are wrongly predicted in test time.
            if test:
                if o > 0.5 and label[j] == 0:
                    f.write(f'FP\t{i}\t{j}\n')
                if o <= 0.5 and label[j] == 1:
                    f.write(f'FN\t{i}\t{j}\n')
            pred_label.append(o)
        labels.append(label)
        probs.append(pred_label)
    f.close()
    return prob2metrics(probs, labels)


def train_RNN(train_data, eval_data, f, lr, num_epoch):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_arr = []
    acc_log = []
    F1_log = []

    for epoch in tqdm(range(num_epoch)):
        print(f'Epoch number: {epoch}.')
        f.write(f'Epoch number: {epoch}.\n')
        model.train()
        loss_log = []
        for i, batch_data in enumerate(train_data):
            batch_loss = train_step(model, batch_data, optimizer, criterion)
            loss_log.append(batch_loss)
            if (i + 1) % 500 == 0:
                print(f"Iteration number: {i + 1}. Loss: {batch_loss}.")
                f.write(f"Iteration number: {i + 1}. Loss: {batch_loss}.\n")
        loss_arr.append(np.mean(loss_log))
        print(f"Epoch loss: {np.mean(loss_log)}.")
        f.write(f"Epoch loss: {np.mean(loss_log)}.\n")

        model.eval()
        acc, p, r, f1 = evaluate(model, eval_data)
        acc_log.append(acc)
        F1_log.append(f1)
        print(f"Eval: acc {acc}, precision {p}, recall {r}, F1 {f1}.")
        f.write(f"Eval: acc {acc}, precision {p}, recall {r}, F1 {f1}.\n\n")

    plt.figure()
    plt.plot(loss_arr, '-x')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('loss_RNN.pdf')

    plt.figure()
    plt.plot(acc_log, '-x', label='acc')
    plt.plot(F1_log, '-o', label='F1')
    plt.legend()
    plt.savefig('eval_RNN.pdf')


def test_RNN(test_data, f):
    model.eval()
    acc, p, r, f1 = evaluate(model, test_data, test=True)
    print(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.")
    f.write(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.\n")


# Read pretrained GloVe vectors.
glove_vectors = GloVe(name='6B', dim=100)
glove_vocab = vocab(glove_vectors.stoi)
glove_vocab.insert_token('<unk>', 0)
glove_vocab.set_default_index(0)
pretrained_embeddings = glove_vectors.vectors
pretrained_embeddings = torch.cat((torch.zeros(1, pretrained_embeddings.shape[1]), pretrained_embeddings))
tokenizer = get_tokenizer('basic_english')
offset_input = torch.tensor([0])

# Build the model.
model = RNN(pretrained_embeddings)

# Read data.
with open('data/train.json', encoding='utf8') as f:
    train_data = json.load(f)

with open('data/valid.json', encoding='utf8') as f:
    eval_data = json.load(f)

with open('data/test.json', encoding='utf8') as f:
    test_data = json.load(f)

# Set arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2)
parser.add_argument('--n_epoch', default=16)
args = parser.parse_args()
lr = float(args.lr)
n_epoch = int(args.n_epoch)

# Train and test.
f = open('RNN_log.txt', 'w', encoding='utf-8')
t0 = time.time()
train_RNN(train_data, eval_data, f, lr=lr, num_epoch=n_epoch)
t1 = time.time()
print(f"Training cost {t1 - t0} seconds.")
f.write(f"Training cost {t1 - t0} seconds.\n\n")
test_RNN(test_data, f)
f.close()
