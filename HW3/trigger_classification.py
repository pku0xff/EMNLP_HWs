import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchtext.vocab import GloVe, vocab
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import json
from utils import trigger_tag, trigger_type
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

t_type2idx = {'Justice:Execute': 0, 'Justice:Pardon': 1, 'Transaction:Transfer-Money': 2, 'Justice:Release-Parole': 3,
              'Justice:Fine': 4, 'Business:Merge-Org': 5, 'Personnel:End-Position': 6, 'Justice:Acquit': 7,
              'Life:Injure': 8, 'Contact:Phone-Write': 9, 'Justice:Arrest-Jail': 10, 'Personnel:Nominate': 11,
              'Life:Divorce': 12, 'Justice:Extradite': 13, 'Life:Marry': 14, 'Business:Declare-Bankruptcy': 15,
              'Movement:Transport': 16, 'Life:Die': 17, 'Conflict:Demonstrate': 18, 'Justice:Sue': 19,
              'Business:End-Org': 20, 'Justice:Convict': 21, 'Contact:Meet': 22, 'Justice:Trial-Hearing': 23,
              'Personnel:Elect': 24, 'Transaction:Transfer-Ownership': 25, 'Business:Start-Org': 26,
              'Justice:Charge-Indict': 27, 'Personnel:Start-Position': 28, 'Life:Be-Born': 29, 'Justice:Appeal': 30,
              'Justice:Sentence': 31, 'Conflict:Attack': 32}

idx2t_type = {i: s for s, i in t_type2idx.items()}


class RNN(nn.Module):
    def __init__(self, pretrained_embeddings, offset_input, hidden_size=256, output_size=33):
        super(RNN, self).__init__()
        self.offset_input = offset_input
        self.hidden_size = hidden_size
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings, freeze=False, sparse=True)
        self.i2h = nn.Linear(pretrained_embeddings.shape[1] + hidden_size, hidden_size)
        self.i2o = nn.Linear(pretrained_embeddings.shape[1] + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_, hidden):
        embedded = self.embedding(input_, self.offset_input)
        combined = torch.cat((embedded, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# train with 1 sentence
def train_step(model, data, tokenizer, glove_vocab, optimizer, criterion):
    optimizer.zero_grad()
    batch_loss = 0
    text = data['words']
    tag_label = trigger_tag(data)
    type_label = trigger_type(data, t_type2idx)
    hidden = model.init_hidden()
    loss = torch.tensor(0.0)
    for i in range(len(text)):
        word = text[i]
        token = tokenizer(word)
        indice = glove_vocab(token)
        input_ = torch.tensor(indice).int()
        output, hidden = model(input_, hidden)
        if tag_label[i]:
            loss += criterion(output, torch.tensor(type_label[i]).view(-1))
    loss.backward(retain_graph=True)

    batch_loss += loss.item()
    optimizer.step()

    return batch_loss


def evaluate(model, tokenizer, glove_vocab, valid_data):
    labels = []
    preds = []
    for i, d in tqdm(enumerate(valid_data)):
        text = d['words']
        tag_label = trigger_tag(d)
        type_label = trigger_type(d, t_type2idx)
        hidden = model.init_hidden()
        true_label = []
        pred_label = []
        for j, word in enumerate(text):
            token = tokenizer(word)
            indice = glove_vocab(token)
            input_ = torch.tensor(indice).int()
            output, hidden = model(input_, hidden)
            output = torch.argmax(output)
            o = output.item()
            if tag_label[j]:
                pred_label.append(o)
                true_label.append(type_label[j])
        labels += true_label
        preds += pred_label
    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds, average='macro')
    r = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return acc, p, r, f1


def train_RNN(model, tokenizer, glove_vocab, train_data, valid_data, lr, num_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_arr = []
    acc_log = []
    F1_log = []

    for epoch in tqdm(range(num_epoch)):
        model.train()
        loss_log = []
        random.shuffle(train_data)
        for i, batch_data in enumerate(train_data):
            batch_loss = train_step(model, batch_data, tokenizer, glove_vocab, optimizer, criterion)
            loss_log.append(batch_loss)
            if (i + 1) % 500 == 0:
                # print(f"Iteration number: {i + 1}. Loss: {batch_loss}.")
                pass
        loss_arr.append(np.mean(loss_log))
        print(f"Epoch {epoch} loss: {np.mean(loss_log)}.")

        model.eval()
        acc, p, r, f1 = evaluate(model, tokenizer, glove_vocab, valid_data)
        acc_log.append(acc)
        F1_log.append(f1)
        print(f"Eval: acc {acc}, precision {p}, recall {r}, F1 {f1}.")

    '''
    plt.figure()
    plt.plot(loss_arr, '-x')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.plot(acc_log, '-x', label='acc')
    plt.plot(F1_log, '-o', label='F1')
    plt.legend()
    plt.show()
    '''


def test_RNN(model, tokenizer, glove_vocab, test_data):
    model.eval()
    acc, p, r, f1 = evaluate(model, tokenizer, glove_vocab, test_data)
    print(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.")


def main():
    print('\n\n')
    print('Train Trigger Classification Model')
    glove_vectors = GloVe(name='6B', dim=100)
    glove_vocab = vocab(glove_vectors.stoi)
    glove_vocab.insert_token('<unk>', 0)
    glove_vocab.set_default_index(0)
    pretrained_embeddings = glove_vectors.vectors
    pretrained_embeddings = torch.cat((torch.zeros(1, pretrained_embeddings.shape[1]), pretrained_embeddings))
    tokenizer = get_tokenizer('basic_english')
    offset_input = torch.tensor([0])

    # Build the model.
    model = RNN(pretrained_embeddings, offset_input)

    # Read data.
    with open('data/train.json', encoding='utf8') as f:
        train_data = json.load(f)

    with open('data/valid.json', encoding='utf8') as f:
        valid_data = json.load(f)

    with open('data/test.json', encoding='utf8') as f:
        test_data = json.load(f)

    # Set arguments.
    lr = 0.5 * 1e-2
    n_epoch = 7

    # Train and test.
    train_RNN(model, tokenizer, glove_vocab, train_data, valid_data, lr, n_epoch)
    test_RNN(model, tokenizer, glove_vocab, test_data)
    torch.save(model, 'trigger_classification.pt')


if __name__ == '__main__':
    main()
