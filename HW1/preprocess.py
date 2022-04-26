from cgitb import text
import csv
import time
import re
import argparse
from tqdm.std import trange


def load_data(data_path):
    texts = []
    labels = []

    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header

        for row in reader:
            text = row[0].strip()
            label = int(row[1].strip())

            texts.append(text)
            labels.append(label)

    return texts, labels


def unique(xs):
    ret = set()
    for x in xs:
        ret.add(x)
    return ret


def calculate_avg_length(texts):
    cnt = 0
    for t in texts:
        cnt += len(t.split())
    return cnt / len(texts)


def write_to_file(texts, filename):
    with open(filename, 'w') as f:
        for sent in texts:
            f.write(' '.join(sent) + '\n')
    print(f'Write data into {filename}.')


def remove_stop_words(texts, stop_words):
    new_texts = []
    for i in trange(len(texts)):
        sent = texts[i]
        tmp = sent.copy()
        for w in sent:
            if w in stop_words:
                tmp.remove(w)
        new_texts.append(tmp)
    return new_texts


def lemmatization(texts, lemma_dict):
    new_texts = []
    for i in trange(len(texts)):
        sent = texts[i]
        tmp = set()
        for w in sent:
            if len(w) < 2 or len(w) > 12:
                continue
            try:
                tmp.add(lemma_dict[w])
            except:
                tmp.add(w)
        new_texts.append(tmp)
    return new_texts


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Preprocess text dataset')
    parser.add_argument('--dataset', default='yelp')
    args = parser.parse_args()
    dataset = args.dataset

    # Load data
    train_texts, train_labels = load_data(f'data/{dataset}_train.csv')
    valid_texts, valid_labels = train_texts[-int(len(train_texts) / 5):], train_labels[-int(len(train_texts) / 5):]
    train_texts, train_labels = train_texts[:-int(len(train_texts) / 5)], train_labels[:-int(len(train_texts) / 5)]
    test_texts, test_labels = load_data(f'data/{dataset}_test.csv')

    # Print basic statistics
    print("Training set size:", len(train_texts))  # sst 6836, yelp 520000
    print("Validation set size:", len(valid_texts))  # sst 1708, yelp 130000
    print("Test set size:", len(test_texts))  # sst 2210, yelp 50000
    categories = unique(train_labels)
    print("Unique labels:", categories)  # sst{0,1,2,3,4} yelp {1,2,3,4,5}
    print("Avg. length:", calculate_avg_length(train_texts + valid_texts + test_texts))  # sst 19, yelp 134

    print(f'Start preprocessing {dataset}')
    t0 = time.time()
    punc = '[^A-Za-z0-9]'
    train_texts = [set(re.sub(punc, ' ', sent).lower().split(' ')) for sent in train_texts]
    valid_texts = [set(re.sub(punc, ' ', sent).lower().split(' ')) for sent in valid_texts]
    test_texts = [set(re.sub(punc, ' ', sent).lower().split(' ')) for sent in test_texts]
    print("Finish removing punctuations")

    # remove stop words
    # stop words in nltk
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                  "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                  'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    train_texts = remove_stop_words(train_texts, stop_words)
    valid_texts = remove_stop_words(valid_texts, stop_words)
    test_texts = remove_stop_words(test_texts, stop_words)
    print("Finish removing stop words")

    # lemmatization
    lemma_list = open('data/lemmatization-en.txt', encoding='utf-8-sig').read().strip().split('\n')
    lemma_dict = {p.split()[1]: p.split()[0] for p in lemma_list}
    train_texts = lemmatization(train_texts, lemma_dict)
    valid_texts = lemmatization(valid_texts, lemma_dict)
    test_texts = lemmatization(test_texts, lemma_dict)
    print("Finish lemmatization")

    # save in files
    write_to_file(train_texts, f'data/{dataset}/train.txt')
    write_to_file(valid_texts, f'data/{dataset}/valid.txt')
    write_to_file(test_texts, f'data/{dataset}/test.txt')

    t1 = time.time()
    print(f'Preprocessing {dataset} cost {t1 - t0} seconds.')  # sst 3.47, yelp 2349.29
