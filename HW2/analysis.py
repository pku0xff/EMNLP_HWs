import json
import nltk
import numpy as np
from collections import defaultdict


# Read data and report some infomation.


def read_file(filepath):
    with open(filepath, encoding='utf8') as f:
        data = json.load(f)
    print(f'Report of {filepath}')
    print(f'Length: {len(data)}')
    avg_sent_length = 0
    sent_list = []
    tag_list = []
    n_trigger = 0
    for sent in data:
        words = sent['words']
        sent_list.append(words)
        triggers = sent['triggers']
        tag_list.append(triggers)
        avg_sent_length += len(words)
        n_trigger += sum(triggers)
    avg_sent_length /= len(data)
    print(f'Average length of sentences: {avg_sent_length}')
    print(f'Trigger number in total: {n_trigger}')
    print(f'Average trigger number per sentence: {n_trigger / len(data)}')
    print()
    return sent_list, tag_list


# Sentences are flattened in Linear Model. Need to recover the position of tokens that are wrongly predicted.
def locate_for_LinearModel(tag, sent_list):
    filename = 'linear_fp.npy' if tag == 'FP' else 'linear_fn.npy'
    src_list = np.load(filename).tolist()
    tgt_list = []
    sent_len_list = [len(sent) for sent in sent_list] + [1000]
    for i, sent_len in enumerate(sent_len_list):
        new_src_list = []
        for idx in src_list:
            if idx < sent_len:
                tgt_list.append([i, idx])
            else:
                new_src_list.append(idx - sent_len)
        src_list = new_src_list
    if len(src_list) > 0:
        raise ValueError
    return tgt_list


def locate_for_rnn():
    fp_list = []
    fn_list = []
    lines = open('False_RNN.txt', encoding='utf8').read().strip().split('\n')
    for line in lines:
        line = line.split()
        if 'P' in line[0]:
            fp_list.append([int(line[1]), int(line[2])])
        else:
            fn_list.append([int(line[1]), int(line[2])])
    return fp_list, fn_list


def pos_tag(sent_list):
    pos_list = []
    for sent in sent_list:
        sent_pos = nltk.pos_tag(sent)
        sent_pos = [p[1] for p in sent_pos]
        pos_list.append(sent_pos)
    return pos_list


# For the wrongly predicted tokens, find the sentence lengths,
# numbers of triggers in sentences and pos tags.
def analysis_wrong_pred(false_list):
    len_sent = []
    n_trigger = []
    pos_cnt = defaultdict(int)
    for i, j in false_list:
        len_sent.append(len(sent_list[i]))
        n_trigger.append(sum(tag_list[i]))
        pos_cnt[pos_list[i][j]] += 1
    return sum(len_sent) / len(len_sent), sum(n_trigger) / len(n_trigger), pos_cnt


_ = read_file('data/train.json')
_ = read_file('data/valid.json')
sent_list, tag_list = read_file('data/test.json')
pos_list = pos_tag(sent_list)

fp_linear = locate_for_LinearModel('FP', sent_list)
fn_linear = locate_for_LinearModel('FN', sent_list)
fp_rnn, fn_rnn = locate_for_rnn()

print('FP, Linear Model')
avg_sent_length, avg_trigger_num, pos_cnt = analysis_wrong_pred(fp_linear)
print(f'avg sent length {avg_sent_length}')
print(f'avg trigger num {avg_trigger_num}')
print(f'pos cnt\n{pos_cnt}')
print()

print('FN, Linear Model')
avg_sent_length, avg_trigger_num, pos_cnt = analysis_wrong_pred(fn_linear)
print(f'avg sent length {avg_sent_length}')
print(f'avg trigger num {avg_trigger_num}')
print(f'pos cnt\n{pos_cnt}')
print()

print('FP, RNN')
avg_sent_length, avg_trigger_num, pos_cnt = analysis_wrong_pred(fp_rnn)
print(f'avg sent length {avg_sent_length}')
print(f'avg trigger num {avg_trigger_num}')
print(f'pos cnt\n{pos_cnt}')
print()

print('FN, RNN')
avg_sent_length, avg_trigger_num, pos_cnt = analysis_wrong_pred(fn_rnn)
print(f'avg sent length {avg_sent_length}')
print(f'avg trigger num {avg_trigger_num}')
print(f'pos cnt\n{pos_cnt}')
print()
