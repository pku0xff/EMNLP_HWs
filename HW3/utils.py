import json
import nltk
from nltk.stem import WordNetLemmatizer


def read_data(filename):
    with open(filename, encoding='utf8') as f:
        data = json.loads(f.read())
    # data: 数据列表
    print(f'Reading {filename}')
    print(f'Number of instances: {len(data)}')
    len_sentences = [len(i['words']) for i in data]
    n_events = [len(i['events']) for i in data]
    print(f'Avg sentence length: {sum(len_sentences) / len(len_sentences)}')
    print(f'Avg event num: {sum(n_events) / len(n_events)}')
    print()
    return data


def get_feature_t(sent):
    '''
    local feature:
    1. words (2 words)
    2. pos_tag
    '''
    lmt = WordNetLemmatizer()
    sent_features = []
    pos_tags = nltk.pos_tag(sent)
    pos_tags = [p[1] for p in pos_tags]
    L = len(sent)
    for i in range(L):
        token_local_feature = {
            # lexical
            'curr_token': sent[i],
            'next_token': '' if i == L - 1 else sent[i + 1],
            'curr_pos': pos_tags[i],
            'next_pos': '' if i == L - 1 else pos_tags[i + 1],
            'lemma': lmt.lemmatize(sent[i]),
            # 'synonym':'',

            # syntactic

            # entity information

        }
        sent_features.append(token_local_feature)
    return sent_features


def trigger_tag(instance):
    text = instance['words']
    t_tag = [0] * len(text)
    events = instance['events']
    for event in events:
        trigger = event['trigger']
        for i in range(trigger['start'], trigger['end']):
            t_tag[i] = 1
    return t_tag


def trigger_type(instance, type2idx):
    text = instance['words']
    t_type = [''] * len(text)
    events = instance['events']
    for event in events:
        trigger = event['trigger']
        for i in range(trigger['start'], trigger['end']):
            t_type[i] = trigger['trigger-type']
    t_type = [type2idx[t] if t else 100 for t in t_type]
    return t_type
