import json

from trigger_detection import LinearModel, train, evaluate
from sklearn.feature_extraction import DictVectorizer
import spacy


def get_candicates(nlp, sent):
    doc = nlp(' '.join(sent))
    arg_candidates = []
    for i in range(len(sent)):
        arg_text = list(doc[i].subtree)
        arg_text = [str(_) for _ in arg_text]
        arg_s = 0
        arg_e = 0
        L = len(arg_text)
        for j in range(len(sent) - len(arg_text)):
            if sent[j:j + L] == arg_text:
                arg_s = j
                arg_e = j + L
                break
        arg_candidates.append({'start': arg_s, 'end': arg_e})

    return arg_candidates


# Filter candidate arguments:
# Linear Model + Perception Algorithm

# Get the feature dictionary of 1 argument
def get_arg_feature(sent, trigger, argument):
    tri_s = trigger['start']
    tri_e = trigger['end']
    tri_type = trigger['trigger-type']
    arg_s = argument['start']
    arg_e = argument['end']
    if tri_e >= arg_s:
        relative_pos = 'after'
    elif tri_s <= arg_e:
        relative_pos = 'before'
    else:
        relative_pos = 'overlap'
    arg_feature = {
        'arg_length': arg_e - arg_s,
        'tokens': set(sent[arg_s:arg_e]),
        # 'last_token': '' if arg_s == 0 else sent[arg_s - 1],
        # 'next_token': '' if arg_e == len(sent) else sent[arg_e],
        'trigger_word': ' '.join(sent[tri_s:tri_e]),
        'trigger_type': tri_type.split(':')[0],
        'trigger_subtype': tri_type.split(':')[1],
        'relative_pos': relative_pos,
    }
    return arg_feature


# Read data from file and turn it into event instances
def read_arg_data(nlp, filepath):
    with open(filepath, encoding='utf8') as f:
        data = json.loads(f.read())

    instances = []
    for sent in data:
        text = sent['words']
        for event in sent['events']:
            trigger = event['trigger']
            gt_args = event['arguments']
            cand_args = get_candicates(nlp, text)
            instances.append((text, trigger, gt_args, cand_args))
    return instances


# From event instances to argument features and labels
# feature: a dictionary for a candidate argument
# label: Is this candidate an argument of the trigger?
# trigger: use ground true triggers to train and results from part 1 to test
# argument: use argument candidates from dependency parsing to train and test
def build_feature_for_id(instances):
    features = []
    tags = []
    for instance in instances:
        text, trigger, gt_args, cand_args = instance
        gt_args = [(i['start'], i['end']) for i in gt_args]
        for cand_arg in cand_args:
            feature = get_arg_feature(text, trigger, cand_arg)
            features.append(feature)
            cand_arg = (cand_arg['start'], cand_arg['end'])
            if cand_arg in gt_args:
                tags.append(1)
            else:
                tags.append(0)
    return features, tags


def main():
    print('\n\n')
    print('Train Argument Identification Model')
    nlp = spacy.load('en_core_web_sm')
    train_instances = read_arg_data(nlp, 'data/train.json')
    valid_instances = read_arg_data(nlp, 'data/valid.json')
    test_instances = read_arg_data(nlp, 'data/test.json')
    # print('finish reading')
    train_features, train_tags = build_feature_for_id(train_instances)
    valid_features, valid_tags = build_feature_for_id(valid_instances)
    test_features, test_tags = build_feature_for_id(test_instances)
    # print('finish feature extraction')
    vectorizer = DictVectorizer()
    vectorizer.fit(train_features)
    train_features = [vectorizer.transform(i) for i in train_features]
    valid_features = [vectorizer.transform(i) for i in valid_features]
    test_features = [vectorizer.transform(i) for i in test_features]
    # print('finish vectorization')
    n_feature = test_features[0].shape[1]
    print(f'Feature size: {n_feature}')
    model = LinearModel(n_feature)

    n_epoch = 8

    train(model, train_features, train_tags, valid_features, valid_tags, n_epoch)
    # print('finish training')
    model.save('argument_identification.npy')
    acc, p, r, f1, preds = evaluate(model, test_features, test_tags)
    print(f"Test: acc {acc}, precision {p}, recall {r}, F1 {f1}.")


if __name__ == '__main__':
    main()
