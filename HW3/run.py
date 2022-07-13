import spacy
import torch
from torchtext.vocab import GloVe, vocab
from torchtext.data.utils import get_tokenizer
import json
import nltk

from trigger_detection import LinearModel, train_vectorizer, form_data, vectorize_data, get_feature
from trigger_classification import RNN, t_type2idx, idx2t_type
from argument_identification import get_arg_feature, get_candicates, read_arg_data, build_feature_for_id
from argument_classification import idx2role
from argument_classification import LinearModel as MultiClassLinearModel
from utils import trigger_type
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction import DictVectorizer

pred_data = []
with open('data/test.json', encoding='utf8') as f:
    gt_data = json.loads(f.read())

print('\n\n')
print('Build the modules all together!')

###################################
# STEP 1: Trigger Detection
###################################

# prepare the trained model
t_detection_vectorizer = train_vectorizer()
test_features, test_tags = form_data('data/test.json')
test_features = vectorize_data(test_features, t_detection_vectorizer)
n_feature = test_features[0].shape[1]
t_detection_model = LinearModel(n_feature)
t_detection_model.load('trigger_detection.npy')

# predict
pred_1 = []
gt_1 = []
for gt_sent in gt_data:
    words = gt_sent['words']
    pred_sent = dict()
    pred_sent['words'] = words
    pred_sent['events'] = []
    pos_tags = nltk.pos_tag(words)
    pos_tags = [p[1] for p in pos_tags]
    gt_triggers = [gt_event['trigger']['text'] for gt_event in gt_sent['events']]
    for token_idx, token in enumerate(words):
        feature = get_feature(token, token_idx, words, pos_tags)
        feature = t_detection_vectorizer.transform(feature)
        pred = t_detection_model.pred_by_word(feature)
        if pred:
            event = dict()
            trigger = dict()
            trigger['text'] = token
            trigger['start'] = token_idx
            trigger['end'] = token_idx + 1
            event['trigger'] = trigger
            pred_sent['events'].append(event)
            pred_1.append(1)
        else:
            pred_1.append(0)
        gt_1.append(1 if token in gt_triggers else 0)
    pred_data.append(pred_sent)

print('Results of trigger detection:')
acc = accuracy_score(gt_1, pred_1)
p = precision_score(gt_1, pred_1, average='macro')
r = recall_score(gt_1, pred_1, average='macro')
f1 = f1_score(gt_1, pred_1, average='macro')
print(f"acc {acc}, precision {p}, recall {r}, f1 {f1}.")

###################################
# STEP 2: Trigger Classification
###################################

# prepare the trained model
glove_vectors = GloVe(name='6B', dim=100)
glove_vocab = vocab(glove_vectors.stoi)
glove_vocab.insert_token('<unk>', 0)
glove_vocab.set_default_index(0)
pretrained_embeddings = glove_vectors.vectors
pretrained_embeddings = torch.cat((torch.zeros(1, pretrained_embeddings.shape[1]), pretrained_embeddings))
tokenizer = get_tokenizer('basic_english')
offset_input = torch.tensor([0])
t_classification_model = torch.load('trigger_classification.pt')

# predict
pred_2 = []
gt_2 = []
pred_1_cp = pred_1
for i, gt_sent in enumerate(gt_data):
    words = gt_sent['words']
    tag_label = pred_1_cp[:len(words)]
    pred_1_cp = pred_1_cp[len(words):]
    true_label = trigger_type(gt_sent, t_type2idx)
    hidden = t_classification_model.init_hidden()
    for j, word in enumerate(words):
        token = tokenizer(word)
        indice = glove_vocab(token)
        input_ = torch.tensor(indice).int()
        output, hidden = t_classification_model(input_, hidden)
        output = torch.argmax(output)
        o = output.item()
        if tag_label[j]:
            pred_2.append(o)
            for k, event in enumerate(pred_data[i]['events']):
                if event['trigger']['text'] == word:
                    pred_data[i]['events'][k]['trigger']['trigger-type'] = idx2t_type[o]
        else:
            pred_2.append(100)
    gt_2 += true_label

print('Results of trigger classification:')
acc = accuracy_score(gt_2, pred_2)
p = precision_score(gt_2, pred_2, average='macro')
r = recall_score(gt_2, pred_2, average='macro')
f1 = f1_score(gt_2, pred_2, average='macro')
print(f"acc {acc}, precision {p}, recall {r}, f1 {f1}.")

###################################
# STEP 3: Argument Identification
# STEP 4: Argument Classification
###################################

nlp = spacy.load('en_core_web_sm')
train_instances = read_arg_data(nlp, 'data/train.json')
train_features, _ = build_feature_for_id(train_instances)
arg_vectorizer = DictVectorizer()
arg_vectorizer.fit(train_features)
feature_size = (arg_vectorizer.transform(train_features[0])).shape[1]
arg_id_model = LinearModel(feature_size)
arg_id_model.load('argument_identification.npy')
arg_classification_model = MultiClassLinearModel(0, 0, 0, 0, 0, 0, feature_size, 0)
arg_classification_model.load('argument_classification.npy')


def trigger_equal(trigger_1, trigger_2):
    if trigger_1['start'] == trigger_2['start'] and trigger_1['end'] == trigger_2['end'] \
            and trigger_1['trigger-type'] == trigger_2['trigger-type']:
        return True
    else:
        return False


def argument_equal(trigger_1, trigger_2, arg_1, arg_2):
    if not trigger_equal(trigger_1, trigger_2):
        return 0
    if arg_1['start'] != arg_2['start'] or arg_1['end'] != arg_2['end']:
        return 0
    if arg_1['role'] == arg_2['role']:  # correctly identified
        return 2
    else:  # correctly identified and classified
        return 1


identify_t = 0
identify_f = 0
classify_t = 0
classify_f = 0
n_gt_triggers = 0
n_gt_args = 0
for i, pred_sent in enumerate(pred_data):
    words = pred_sent['words']
    cand_args = get_candicates(nlp, words)
    pred_events = pred_sent['events']
    gt_events = gt_data[i]['events']
    n_gt_triggers += len(gt_events)
    for gt_event in gt_events:
        n_gt_args += len(gt_event['arguments'])
    for j, pred_event in enumerate(pred_events):
        pred_trigger = pred_event['trigger']
        # If this is a true trigger of an event?
        gt_event_paired = dict()
        for gt_event in gt_events:
            if trigger_equal(pred_trigger, gt_event['trigger']):
                gt_event_paired = gt_event
        for cand_arg in cand_args:
            feature = get_arg_feature(words, pred_trigger, cand_arg)
            feature = arg_vectorizer.transform(feature)
            pred_arg_extract = arg_id_model.pred_by_word(feature)
            if pred_arg_extract:
                pred_arg = cand_arg
                pred_arg_class = arg_classification_model.predict(feature)
                pred_arg_role = idx2role[pred_arg_class]
                pred_arg['role'] = pred_arg_role
                if gt_event_paired:
                    for gt_arg in gt_event_paired['arguments']:
                        result = argument_equal(gt_event_paired['trigger'], pred_trigger, gt_arg, pred_arg)
                        if result == 2:
                            identify_t += 1
                            classify_t += 1
                        elif result == 1:
                            identify_t += 1
                            classify_f += 1
                        else:
                            identify_f += 1
                            classify_f += 1
                else:
                    identify_f += 1
                    classify_f += 1

# calculate metrics for STEP 3
TP = identify_t
FP = identify_f
FN = n_gt_args - TP
print('Results of argument identification:')
print(f'TP {TP}, FP {FP}, FN {FN}')
print(f"precision {TP / (TP + FP)}, recall {TP / (TP + FN)}, f1 {(2 * TP) / (2 * TP + FP + FN)}.")

# calculate metrics for STEP 4
TP = classify_t
FP = classify_f
FN = n_gt_args - TP
print('Results of argument classification:')
print(f'TP {TP}, FP {FP}, FN {FN}')
print(f"precision {TP / (TP + FP)}, recall {TP / (TP + FN)}, f1 {(2 * TP) / (2 * TP + FP + FN)}.")
