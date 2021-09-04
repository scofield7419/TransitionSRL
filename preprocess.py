# -*- coding: utf-8 -*-

'''
   Read data from JSON files,
   in the meantime, we do preprocess like capitalize the first character of a sentence or normalize digits
'''
import os

import json
from collections import Counter

import numpy as np
import argparse

from io_utils import read_yaml, read_lines, read_json_lines, load_embedding_dict, save_pickle
from str_utils import capitalize_first_char, normalize_tok, normalize_sent, collapse_role_type
from vocab import Vocab

from actions import Actions

joint_config = read_yaml('joint_config.yaml')

parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--seed', '-s', required=False, type=int, default=joint_config['random_seed'])
args = parser.parse_args()
joint_config['random_seed'] = args.seed
print('seed:', joint_config['random_seed'])

np.random.seed(joint_config['random_seed'])

data_config = read_yaml('data_config.yaml')

data_dir = data_config['data_dir']
cur_dataset_dir = data_config['cur_dataset_dir']
embedding_dir = data_config['embedding_dir']
embedding_file = data_config['embedding_file']
embedding_type = data_config['embedding_type']

normalize_digits = data_config['normalize_digits']
lower_case = data_config['lower_case']

vocab_dir = data_config['vocab_dir']
token_vocab_file = os.path.join(vocab_dir, data_config['token_vocab_file'])
char_vocab_file = os.path.join(vocab_dir, data_config['char_vocab_file'])
prd_type_vocab_file = os.path.join(vocab_dir, data_config['prd_type_vocab_file'])
role_type_vocab_file = os.path.join(vocab_dir, data_config['role_type_vocab_file'])
action_vocab_file = os.path.join(vocab_dir, data_config['action_vocab_file'])
pos_vocab_file = os.path.join(vocab_dir, data_config['pos_vocab_file'])
dep_type_vocab_file = os.path.join(vocab_dir, data_config['dep_type_vocab_file'])

pickle_dir = data_config['pickle_dir']
vec_npy_file = data_config['vec_npy']
inst_pl_file = data_config['inst_pl_file']

train_list = read_json_lines(os.path.join(cur_dataset_dir, 'train.json'))
dev_list = read_json_lines(os.path.join(cur_dataset_dir, 'dev.json'))
test_list = read_json_lines(os.path.join(cur_dataset_dir, 'test.json'))

print('Sentence size Train: %d, Dev: %d, Test: %d' % (len(train_list), len(dev_list), len(test_list)))

embedd_dict, embedd_dim = None, None


def read_embedding():
    global embedd_dict, embedd_dim
    embedd_dict, embedd_dim = load_embedding_dict(embedding_type,
                                                  os.path.join(embedding_dir, embedding_file),
                                                  normalize_digits=normalize_digits)
    print('Embedding type %s, file %s' % (embedding_type, embedding_file))


def build_vocab():
    token_list = []
    char_list = []

    prd_type_list = []

    role_type_list = []

    actions_list = []

    pos_list = []
    dep_types_list = []

    for inst in train_list:
        
        words = inst['nlp_words']
        prds = inst['Predicates']  # idx, prds_type
        pairs = inst['Pair']  # arg_id, prd_id, role
        pos_list.extend(inst['nlp_pos'])
        dep_types_list.extend(inst['nlp_dep_types'])

        # arg_prd_pairs = []

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))

        for prd in prds:
            prd_type_list.append(prd[1].lower())

        for pair in pairs:
            role_type_list.append(pair[2])

        actions = Actions.make_oracle(words, pairs, prds)
        actions_list.extend(actions)

    train_token_set = set(token_list)

    for inst in dev_list:
        words = inst['nlp_words']
        prds = inst['Predicates']  # idx, prds_type
        pairs = inst['Pair']  # arg_id, prd_id, role
        pos_list.extend(inst['nlp_pos'])
        dep_types_list.extend(inst['nlp_dep_types'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))

        for prd in prds:
            prd_type_list.append(prd[1].lower())

        for pair in pairs:
            role_type_list.append(pair[2])

        actions = Actions.make_oracle(words, pairs, prds)
        actions_list.extend(actions)

    # test_oo_train_but_in_glove = 0
    for inst in test_list:
        words = inst['nlp_words']
        prds = inst['Predicates']  # idx, prds_type
        pairs = inst['Pair']  # arg_id, prd_id, role
        pos_list.extend(inst['nlp_pos'])
        dep_types_list.extend(inst['nlp_dep_types'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))

        for prd in prds:
            prd_type_list.append(prd[1].lower())

        for pair in pairs:
            role_type_list.append(pair[2])

        actions = Actions.make_oracle(words, pairs, prds)
        actions_list.extend(actions)

    print('--------token_vocab---------------')
    token_vocab = Vocab()
    token_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    token_vocab.add_counter(Counter(token_list))
    token_vocab.save(token_vocab_file)
    print(token_vocab)

    print('--------char_vocab---------------')
    char_vocab = Vocab()
    char_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    char_vocab.add_counter(Counter(char_list))
    char_vocab.save(char_vocab_file)
    print(char_vocab)

    print('--------prd_type_vocab---------------')
    prd_type_vocab = Vocab()
    prd_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    prd_type_vocab.add_counter(Counter(prd_type_list))
    prd_type_vocab.save(prd_type_vocab_file)
    print(prd_type_vocab)

    print('--------role_type_vocab---------------')
    role_type_vocab = Vocab()
    role_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    role_type_vocab.add_counter(Counter(role_type_list))
    role_type_vocab.save(role_type_vocab_file)
    print(role_type_vocab)

    print('--------action_vocab---------------')
    action_vocab = Vocab()
    action_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    action_vocab.add_counter(Counter(actions_list))
    action_vocab.save(action_vocab_file)
    print(action_vocab)

    print('--------pos_vocab---------------')
    pos_vocab = Vocab()
    pos_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    pos_vocab.add_counter(Counter(pos_list))
    pos_vocab.save(pos_vocab_file)
    print(pos_vocab)

    print('--------dep_type_vocab---------------')
    dep_type_vocab = Vocab()
    dep_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    dep_type_vocab.add_counter(Counter(dep_types_list))
    dep_type_vocab.save(dep_type_vocab_file)
    print(dep_type_vocab)


def construct_instance(inst_list, token_vocab, char_vocab, prd_type_vocab, role_type_vocab,
                       action_vocab, pos_vocab, dep_type_vocab, is_train=True):
    word_num = 0
    processed_inst_list = []
    for inst in inst_list:
        words = inst['nlp_words']
        prds = inst['Predicates']  # idx, prds_type
        pairs = inst['Pair']  # arg_id, prd_id, role
        pos = inst['nlp_pos']
        deps = inst['nlp_deps']
        dep_labels = inst['nlp_dep_types']

        if is_train and len(prds) == 0: continue

        words_processed = []
        word_indices = []
        char_indices = []
        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            words_processed.append(word)
            word_idx = token_vocab.get_index(word)
            word_indices.append(word_idx)
            char_indices.append([char_vocab.get_index(c) for c in word])

        del inst['Sent']
        inst['words'] = words_processed
        inst['word_indices'] = word_indices
        inst['char_indices'] = char_indices

        inst['pos_indices'] = [pos_vocab.get_index(p) for p in pos]
        inst['dep_label_indices'] = [dep_type_vocab.get_index(p) for p in dep_labels]

        inst['prd_indices'] = [tri[0] for tri in prds]
        inst['arg_indices'] = [pair[0] for pair in pairs]

        inst['prd_type_indices'] = [[tri[0], prd_type_vocab.get_index(tri[1].lower())] for tri in prds]

        inst['role_type_indices'] = [[ent[0], ent[1], role_type_vocab.get_index(ent[2])] for ent in pairs]

        actions = Actions.make_oracle(words, pairs, prds)
        inst['actions'] = actions
        inst['action_indices'] = [action_vocab.get_index(act) for act in actions]

        inst['sent_range'] = list(range(word_num, word_num + len(words)))
        word_num += len(words)
        processed_inst_list.append(inst)

    return processed_inst_list


def pickle_data():
    token_vocab = Vocab.load(token_vocab_file)
    char_vocab = Vocab.load(char_vocab_file)
    prd_type_vocab = Vocab.load(prd_type_vocab_file)
    role_type_vocab = Vocab.load(role_type_vocab_file)
    action_vocab = Vocab.load(action_vocab_file)
    pos_vocab = Vocab.load(pos_vocab_file)
    dep_type_vocab = Vocab.load(dep_type_vocab_file)

    processed_train = construct_instance(train_list, token_vocab, char_vocab, prd_type_vocab, role_type_vocab,
                                         action_vocab, pos_vocab, dep_type_vocab)
    processed_dev = construct_instance(dev_list, token_vocab, char_vocab, prd_type_vocab, role_type_vocab,
                                       action_vocab, pos_vocab, dep_type_vocab, False)
    processed_test = construct_instance(test_list, token_vocab, char_vocab, prd_type_vocab, role_type_vocab,
                                        action_vocab, pos_vocab, dep_type_vocab, False)

    print('Saving pickle to ', inst_pl_file)
    print('Saving sent size Train: %d, Dev: %d, Test:%d' % (
        len(processed_train), len(processed_dev), len(processed_test)))
    save_pickle(inst_pl_file, [processed_train, processed_dev, processed_test, token_vocab, char_vocab, prd_type_vocab,
                               role_type_vocab, action_vocab, pos_vocab, dep_type_vocab])

    scale = np.sqrt(3.0 / embedd_dim)
    vocab_dict = token_vocab.tok2idx
    table = np.empty([len(vocab_dict), embedd_dim], dtype=np.float32)
    oov = 0
    for word, index in vocab_dict.items():
        if word in embedd_dict:
            embedding = embedd_dict[word]
        elif word.lower() in embedd_dict:
            embedding = embedd_dict[word.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding

    np.save(vec_npy_file, table)
    print('pretrained embedding oov: %d' % oov)
    print()


if __name__ == '__main__':
    read_embedding()
    build_vocab()
    pickle_data()
