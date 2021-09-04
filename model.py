from io_utils import read_yaml
joint_config = read_yaml('joint_config.yaml')
data_config = read_yaml('data_config.yaml')

import numpy as np
import random
import dynet_config

random.seed(joint_config['random_seed'])
np.random.seed(joint_config['random_seed'])
# dynet_config.set(autobatch=1, mem='4096', random_seed=joint_config['random_seed'])
dynet_config.set(autobatch=1, mem='2700', random_seed=joint_config['random_seed'])

import dynet as dy
import nn
import ops
from dy_utils import ParamManager as pm
from dy_utils import AdamTrainer
from event_eval import EventEval
from io_utils import to_set, get_logger
from shift_reduce import ShiftReduce

logger = get_logger('transition', log_dir='log',log_name='trains.log')

sent_vec_dim = 0
if joint_config['use_sentence_vec']:
    train_sent_file = data_config['train_sent_file']
    test_sent_file = data_config['test_sent_file']
    dev_sent_file = data_config['dev_sent_file']
    train_sent_arr = np.load(train_sent_file)
    dev_sent_arr = np.load(dev_sent_file)
    test_sent_arr = np.load(test_sent_file)
    sent_vec_dim = train_sent_arr.shape[1]
    joint_config['sent_vec_dim'] = sent_vec_dim
    logger.info('train_sent_arr shape:%s'%str(train_sent_arr.shape))



class MainModel(object):

    def __init__(self, n_words, action_dict, prd_type_vocab, role_type_vocab, dep_type_vocab, pos_dict, pretrained_vec=None):
        
        pm.init_param_col()
        self.model = pm.global_collection()
        self.sent_model = dy.Model()
        self.optimizer = AdamTrainer(alpha=joint_config['init_lr'])
        self.optimizer.set_clip_threshold(joint_config['grad_clipping'])


        if not joint_config['use_pretrain_embed'] and not joint_config['use_sentence_vec']:
            raise AttributeError('At least one of use_pretrain_embed and use_sentence_vec should set to True')


        if joint_config['use_pretrain_embed']:
            self.word_embed = nn.Embedding(n_words,
                                       joint_config['word_embed_dim'],
                                       init_weight=pretrained_vec,
                                       trainable=joint_config['pretrain_embed_tune'])

        if joint_config['use_char_rnn']:
            self.char_embed = nn.Embedding(joint_config['n_chars'],
                                           joint_config['char_embed_dim'],
                                           trainable=True)
            self.char_rnn = nn.MultiLayerLSTM(joint_config['char_embed_dim'], joint_config['char_rnn_dim'], bidirectional=True)

        if joint_config['use_pos']:
            self.pos_embed = nn.Embedding(len(pos_dict), joint_config['pos_embed_dim'], trainable=True)

        if joint_config['use_dep']:
            self.dep_embed = nn.Embedding(len(dep_type_vocab), joint_config['dep_embed_dim'], trainable=True)
            self.dep_treeLSTM = nn.TreeLSTMEncoder(joint_config['dep_embed_dim'], joint_config['treeLSTM_dim'])

        if joint_config['random_word_embed']:
            print('Random_word_embed: True')
            self.word_embed_tune = nn.Embedding(n_words, joint_config['word_embed_dim'], trainable=True)
            self.word_linear = nn.Linear(joint_config['word_embed_dim'] * 2, joint_config['word_embed_dim'], activation='relu')

        if joint_config['use_sentence_vec']:
            print('Use_sentence_vec (BERT): True')
            self.train_sent_embed = nn.Embedding(train_sent_arr.shape[0], sent_vec_dim,
                                                 init_weight=train_sent_arr,
                                                 trainable=False,
                                                 name='trainSentEmbed',
                                                 model=self.sent_model)

            self.dev_sent_embed = nn.Embedding(dev_sent_arr.shape[0], sent_vec_dim,
                                                 init_weight=dev_sent_arr,
                                                 trainable=False,
                                                 name='devSentEmbed')

            self.test_sent_embed = nn.Embedding(test_sent_arr.shape[0], sent_vec_dim,
                                                 init_weight=test_sent_arr,
                                                 trainable=False,
                                                name='testSentEmbed',
                                                model=self.sent_model)


            if joint_config['sent_vec_project'] > 0:
                print('Sentence_vec project to', joint_config['sent_vec_project'])
                self.sent_project = nn.Linear(sent_vec_dim, joint_config['sent_vec_project'],
                                         activation=joint_config['sent_vec_project_activation'])


        rnn_input = 0  
        if joint_config['use_pretrain_embed']:
            rnn_input += joint_config['word_embed_dim']
            print('use_pretrain_embed:', joint_config['use_pretrain_embed'])

        if joint_config['use_sentence_vec'] and not joint_config['cat_sent_after_rnn']:
            rnn_input += sent_vec_dim
            print('use_sentence_vec:', joint_config['use_sentence_vec'])

        if joint_config['use_pos']:
            rnn_input += joint_config['pos_embed_dim']
            print('use_pos:', joint_config['use_pos'])

        if joint_config['use_dep']:
            rnn_input += joint_config['treeLSTM_dim']
            print('use_dep:', joint_config['use_dep'])

        if joint_config['use_char_rnn']:
            rnn_input += joint_config['char_rnn_dim'] * 2
            print('use_char_rnn:', joint_config['use_char_rnn'])


        if joint_config['use_rnn_encoder']:
            self.encoder = nn.MultiLayerLSTM(rnn_input, joint_config['rnn_dim'],
                                    n_layer=joint_config['encoder_layer'], bidirectional=True,
                                    dropout_x=joint_config['dp_state'], dropout_h=joint_config['dp_state_h'])


        self.encoder_output_dim = 0
        if joint_config['use_rnn_encoder']:
            self.encoder_output_dim += joint_config['rnn_dim'] * 2
        elif joint_config['use_pretrain_embed']:
            self.encoder_output_dim += joint_config['word_embed_dim']
            if joint_config['use_pos']:
                self.encoder_output_dim += joint_config['pos_embed_dim']
            if joint_config['use_dep']:
                self.encoder_output_dim += joint_config['treeLSTM_dim']


        if joint_config['cat_sent_after_rnn'] and joint_config['use_sentence_vec']:
            self.encoder_output_dim += sent_vec_dim


        self.encoder_output_dim = joint_config['encoder_project'] if joint_config['encoder_project'] > 0 else self.encoder_output_dim

        # shift reduce parser
        self.shift_reduce = ShiftReduce(joint_config, self.encoder_output_dim, action_dict,
                                        prd_type_vocab, role_type_vocab)


    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.populate(path)


    def get_word_embed(self, toks, pos_list, is_train=True):
        tok_emb = self.word_embed(toks)

        if joint_config['random_word_embed']:
            tok_emb_tune = self.word_embed_tune(toks)
            tok_emb = ops.cat_list(tok_emb , tok_emb_tune)
            tok_emb = self.word_linear(tok_emb)

        return tok_emb


    def get_sent_embed(self, range, is_train=True, mtype='train'):
        if mtype=='train':
            sent_emb = self.train_sent_embed(range)
        elif mtype =='dev':
            sent_emb = self.dev_sent_embed(range)
        else:
            sent_emb = self.test_sent_embed(range)

        if joint_config['sent_vec_project'] > 0:
            sent_emb = self.sent_project(sent_emb)

        return sent_emb, sent_emb


    def get_char_embed(self, chars, is_train=True):
        self.char_rnn.init_sequence(not is_train)
        encoder_char = []
        for word_char in chars:
            char_embed = self.char_embed(word_char)
            _, (last_h, last_c) = self.char_rnn.last_step(char_embed)
            encoder_char.append(last_h)

        return encoder_char


    def input_embed(self, toks, chars, pos_list,dep_list,dep_head, range, is_train=True,
                    return_last_h = False, return_sent_vec=False, mtype='train'):
        tok_emb = None
        last_h = None
        output_elmo_emb = None
        if joint_config['use_rnn_encoder']:
            self.encoder.init_sequence(not is_train)

        if joint_config['use_pretrain_embed']:
            tok_emb = self.get_word_embed(toks, pos_list,  is_train)
            if joint_config['cat_sent_after_rnn'] and joint_config['use_rnn_encoder']:
                tok_emb, (last_h, last_c) = self.encoder.last_step(tok_emb)

        if joint_config['use_sentence_vec']:
            sent_vec, output_elmo_emb = self.get_sent_embed(range, is_train, mtype=mtype)
            if tok_emb is not None:
                tok_emb = ops.cat_list(tok_emb, sent_vec)
            else:
                tok_emb = sent_vec

        if joint_config['use_char_rnn']:
            char_embed = self.get_char_embed(chars, is_train)
            tok_emb = ops.cat_list(tok_emb, char_embed)

        if joint_config['use_pos']:
            pos_emb = self.pos_embed(pos_list)
            tok_emb = ops.cat_list(tok_emb, pos_emb)


        if joint_config['use_dep']:
            dep_emb = self.dep_embed(dep_list)
            dep_rep = self.dep_treeLSTM(dep_emb, dep_head)
            tok_emb = ops.cat_list(tok_emb, dep_rep)


        if is_train:
            tok_emb = ops.dropout_list(tok_emb, joint_config['dp_emb'])

        if not joint_config['cat_sent_after_rnn'] and joint_config['use_rnn_encoder']:
            tok_emb, (last_h, last_c) = self.encoder.last_step(tok_emb)

        if is_train:
            tok_emb = ops.dropout_list(tok_emb, joint_config['dp_rnn'])


        if return_sent_vec:
            return tok_emb, output_elmo_emb
        else:
            return tok_emb


    def iter_batch_data(self, batch_data):
        batch_size = len(batch_data['tokens_ids'])

        for i in range(batch_size):
            one_data = {name:val[i]  for name, val in batch_data.items()}
            yield one_data


    def decay_lr(self, rate):
        self.optimizer.learning_rate *= rate

    def get_lr(self):
        return self.optimizer.learning_rate

    def set_lr(self, lr):
        self.optimizer.learning_rate = lr

    def update(self):
        try:
            self.optimizer.update()
        except RuntimeError:
            pass


    def regularization_loss(self, coef=0.001):
        losses = [dy.l2_norm(p) ** 2 for p in self.model.parameters_list() if  p.name().startswith('/linearW')]
        return (coef / 2) * dy.esum(losses)


    def __call__(self, toks, chars, act_ids, acts, prds, roles, sent_range, pos_list, dep_list, dep_head ):
        
        context_emb = self.input_embed(toks, chars, pos_list, dep_list,dep_head, sent_range, is_train=True, return_sent_vec=False)

        sent_vec = None
        log_prob_list, loss_roles, pred_prds, pred_args, pred_pairs, pred_acts = \
                self.shift_reduce(toks,
                            context_emb, sent_vec, oracle_actions=act_ids, oracle_action_strs=acts,
                            is_train=True, prds=prds, roles=roles)
        act_loss = -dy.esum(log_prob_list)
        role_loss = dy.esum(loss_roles) if loss_roles else 0
        loss = act_loss + 0.8 * role_loss
        
        return loss


    def decode(self, toks, chars, act_ids, acts, prds, roles, sent_range, pos_list, dep_list,dep_head,  mtype='dev'):
        dy.renew_cg()

        context_emb = self.input_embed(toks, chars, pos_list,dep_list, dep_head, sent_range, is_train=False, return_sent_vec=False, mtype=mtype)


        sent_vec = None
        log_prob_list, loss_roles, pred_prds, pred_args, pred_pairs, pred_acts = \
                                            self.shift_reduce(toks,
                                                context_emb, sent_vec, oracle_actions=act_ids, oracle_action_strs=acts,
                                                              is_train=False, prds=prds, roles=roles)

        return 0, pred_prds, pred_args, pred_pairs