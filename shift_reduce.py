import numpy as np
import dynet as dy
import nn
import ops
from dy_utils import ParamManager as pm
from actions import Actions
from vocab import Vocab
from event_constraints import EventConstraint
import io_utils


class RoleLabeler(object):

    def __init__(self, config, encoder_output_dim, action_dict, role_type_dict, prd_type_dict):
        self.config = config
        self.model = pm.global_collection()
        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim

        hidden_input_dim = lmda_dim * 5 + bi_rnn_dim * 2 + config['out_rnn_dim']

        self.hidden_arg = nn.Linear(hidden_input_dim+len(role_type_dict), config['output_hidden_dim'],
                                    activation='tanh')

        self.output_arg = nn.Linear(config['output_hidden_dim'], len(role_type_dict))

        hidden_input_dim_co = lmda_dim * 3 + bi_rnn_dim * 2 + config['out_rnn_dim']
        self.hidden_ent_corel = nn.Linear(hidden_input_dim_co, config['output_hidden_dim'],
                                          activation='tanh')
        self.output_ent_corel = nn.Linear(config['output_hidden_dim'], 2)

        self.position_embed = nn.Embedding(500, 20)

        attn_input = self.bi_rnn_dim * 1 + 20 * 2
        self.attn_hidden = nn.Linear(attn_input, 80, activation='tanh')
        self.attn_out = nn.Linear(80, 1)

        self.distrib_attn_hidden = nn.Linear(hidden_input_dim +len(role_type_dict), 80, activation='tanh')
        self.distrib_attn_out = nn.Linear(80, 1)
        self.empty_embedding = self.model.add_parameters((len(role_type_dict),), name='stackGuardEmb')


    def arg_prd_distributions_role_attn(self, inputs, arg_prd_distributions_role):
        inputs_ = [inputs for _ in range(len(arg_prd_distributions_role))]
        arg_prd_distributions_role = ops.cat(arg_prd_distributions_role, 1)
        inputs_ = ops.cat(inputs_, 1)
        att_input = dy.concatenate([arg_prd_distributions_role, inputs_], 0)
        hidden = self.distrib_attn_hidden(att_input)
        attn_out = self.distrib_attn_out(hidden)
        attn_prob = nn.softmax(attn_out, dim=1)
        rep = arg_prd_distributions_role * dy.transpose(attn_prob)
        return rep


    def forward(self, beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed, sigma_right_embed, alpha_right_embed,
                out_embed, hidden_mat, prd_idx, arg_idx, seq_len, last_h, gold_role_label, arg_prd_distributions_role):
        attn_rep = self.position_aware_attn(hidden_mat, last_h, prd_idx, prd_idx, arg_idx, arg_idx, seq_len)

        state_embed = ops.cat(
            [beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed, sigma_right_embed, alpha_right_embed,
             out_embed, attn_rep], dim=0)

        if len(arg_prd_distributions_role) > 1:
            rep = ops.cat([self.arg_prd_distributions_role_attn(state_embed, arg_prd_distributions_role), state_embed], 0)
        else:
            rep = ops.cat([self.empty_embedding, state_embed], 0)

        rep = dy.dropout(rep, 0.25)
        hidden = self.hidden_arg(rep)
        out = self.output_arg(hidden)

        loss = dy.pickneglogsoftmax(out, gold_role_label)
        return loss

    def decode(self, beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed, sigma_right_embed, alpha_right_embed,
               out_embed, hidden_mat, prd_idx, arg_idx, seq_len, last_h, arg_prd_distributions_role):
        attn_rep = self.position_aware_attn(hidden_mat, last_h, prd_idx, prd_idx, arg_idx, arg_idx, seq_len)


        state_embed = ops.cat(
            [beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed, sigma_right_embed, alpha_right_embed,
             out_embed, attn_rep], dim=0)

        if len(arg_prd_distributions_role) > 1:
            rep = ops.cat([self.arg_prd_distributions_role_attn(state_embed, arg_prd_distributions_role), state_embed], 0)
        else:
            rep = ops.cat([self.empty_embedding, state_embed], 0)

        hidden = self.hidden_arg(rep)
        out = self.output_arg(hidden)
        np_score = out.npvalue().flatten()
        return np.argmax(np_score)

    def position_aware_attn(self, hidden_mat, last_h, start1, ent1, start2, end2, seq_len):
        tri_pos_list = []
        ent_pos_list = []

        for i in range(seq_len):
            tri_pos_list.append(io_utils.relative_position(start1, ent1, i))
            ent_pos_list.append(io_utils.relative_position(start2, end2, i))

        tri_pos_emb = self.position_embed(tri_pos_list)
        tri_pos_mat = ops.cat(tri_pos_emb, 1)
        ent_pos_emb = self.position_embed(ent_pos_list)
        ent_pos_mat = ops.cat(ent_pos_emb, 1)

        att_input = ops.cat([hidden_mat, tri_pos_mat, ent_pos_mat], 0)

        hidden = self.attn_hidden(att_input)
        attn_out = self.attn_out(hidden)

        attn_prob = nn.softmax(attn_out, dim=1)

        rep = hidden_mat * dy.transpose(attn_prob)

        return rep


class ShiftReduce(object):

    def __init__(self, config, encoder_output_dim, action_dict, prd_type_dict, role_type_dict):

        self.config = config
        self.model = pm.global_collection()

        self.role_labeler = RoleLabeler(config, encoder_output_dim, action_dict, role_type_dict, prd_type_dict)
        self.role_null_id = role_type_dict[Vocab.NULL]

        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim

        dp_state = config['dp_state']
        dp_state_h = config['dp_state_h']

        # ------ states
        self.gamma_var = nn.LambdaVar(lmda_dim)

        self.sigma_left_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # stack leftford
        self.alpha_left_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # will be pushed back leftford

        self.sigma_right_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # stack rightford
        self.alpha_right_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # will be pushed back rightford

        self.actions_rnn = nn.StackLSTM(config['action_embed_dim'], config['action_rnn_dim'], dp_state, dp_state_h)
        self.out_rnn = nn.StackLSTM(bi_rnn_dim, config['out_rnn_dim'], dp_state, dp_state_h)

        # ------ states

        self.act_table = nn.Embedding(len(action_dict), config['action_embed_dim'])
        self.role_table = nn.Embedding(len(role_type_dict), config['role_embed_dim'])
        self.prd_table = nn.Embedding(1, config['prd_embed_dim'])

        self.act = Actions(action_dict, role_type_dict, prd_type_dict)

        hidden_input_dim = bi_rnn_dim + lmda_dim * 5 \
                           + config['action_rnn_dim'] + config['out_rnn_dim']

        self.hidden_linear = nn.Linear(hidden_input_dim, config['output_hidden_dim'], activation='tanh')
        self.output_linear = nn.ActionGenerator(config['output_hidden_dim'],  len(action_dict))

        prd_embed_dim = config['prd_embed_dim']

        prd_to_lmda_dim = bi_rnn_dim + prd_embed_dim  # + config['sent_vec_dim']
        self.prd_to_lmda = nn.Linear(prd_to_lmda_dim, lmda_dim, activation='tanh')

        self.arg_prd_distrib_prd = nn.Linear(lmda_dim * 2 + config['role_embed_dim'], len(action_dict), activation='softmax')
        self.arg_prd_distrib_role = nn.Linear(lmda_dim * 2 + config['role_embed_dim'], len(role_type_dict), activation='softmax')

        # beta
        self.empty_buffer_emb = self.model.add_parameters((bi_rnn_dim,), name='bufferGuardEmb')

    def __call__(self, toks, hidden_state_list, last_h, oracle_actions=None,
                 oracle_action_strs=None, is_train=True, prds=None, roles=None):
        prd_dic = dict()
        arg_dic = dict()
        gold_pair_dict = {(arg[0], arg[1]): arg[2] for arg in roles}  # (ent_start, tri_idx): role_type（非action_role, 纯role）

        frames = []

        hidden_mat = ops.cat(hidden_state_list, 1)
        seq_len = len(toks)

        # beta, queue, for candidate sentence.
        buffer = nn.Buffer(self.bi_rnn_dim, hidden_state_list)

        losses = []
        loss_roles = []
        pred_action_strs = []

        arg_prd_distributions_prd = []
        arg_prd_distributions_role = []


        self.sigma_left_rnn.init_sequence(not is_train)
        self.alpha_left_rnn.init_sequence(not is_train)
        self.sigma_right_rnn.init_sequence(not is_train)
        self.alpha_right_rnn.init_sequence(not is_train)
        self.actions_rnn.init_sequence(not is_train)
        self.out_rnn.init_sequence(not is_train)


        steps = 0
        while not (buffer.is_empty() and self.gamma_var.is_empty()):
            # 上一个action
            pre_action = None if self.actions_rnn.is_empty() else self.actions_rnn.last_idx()

            # based on parser state, get valid actions.
            # only a very small subset of actions are valid, as below.
            valid_actions = []

            if pre_action is not None and self.act.is_prd_gen(pre_action):
                valid_actions += [self.act.no_right_arc_id,self.act.no_left_arc_id]
                valid_actions += self.act.get_pair_gen_list()
            elif pre_action is not None and self.act.is_shift(pre_action):
                valid_actions += [self.act.predicate_gen_id, self.act.no_predicate_id]
            elif pre_action is not None and (self.act.is_left_arc(pre_action) or self.act.is_no_left_arc(pre_action)):
                valid_actions += [self.act.no_right_arc_id,self.act.no_left_arc_id,self.act.shift_id]
                valid_actions += self.act.get_pair_gen_list()
            elif pre_action is not None and (self.act.is_right_arc(pre_action) or self.act.is_no_right_arc(pre_action)):
                valid_actions += [self.act.no_left_arc_id,self.act.no_right_arc_id,self.act.shift_id]
                valid_actions += self.act.get_pair_gen_list()
            elif self.sigma_left_rnn.is_empty():
                valid_actions += [self.act.no_right_arc_id, self.act.no_left_arc_id, self.act.no_predicate_id, self.act.predicate_gen_id]
                valid_actions += self.act.get_pair_gen_right_list()

            elif self.sigma_right_rnn.is_empty():
                valid_actions += [self.act.no_right_arc_id, self.act.no_left_arc_id, self.act.no_predicate_id, self.act.predicate_gen_id]
                valid_actions += self.act.get_pair_gen_left_list()

            elif self.sigma_right_rnn.is_empty() and self.sigma_left_rnn.is_empty():
                valid_actions += [self.act.shift_id]

            else:
                valid_actions += [self.act.no_right_arc_id, self.act.no_left_arc_id, self.act.no_predicate_id, self.act.predicate_gen_id]


            # predicting action
            beta_embed = self.empty_buffer_emb if buffer.is_empty() else buffer.hidden_embedding()
            lmda_embed = self.gamma_var.embedding()
            sigma_left_embed = self.sigma_left_rnn.embedding()
            alpha_left_embed = self.alpha_left_rnn.embedding()
            sigma_right_embed = self.sigma_right_rnn.embedding()
            alpha_right_embed = self.alpha_right_rnn.embedding()
            action_embed = self.actions_rnn.embedding()
            out_embed = self.out_rnn.embedding()

            state_embed = ops.cat([beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed,
                 sigma_right_embed, alpha_right_embed, action_embed, out_embed], dim=0)
            if is_train:
                state_embed = dy.dropout(state_embed, self.config['dp_out'])

            hidden_rep = self.hidden_linear(state_embed)

            logits = self.output_linear(hidden_rep, arg_prd_distributions_prd)
            if is_train:
                log_probs = dy.log_softmax(logits, valid_actions)
            else:
                log_probs = dy.log_softmax(logits, valid_actions)

            if is_train:
                action = oracle_actions[steps]
                action_str = oracle_action_strs[steps]
                if action not in valid_actions:
                    raise RuntimeError('Action %s dose not in valid_actions, %s(pre) %s: [%s]' % (
                    action_str, self.act.to_act_str(pre_action),
                    self.act.to_act_str(action), ','.join(
                        [self.act.to_act_str(ac) for ac in valid_actions])))
                losses.append(dy.pick(log_probs, action))
            else:
                np_log_probs = log_probs.npvalue()
                act_prob = np.max(np_log_probs)
                action = np.argmax(np_log_probs)
                action_str = self.act.to_act_str(action)
                pred_action_strs.append(action_str)


            # if True:continue
            # update the parser state according to the action.
            if self.act.is_no_prd(action):
                hx, idx = buffer.pop()
                self.out_rnn.push(hx, idx)
                self.sigma_left_rnn.push(hx, idx)

                if not self.sigma_right_rnn.is_empty():
                    _, _ = self.sigma_right_rnn.pop()

            elif self.act.is_prd_gen(action):
                hx, idx = buffer.pop()

                ####### fulfill the sigma_right_rnn from prd_idx
                for x in range(len(toks) - 1, idx, -1):
                    self.sigma_right_rnn.push(self.sigma_right_rnn.empty_embedding, x)

                prd_embed = self.prd_table[0]
                prd_rep = self.prd_to_lmda(ops.cat([hx, prd_embed], dim=0))
                self.gamma_var.push(prd_rep, idx, nn.LambdaVar.is_prd)

                prd_dic[idx] = idx


            elif self.act.is_no_left_arc(action):
                if not self.sigma_left_rnn.is_empty():
                    sigma_last_embed_left, sigma_last_idx_left = self.sigma_left_rnn.pop()
                    self.alpha_left_rnn.push(sigma_last_embed_left, sigma_last_idx_left)

            elif self.act.is_no_right_arc(action):
                if not self.sigma_right_rnn.is_empty():
                    sigma_last_embed_right, sigma_last_idx_right = self.sigma_right_rnn.pop()
                    self.alpha_right_rnn.push(sigma_last_embed_right, sigma_last_idx_right)

            elif self.act.is_left_arc(action):
                lmda_idx = self.gamma_var.idx
                lmda_embed = self.gamma_var.embedding()
                if not self.sigma_left_rnn.is_empty():
                    sigma_last_embed, sigma_last_idx = self.sigma_left_rnn.pop()

                if is_train:
                    role_label = gold_pair_dict.get((sigma_last_idx, lmda_idx), self.role_null_id)
                    loss_role = self.role_labeler.forward(beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed,
                                                          sigma_right_embed, alpha_right_embed,
                                                          out_embed, hidden_mat, lmda_idx, sigma_last_idx, seq_len,
                                                          last_h,
                                                          role_label, arg_prd_distributions_role)
                    loss_roles.append(loss_role)

                else:
                    role_label = self.role_labeler.decode(beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed,
                                                          sigma_right_embed, alpha_right_embed,
                                                          out_embed, hidden_mat, lmda_idx, sigma_last_idx, seq_len,
                                                          last_h, arg_prd_distributions_role)

                self.alpha_left_rnn.push(sigma_last_embed, sigma_last_idx)

                frame = (sigma_last_idx, lmda_idx, role_label)
                frames.append(frame)

                arg_dic[sigma_last_idx] = sigma_last_idx


                # arg_prd_distributions
                role_emb = self.role_table[role_label]
                arg_prd_distributions_prd.append(self.arg_prd_distrib_prd(ops.cat([sigma_last_embed, lmda_embed, role_emb], dim=0)))
                arg_prd_distributions_role.append(self.arg_prd_distrib_role(ops.cat([sigma_last_embed, lmda_embed, role_emb], dim=0)))


            elif self.act.is_right_arc(action):
                lmda_idx = self.gamma_var.idx
                lmda_embed = self.gamma_var.embedding()
                if not self.sigma_right_rnn.is_empty():
                    sigma_last_embed, sigma_last_idx = self.sigma_right_rnn.pop()

                if is_train:
                    role_label = gold_pair_dict.get((sigma_last_idx, lmda_idx), self.role_null_id)
                    loss_role = self.role_labeler.forward(beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed,
                                                          sigma_right_embed, alpha_right_embed,
                                                          out_embed, hidden_mat, lmda_idx, sigma_last_idx, seq_len,
                                                          last_h,
                                                          role_label, arg_prd_distributions_role)
                    loss_roles.append(loss_role)

                else:
                    role_label = self.role_labeler.decode(beta_embed, lmda_embed, sigma_left_embed, alpha_left_embed,
                                                          sigma_right_embed, alpha_right_embed,
                                                          out_embed, hidden_mat, lmda_idx, sigma_last_idx, seq_len,
                                                          last_h, arg_prd_distributions_role)

                self.alpha_right_rnn.push(sigma_last_embed, sigma_last_idx)

                frame = (sigma_last_idx, lmda_idx, role_label)
                frames.append(frame)
                arg_dic[sigma_last_idx] = sigma_last_idx

                # arg_prd_distributions
                role_emb = self.role_table[role_label]
                arg_prd_distributions_prd.append(self.arg_prd_distrib_prd(ops.cat([sigma_last_embed, lmda_embed, role_emb], dim=0)))
                arg_prd_distributions_role.append(self.arg_prd_distrib_role(ops.cat([sigma_last_embed, lmda_embed, role_emb], dim=0)))

            elif self.act.is_shift(action):
                while not self.alpha_left_rnn.is_empty():
                    self.sigma_left_rnn.push(*self.alpha_left_rnn.pop())
                self.sigma_left_rnn.push(*self.gamma_var.pop())

                while not self.alpha_right_rnn.is_empty():
                    self.sigma_right_rnn.push(*self.alpha_right_rnn.pop())
                if not self.sigma_right_rnn.is_empty():
                    _, _ = self.sigma_right_rnn.pop()

            else:
                raise RuntimeError('Unknown action type:' + str(action))

            self.actions_rnn.push(self.act_table[action], action)

            steps += 1

        self.clear()

        return losses, loss_roles, set(prd_dic.values()), set(arg_dic.values()), set(frames), pred_action_strs

    def clear(self):
        self.sigma_left_rnn.clear()
        self.alpha_left_rnn.clear()
        self.sigma_right_rnn.clear()
        self.alpha_right_rnn.clear()
        self.actions_rnn.clear()
        self.gamma_var.clear()
        self.out_rnn.clear()

    def same(self, args):
        same_event_ents = set()
        for arg1 in args:
            ent_start1, ent_end1, tri_idx1, _ = arg1
            for arg2 in args:
                ent_start2, ent_end2, tri_idx2, _ = arg2
                if tri_idx1 == tri_idx2:
                    same_event_ents.add((ent_start1, ent_start2))
                    same_event_ents.add((ent_start2, ent_start1))

        return same_event_ents

    def get_valid_args(self, ent_type_id, tri_type_id):
        return self.cached_valid_args[(ent_type_id, tri_type_id)]
