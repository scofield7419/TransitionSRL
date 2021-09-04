# -*- coding: utf-8 -*-

class Actions(object):

    predicate_gen = 'PRD-GEN'
    no_predicate = 'NO-PRD'

    left_arc = 'LEFT-ARC'
    right_arc = 'RIGHT-ARC'
    no_left_arc = 'NO-ARC-LEFT'
    no_right_arc = 'NO-ARC-RIGHT'

    o_delete = 'O-DELETE'

    shift = 'SHIFT'

    def __init__(self, action_dict, role_type_dict, prd_type_dict, with_copy_shift=True):
        self.predicate_gen_id = action_dict[Actions.predicate_gen]
        self.no_predicate_id = action_dict[Actions.no_predicate]
        self.no_left_arc_id = action_dict[Actions.no_left_arc]
        self.no_right_arc_id = action_dict[Actions.no_right_arc]
        self.shift_id = action_dict[Actions.shift]

        self.pair_gen_left_group = set()
        self.pair_gen_right_group = set()

        self.act_to_ent_id = {}

        self.act_id_to_str = {v:k for k, v in action_dict.items()}

        for name, id in action_dict.items():
            if name.startswith(Actions.left_arc):
                self.pair_gen_left_group.add(id)

            elif name.startswith(Actions.right_arc):
                self.pair_gen_right_group.add(id)


    def get_act_ids_by_args(self, arg_type_ids):
        acts = []
        for arg_id in arg_type_ids:
            acts.append(self.arg_to_act_id[arg_id])

        return acts


    def get_pair_gen_left_list(self):
        return list(self.pair_gen_left_group)

    def get_pair_gen_right_list(self):
        return list(self.pair_gen_right_group)

    def get_pair_gen_list(self):
        return list(self.pair_gen_right_group)+list(self.pair_gen_left_group)


    def to_act_str(self, act_id):
        return self.act_id_to_str[act_id]


    # action check

    def is_prd_gen(self, act_id):
        return self.predicate_gen_id == act_id

    def is_no_prd(self, act_id):
        return self.no_predicate_id == act_id

    def is_shift(self, act_id):
        return self.shift_id == act_id

    def is_left_arc(self, act_id):
        return act_id in self.pair_gen_left_group

    def is_right_arc(self, act_id):
        return act_id in self.pair_gen_right_group

    def is_no_right_arc(self, act_id):
        return self.no_right_arc_id == act_id

    def is_no_left_arc(self, act_id):
        return self.no_left_arc_id == act_id



    @staticmethod
    def make_oracle(tokens, pairs, prds):

        prd_dic = {prd[0]:prd[1] for prd in prds}

        pair_dic = {(pair[0], pair[1]): pair for pair in pairs}

        actions = []

        # GEN entities and triggers
        prd_actions = {} # start_idx : actions list

        sent_length = len(tokens)

        for prd_idx in range(sent_length):

            if prd_idx in prd_dic.keys():
                prd_sense = prd_dic[prd_idx]
                actions.append(Actions.predicate_gen) 

                left_max_inc = prd_idx
                right_max_inc = sent_length-prd_idx-1

                for i in range(1, max(left_max_inc, right_max_inc)+1):
                    # left
                    if prd_idx-i >=0:
                        if (prd_idx-i, prd_idx) in pair_dic.keys():
                            actions.append(Actions.left_arc)
                        else:
                            actions.append(Actions.no_left_arc)

                    # right
                    if prd_idx+i < sent_length:
                        if (prd_idx+i, prd_idx) in pair_dic.keys():
                            actions.append(Actions.right_arc)
                        else:
                            actions.append(Actions.no_right_arc)

                actions.append(Actions.shift)

            else:
                actions.append(Actions.no_predicate)

        return actions  







