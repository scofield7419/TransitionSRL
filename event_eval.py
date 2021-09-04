
from vocab import Vocab

def to_set(input):
    out_set = set()
    out_type_set = set()
    for x in input:
        out_set.add(tuple(x[:-1]))
        out_type_set.add(tuple(x))

    return out_set, out_type_set

class EventEval(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_prd = 0.
        self.correct_prd_with_type = 0.
        self.num_pre_prd = 0.
        self.num_gold_prd = 0.

        self.num_pre_pair = 0.
        self.num_gold_pair = 0.
        self.correct_pair = 0.
        self.correct_pair_with_role = 0.


    def update(self, pred_prds, gold_prds,
               pred_args, gold_args,
               pred_pairs, gold_pairs, eval_arg=True, words=None):
        

        pred_prds=set(pred_prds)
        gold_prds=set(gold_prds)

        self.num_pre_prd += len(pred_prds)
        self.num_gold_prd += len(gold_prds)

        self.correct_prd += len(pred_prds & gold_prds)
        self.correct_prd_with_type += len(pred_prds & gold_prds)

        self.num_pre_pair += len(pred_pairs)
        self.num_gold_pair += len(gold_pairs)

        gold_pairs_ = list(set([(pair[1], pair[2]) for pair in gold_pairs]))
        pred_pairs_ = list(set([(pair[1], pair[2]) for pair in pred_pairs]))
        for i in gold_pairs_:
            for j in pred_pairs_:
                if i[0] == j[0] and i[1] == j[1]:
                    self.correct_pair_with_role += 1


    def report(self):
        p_prd = self.correct_prd / (self.num_pre_prd + 1e-18)
        r_prd = self.correct_prd / (self.num_gold_prd + 1e-18)
        f_prd = 2 * p_prd * r_prd / (p_prd + r_prd + 1e-18)

        p_pair = self.correct_pair / (self.num_pre_pair + 1e-18)
        r_pair = self.correct_pair / (self.num_gold_pair + 1e-18)
        f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-18)

        p_pair_role = self.correct_pair_with_role / (self.num_pre_pair + 1e-18)
        r_pair_role = self.correct_pair_with_role / (self.num_gold_pair + 1e-18)
        f_pair_role = 2 * p_pair_role * r_pair_role / (p_pair_role + r_pair_role + 1e-18)


        return (p_prd, r_prd, f_prd), (p_pair, r_pair, f_pair), (p_pair_role, r_pair_role, f_pair_role)

    def get_coref_ent(self, g_ent_typed):
        ent_ref_dict = {}
        for ent1 in g_ent_typed:
            start1, end1, ent_type1, ent_ref1 = ent1
            coref_ents = []
            ent_ref_dict[(start1, end1)] = coref_ents
            for ent2 in g_ent_typed:
                start2, end2, ent_type2, ent_ref2 = ent2
                if ent_ref1 == ent_ref2:
                    coref_ents.append((start2, end2))
        return ent_ref_dict

    def split_prob(self, pred_args):
        sp_args, probs = [], []
        for arg in pred_args:
            sp_args.append(arg[:-1])
            probs.append(arg[-1])
        return sp_args, probs
