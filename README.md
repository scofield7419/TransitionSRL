# Transition-based SRL


Code for the AAAI 2021 work [**End-to-end Semantic Role Labeling with Neural Transition-based Model**](https://ojs.aaai.org/index.php/AAAI/article/view/17515)


## Requirement

* numpy
* dynet
* random

## Step 1: Data Processing

#### Donwload the SRL dataset:
* [CoNLL09 (English)](https://catalog.ldc.upenn.edu/LDC2012T03)
* [UPB](https://github.com/System-T/UniversalPropositions)


#### Reformat the data, as exemplified in _data/sample.json_


#### Run _preprocess.py_

## Step 2: Training & Evaluating


#### Word embedding:

* [Fasttext](https://fasttext.cc/)
* [ELMo](https://allennlp.org/elmo)
* [BERT (base-cased-version)](https://github.com/google-research/bert)
* [XLNet (base-version)](https://github.com/zihangdai/xlnet)


#### Run _train.py_


***

```
@inproceedings{fei-end-SRL,
  author    = {Hao Fei and
               Meishan Zhang and
               Bobo Li and
               Donghong Ji},
  title     = {End-to-end Semantic Role Labeling with Neural Transition-based Model},
  booktitle = {Proceedings of the AAAI},
  pages     = {12803--12811},
  year      = {2021},
}

