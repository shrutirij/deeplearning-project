'''
Make predictions using trained models
'''

# embed_size, char_hidden_size, word_hidden_size, mlp_layer_size, training_set, dev_set, test_set)

from __future__ import print_function
import dynet as dy
import codecs
from collections import defaultdict
import mlp
import random
import numpy as np
import argparse
import glob
import sys

model_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
embed_size = 64
char_hidden_size = 64
word_hidden_size = 64
mlp_layer_size = 128
DATA = './data/PennTreebank/pos_parsed/'
char_vocab = defaultdict(lambda: len(char_vocab))
tags = defaultdict(lambda: len(tags))
ners = defaultdict(lambda: len(ners))

def build_vocab(file_range):
    for i in range(int(file_range[0]), int(file_range[-1])+1):
        file_names = glob.glob(DATA + str(i).zfill(2) + '/*.tagged')
        
        for filename in file_names:
            with codecs.open(filename, 'r', 'utf8') as fh:
                for line in fh:
                    line = line.strip().split()
                    line = [item for item in line if '/-NONE-' not in item]
                    sent = [tuple(x.split("/")) for x in line]
                    sent = [([char_vocab[c] for c in word], tags[tag], ners[ner]) for word, tag, ner in sent]
    return char_vocab, tags, ners


def read(filename):
    train_sents = []
    with codecs.open(filename, 'r', 'utf8') as fh:
        for line in fh:
            line = line.strip().split()
            line = [item for item in line if '/-NONE-' not in item]
            sent = [tuple(x.split("/")) for x in line]
            sent = [([char_vocab[c] for c in word], tags[tag], ners[ner]) for word, tag, ner in sent]
            train_sents.append(sent)
    return train_sents

model = dy.ParameterCollection()
char_embeds = model.add_lookup_parameters((len(char_vocab), embed_size))
char_lstm_fwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, model)
char_lstm_bwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, model)
word_lstm = dy.BiRNNBuilder(1, embed_size, word_hidden_size, model, dy.LSTMBuilder)
feedforward_pos = mlp.MLP(model, 2, [(word_hidden_size,mlp_layer_size), (mlp_layer_size,len(tags))], 'tanh', 0.0)
feedforward_ner = mlp.MLP(model, 2, [(word_hidden_size,mlp_layer_size), (mlp_layer_size,len(ners))], 'tanh', 0.0)

# Define the gating parameters
w_p = model.add_parameters((embed_size, word_hidden_size))
u_p = model.add_parameters((word_hidden_size, word_hidden_size))
b_p = model.add_parameters((word_hidden_size))
w_n = model.add_parameters((embed_size, word_hidden_size))
u_n = model.add_parameters((word_hidden_size, word_hidden_size))
b_n = model.add_parameters((word_hidden_size))

# Load the model
model.populate(model_file)

def get_output(sents):
    dy.renew_cg()
    w_pos = dy.parameter(w_p)
    u_pos = dy.parameter(u_p)
    b_pos = dy.parameter(b_p)
    w_ner = dy.parameter(w_n)
    u_ner = dy.parameter(u_n)
    b_ner = dy.parameter(b_n)
    tagged_sents = []

    for sent in sents:
        cur_preds = []
        char_embeds = [[char_embeds[c] for c in word] for word,tag,ner in sent]
        word_reps = [dy.concatenate([char_lstm_fwd.initial_state().transduce(emb)[-1], char_lstm_bwd.initial_state().transduce(reversed(emb))[-1]]) for emb in char_embeds]
        contexts = word_lstm.transduce(word_reps)
        h_init = dy.inputTensor(np.zeros((contexts[0].dim()[0])))

        for t in range(len(word_reps)):
            x_t = word_reps[t]
            h_t = contexts[t]
            if t-1 >= 0:
                h_t_m_1 = contexts[t-1]
            else:
                h_t_m_1 = h_init

            z_t_pos = dy.logistic(dy.transpose(w_pos) * x_t + u_pos * h_t_m_1 + b_pos)
            z_t_ner = dy.logistic(dy.transpose(w_ner) * x_t + u_ner * h_t_m_1 + b_ner)
            h_t_pos = dy.cmult(h_t, z_t_pos)
            h_t_ner = dy.cmult(h_t, z_t_ner)

            # Predict POS tags
            fwd_val = feedforward_pos.forward(h_t_pos)
            pos_probs = dy.softmax(fwd_val).vec_value()
            pred_tag = pos_probs.index(max(pos_probs))

            #Predict NER
            fwd_val = feedforward_ner.forward(h_t_ner)
            ner_probs = dy.softmax(fwd_val).vec_value()
            pred_ner = ner_probs.index(max(ner_probs))
            cur_preds.append((pred_tag, pred_ner))

        tagged_sents.append(cur_preds)
    return tagged_sents

build_vocab((0,18))
test_sents = read(test_file)
pred_tags = get_output(sents)
f = open(output_file, 'wb', '0')

for sent, output in zip(test_sents, pred_tags):
    pos_out = []
    ner_out = []
    for (chars, tag, ner), (pred_tag, pred_ner) in zip(sent, output):
        pos_out.append(pred_tag, tag)
        ner_out.append(pred_ner, ner)
    f.write("POS (predicted, correct) = " + str(pos_out) + '\n')
    f.write("NER (predicted, correct) = " + str(ner_out) + '\n')

f.close()










