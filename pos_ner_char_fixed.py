'''
Classifying the POS tags and NER for the sentences from PennTreebank using fixed neurons in a single model.
'''

from __future__ import print_function
import dynet as dy
import codecs
from collections import defaultdict
import mlp
import random

BATCH_SIZE = 32
DROPOUT = 0.5
UNK = '$unk'

f_out = open('joint_fixedneuron.log', 'w', 0)

class BiLSTMTagger(object):
    def __init__(self, embed_size, char_hidden_size, word_hidden_size, mlp_layer_size, training_file, dev_file, test_file):
        # Setting the constants
        self.FIXED = int(word_hidden_size/2)
        self.POS_IDX = random.sample(range(0, word_hidden_size), self.FIXED)
        self.NER_IDX = list(set(range(0, word_hidden_size)) - set(self.POS_IDX))
        self.training_data, self.char_vocab, self.tag_vocab, self.ner_vocab = self.read(training_file)
        self.tag_lookup = dict((v,k) for k,v in self.tag_vocab.iteritems())
        self.ner_lookup = dict((v,k) for k,v in self.ner_vocab.iteritems())
        self.dev_data = self.read_unk(dev_file)
        self.test_data = self.read_unk(test_file)

        self.model = dy.Model()

        self.char_embeds = self.model.add_lookup_parameters((len(self.char_vocab), embed_size))
        #self.char_lstm = dy.BiRNNBuilder(1, embed_size, char_hidden_size, self.model, dy.LSTMBuilder)
        self.char_lstm_fwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, self.model)
        self.char_lstm_bwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, self.model)
        self.word_lstm = dy.BiRNNBuilder(1, embed_size, word_hidden_size, self.model, dy.LSTMBuilder)
        self.feedforward_pos = mlp.MLP(self.model, 2, [(self.FIXED,mlp_layer_size), (mlp_layer_size,len(self.tag_vocab))], 'tanh', 0.0)
        self.feedforward_ner = mlp.MLP(self.model, 2, [(self.FIXED,mlp_layer_size), (mlp_layer_size,len(self.ner_vocab))], 'tanh', 0.0)
        
        if DROPOUT > 0.:
            #self.char_lstm.set_dropout(DROPOUT)
            self.char_lstm_fwd.set_dropout(DROPOUT)
            self.char_lstm_bwd.set_dropout(DROPOUT)
            self.word_lstm.set_dropout(DROPOUT)

    def read(self, filename):
        train_sents = []
        char_vocab = defaultdict(lambda: len(char_vocab))
        tags = defaultdict(lambda: len(tags))
        ners = defaultdict(lambda: len(ners))
        with codecs.open(filename, 'r', 'utf8') as fh:
            for line in fh:
                line = line.strip().split()
                sent = [tuple(x.split("/")) for x in line]
                sent = [([char_vocab[c] for c in word], tags[tag], ners[ner]) for word, tag, ner in sent]
                train_sents.append(sent)        
        return train_sents, char_vocab, tags, ners

    def read_unk(self, filename):
        sents = []
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                line = line.strip().split()
                sent = [tuple(x.split("/")) for x in line]
                sent = [([self.char_to_int(c) for c in word], self.tag_vocab[tag], self.ner_vocab[ner]) for word, tag, ner in sent]
                sents.append(sent)
        return sents

    def char_to_int(self, char):
        if char in self.char_vocab:
            return self.char_vocab[char]
        else:
            return self.char_vocab[UNK]

    def get_output(self, sents):
        tagged_sents = []

        for sent in sents:
            dy.renew_cg()
            cur_preds = []
            char_embeds = [[self.char_embeds[c] for c in word] for word,tag,ner in sent]
            word_reps = [dy.concatenate([self.char_lstm_fwd.initial_state().transduce(emb)[-1], self.char_lstm_bwd.initial_state().transduce(reversed(emb))[-1]]) for emb in char_embeds]
            contexts = self.word_lstm.transduce(word_reps)
            for context in contexts:
                # Predict POS tags
                pos_temp = context[self.POS_IDX[0]]
                for i in range(1, len(self.POS_IDX)):
                    pos_temp = dy.concatenate([pos_temp, context[self.POS_IDX[i]]])
                fwd_val = self.feedforward_pos.forward(pos_temp)
                pos_probs = dy.softmax(fwd_val).vec_value()
                pred_tag = pos_probs.index(max(pos_probs))

                # Predict NER
                ner_temp = context[self.NER_IDX[0]]
                for i in range(1, len(self.NER_IDX)):
                    ner_temp = dy.concatenate([ner_temp, context[self.NER_IDX[i]]])
                fwd_val = self.feedforward_ner.forward(ner_temp)
                ner_probs = dy.softmax(fwd_val).vec_value()
                pred_ner = ner_probs.index(max(ner_probs))
                cur_preds.append((pred_tag, pred_ner))

            tagged_sents.append(cur_preds)
        return tagged_sents

    def calculate_loss(self, sents):
        dy.renew_cg()
        pos_losses = []
        ner_losses = []
        words = 0

        for sent in sents:
            char_embeds = [[self.char_embeds[c] for c in word] for word,tag,ner in sent]
            word_reps = [dy.concatenate([self.char_lstm_fwd.initial_state().transduce(emb)[-1], self.char_lstm_bwd.initial_state().transduce(reversed(emb))[-1]]) for emb in char_embeds]
            contexts = self.word_lstm.transduce(word_reps)
            pos_probs_list = []
            ner_probs_list = []
            for context in contexts:
                # Compute losses for POS
                pos_temp = context[self.POS_IDX[0]]
                for i in range(1, len(self.POS_IDX)):
                    pos_temp = dy.concatenate([pos_temp, context[self.POS_IDX[i]]])
                pos_prob = self.feedforward_pos.forward(pos_temp)
                pos_probs_list.append(pos_prob)

                # Compute losses for NER
                ner_temp = context[self.NER_IDX[0]]
                for i in range(1, len(self.NER_IDX)):
                    ner_temp = dy.concatenate([ner_temp, context[self.NER_IDX[i]]])
                ner_prob = self.feedforward_pos.forward(ner_temp)
                ner_probs_list.append(ner_prob)

            pos_probs = dy.concatenate_to_batch(pos_probs_list)
            ner_probs = dy.concatenate_to_batch(ner_probs_list)

            if DROPOUT > 0.:
                pos_probs = dy.dropout_batch(pos_probs, DROPOUT)
                ner_probs = dy.dropout_batch(ner_probs, DROPOUT)

            cur_pos_losses = dy.pickneglogsoftmax_batch(pos_probs, [tag for word,tag,ner in sent])
            pos_losses.append(dy.sum_batches(cur_pos_losses)) 

            cur_ner_losses = dy.pickneglogsoftmax_batch(ner_probs, [ner for word,tag,ner in sent])
            ner_losses.append(dy.sum_batches(cur_ner_losses))

        return dy.esum(pos_losses) + dy.esum(ner_losses)

    def train(self, epochs):
        trainer = dy.SimpleSGDTrainer(self.model)

        for ep in range(epochs):
            f_out.write('Epoch: %d\n' % ep)
            ep_loss = 0
            num_batches = 0
            random.shuffle(self.training_data)
            for i in range(0, len(self.training_data), BATCH_SIZE):
                if num_batches % 50 == 0:
                    f_out.write('Validation loss: %f\n' % self.get_loss(self.dev_data))
                    v_pos_acc, v_ner_acc = self.get_accuracy(self.dev_data)
                    f_out.write('Validation accuracy for POS: %f\n' % v_pos_acc)
                    f_out.write('Validation accuracy for NER: %f\n' % v_ner_acc)

                    t_pos_acc, t_ner_acc = self.get_accuracy(self.test_data)
                    f_out.write('Test accuracy for POS: %f\n' % t_pos_acc)
                    f_out.write('Test accuracy for NER: %f\n' % t_ner_acc)
                cur_size = min(BATCH_SIZE, len(self.training_data)-i)
                loss = self.calculate_loss(self.training_data[i:i+cur_size])            
                ep_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
                num_batches += 1

            train_pos_acc, train_ner_acc = self.get_accuracy(self.training_data)
            f_out.write('Training loss: %f\n' % ep_loss)
            f_out.write('Training accuracy for POS: %f\n' % train_pos_acc)
            f_out.write('Training accuracy for NER: %f\n' % train_ner_acc)
            f_out.write('\n')

    def get_loss(self, sents):
        val_loss = 0
        val_words = 0
        loss = self.calculate_loss(sents)          
        return loss.scalar_value()

    # Returns both POS and NER accuracy
    def get_accuracy(self, sents):
        outputs = self.get_output(sents)
        corr_tags = 0.0
        corr_ner = 0.0
        total = 0
        for sent, output in zip(sents, outputs):
            for (chars, tag, ner), (pred_tag, pred_ner) in zip(sent, output):
                if tag == pred_tag:
                    corr_tags += 1
                if ner == pred_ner:
                    corr_ner += 1
                total += 1
        return corr_tags/total, corr_ner/total


if __name__ == '__main__':
    tagger_model = BiLSTMTagger(8, 8, 16, 8, './data/joint_small.txt','./data/joint_small.txt','./data/joint_small.txt')
    tagger_model.train(100)





