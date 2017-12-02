'''
Classifying the POS tags and NER for the sentences from PennTreebank using fixed neurons in a single model.
'''

from __future__ import print_function
import dynet as dy
import codecs
from collections import defaultdict
import mlp
import random
import glob

BATCH_SIZE = 32
DROPOUT = 0.5
UNK = '$unk'
DATA = './data/PennTreebank/pos_parsed/'

class BiLSTMTagger(object):
    def __init__(self, embed_size, char_hidden_size, word_hidden_size, mlp_layer_size, training_set, dev_set, test_set):
        # Setting the constants
        self.FIXED = int(word_hidden_size/4)
        self.training_data, self.char_vocab, self.tag_vocab, self.ner_vocab = self.read(training_set)
        self.tag_lookup = dict((v,k) for k,v in self.tag_vocab.iteritems())
        self.ner_lookup = dict((v,k) for k,v in self.ner_vocab.iteritems())
        self.dev_data = self.read_unk(dev_set)
        self.test_data = self.read_unk(test_set)

        self.model = dy.Model()

        self.char_embeds = self.model.add_lookup_parameters((len(self.char_vocab), embed_size))
        self.char_lstm_fwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, self.model)
        self.char_lstm_bwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, self.model)
        self.word_lstm = dy.BiRNNBuilder(1, embed_size, word_hidden_size, self.model, dy.LSTMBuilder)
        self.feedforward_pos = mlp.MLP(self.model, 2, [(self.FIXED,mlp_layer_size), (mlp_layer_size,len(self.tag_vocab))], 'tanh', 0.0)
        self.feedforward_ner = mlp.MLP(self.model, 2, [(self.FIXED,mlp_layer_size), (mlp_layer_size,len(self.ner_vocab))], 'tanh', 0.0)
        
        if DROPOUT > 0.:
            self.char_lstm_fwd.set_dropout(DROPOUT)
            self.char_lstm_bwd.set_dropout(DROPOUT)
            self.word_lstm.set_dropout(DROPOUT)

    def read(self, file_range):
        train_sents = []
        char_vocab = defaultdict(lambda: len(char_vocab))
        tags = defaultdict(lambda: len(tags))
        ners = defaultdict(lambda: len(ners))
        for i in range(int(file_range[0]), int(file_range[-1])+1):
            file_names = glob.glob(DATA + str(i).zfill(2) + '/*.tagged')
            
            for filename in file_names:
                with codecs.open(filename, 'r', 'utf8') as fh:
                    for line in fh:
                        line = line.strip().split()
                        line = [item for item in line if '/-NONE-' not in item]
                        sent = [tuple(x.split("/")) for x in line]
                        sent = [([char_vocab[c] for c in word], tags[tag], ners[ner]) for word, tag, ner in sent]
                        train_sents.append(sent)        
        return train_sents, char_vocab, tags, ners

    def read_unk(self, file_range):
        sents = []
        for i in range(int(file_range[0]), int(file_range[-1])+1):
            file_names = glob.glob(DATA + str(i).zfill(2) + '/*.tagged')
            
            for filename in file_names:
                with codecs.open(filename, 'r', 'utf8') as f:
                    for line in f:
                        line = line.strip().split()
                        line = [item for item in line if '/-NONE-' not in item]
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
        dy.renew_cg()
        tagged_sents = []

        for sent in sents:
            cur_preds = []
            char_embeds = [[self.char_embeds[c] for c in word] for word,tag,ner in sent]
            word_reps = [dy.concatenate([self.char_lstm_fwd.initial_state().transduce(emb)[-1], self.char_lstm_bwd.initial_state().transduce(reversed(emb))[-1]]) for emb in char_embeds]
            contexts = self.word_lstm.transduce(word_reps)
            for context in contexts:
                # Predict POS tags
                pos_probs = dy.softmax(self.feedforward_pos.forward(dy.concatenate([context[:self.FIXED], context[-self.FIXED:]]))).vec_value()
                pred_tag = pos_probs.index(max(pos_probs))

                # Predict NER
                ner_probs = dy.softmax(self.feedforward_ner.forward(context[self.FIXED:-self.FIXED])).vec_value()
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

            pos_probs = dy.concatenate_to_batch([self.feedforward_pos.forward(dy.concatenate([context[:self.FIXED], context[-self.FIXED:]])) for context in contexts])
            ner_probs = dy.concatenate_to_batch([self.feedforward_ner.forward(context[self.FIXED:-self.FIXED]) for context in contexts])
            
            if DROPOUT > 0.:
                pos_probs = dy.dropout_batch(pos_probs, DROPOUT)
                ner_probs = dy.dropout_batch(ner_probs, DROPOUT)

            cur_pos_losses = dy.pickneglogsoftmax_batch(pos_probs, [tag for word,tag,ner in sent])
            pos_losses.append(dy.sum_batches(cur_pos_losses)) 

            cur_ner_losses = dy.pickneglogsoftmax_batch(ner_probs, [ner for word,tag,ner in sent])
            ner_losses.append(dy.sum_batches(cur_ner_losses))

        return dy.esum(pos_losses) + dy.esum(ner_losses)

    def train(self, epochs):
        if args.trainer == 'sgd':
            trainer = dy.SimpleSGDTrainer(self.model)
        else:
            trainer = dy.AdamTrainer(self.model)

        for ep in range(epochs):
            f_out.write('Epoch: %d\n' % ep)
            ep_loss = 0
            num_batches = 0
            random.shuffle(self.training_data)
            for i in range(0, len(self.training_data), BATCH_SIZE):
                if num_batches % 160 == 0:
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
        for i in range(0, len(sents), BATCH_SIZE):
            cur_size = min(BATCH_SIZE, len(sents)-i)
            loss = self.calculate_loss(sents[i:i+cur_size])
            val_loss += loss.scalar_value()       
        return val_loss

    # Returns both POS and NER accuracy
    def get_accuracy(self, sents):      
        outputs = []
        for i in range(0, len(sents), BATCH_SIZE):
            cur_size = min(BATCH_SIZE, len(sents)-i)
            outputs += self.get_output(sents[i:i+cur_size])
        corr_tags = 0.0
        corr_ner = 0.0
        total = 0

        assert len(sents) == len(outputs)
        
        for sent, output in zip(sents, outputs):
            for (chars, tag, ner), (pred_tag, pred_ner) in zip(sent, output):
                if tag == pred_tag:
                    corr_tags += 1
                if ner == pred_ner:
                    corr_ner += 1
                total += 1
        return corr_tags/total, corr_ner/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='baseline')
    parser.add_argument('--char', default='256')
    parser.add_argument('--embed', default='256')
    parser.add_argument('--word', default='512')
    parser.add_argument('--mlp', default='512')
    parser.add_argument('--train', default='0,18')
    parser.add_argument('--dev', default='19,21')
    parser.add_argument('--test', default='22,24')
    parser.add_argument('--trainer', default='sgd')
    args, unknown = parser.parse_known_args()

    f_out = open(args.output + '.log', 'w', 0)
    f_out.write(str(args) + '\n')

    tagger_model = BiLSTMTagger(int(args.embed), int(args.char), int(args.word), int(args.mlp), args.train.split(','), args.dev.split(','),args.test.split(','))
    tagger_model.train(100)

    f_out.close()
