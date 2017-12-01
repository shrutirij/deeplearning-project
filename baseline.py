from __future__ import print_function
import dynet as dy
import codecs
from collections import defaultdict
import mlp
import random
import argparse
import glob

BATCH_SIZE = 32
DROPOUT = 0.5
UNK = '$unk'    # Do we still need UNK for char lstm?

DATA = './data/PennTreebank/pos_parsed/'

class BiLSTMTagger(object):
    def __init__(self, embed_size, char_hidden_size, word_hidden_size, mlp_layer_size, training_set, dev_set, test_set, task_file_type):
        self.training_data, self.char_vocab, self.tag_vocab = self.read(training_set, task_file_type)
        self.tag_lookup = dict((v,k) for k,v in self.tag_vocab.iteritems())
        self.dev_data = self.read_unk(dev_set, task_file_type)
        self.test_data = self.read_unk(test_set, task_file_type)
        
        self.model = dy.Model()

        self.char_embeds = self.model.add_lookup_parameters((len(self.char_vocab), embed_size))
        self.char_lstm_fwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, self.model)
        self.char_lstm_bwd = dy.LSTMBuilder(1, embed_size, char_hidden_size/2, self.model)
        self.word_lstm = dy.BiRNNBuilder(1, embed_size, word_hidden_size, self.model, dy.LSTMBuilder)
        self.feedforward = mlp.MLP(self.model, 2, [(word_hidden_size,mlp_layer_size), (mlp_layer_size,len(self.tag_vocab))], 'tanh', 0.0)
        
        if DROPOUT > 0.:
            self.char_lstm_fwd.set_dropout(DROPOUT)
            self.char_lstm_bwd.set_dropout(DROPOUT)
            self.word_lstm.set_dropout(DROPOUT)

    def read(self, file_range, file_type):
        train_sents = []
        char_vocab = defaultdict(lambda: len(char_vocab))
        tags = defaultdict(lambda: len(tags))

        tag_count = defaultdict(lambda: 0)

        for i in range(int(file_range[0]), int(file_range[-1])+1):
            file_names = glob.glob(DATA + str(i).zfill(2) + '/*' + file_type)
            
            for filename in file_names:
                with open(filename, 'r') as fh:
                    for line in fh:
                        line = line.strip().split()
                        sent = [tuple(x.rsplit("/",1)) for x in line]
                        for word, tag in sent:
                            tag_count[tag] += 1  
                        #sent = [(word_vocab[word], tags[tag]) for word, tag in sent]
                        sent = [([char_vocab[c] for c in word], tags[tag]) for word, tag in sent]
                        train_sents.append(sent)                  
        print(len(char_vocab))
        return train_sents, char_vocab, tags
    
    def read_unk(self, file_range, file_type):
        sents = []
        tag_count = defaultdict(lambda: 0)
        total = 0

        for i in range(int(file_range[0]), int(file_range[-1])+1):
            file_names = glob.glob(DATA + str(i).zfill(2) + '/*' + file_type)
            
            for filename in file_names:
                with codecs.open(filename, 'r', 'utf8') as f:
                    for line in f:
                        line = line.strip().split()
                        sent = [tuple(x.rsplit("/",1)) for x in line]
                        for word, tag in sent:
                            tag_count[tag] += 1
                            total += 1
                        #sent = [(self.word_to_int(word), self.tag_vocab[tag]) for word, tag in sent]
                        sent = [([self.char_to_int(c) for c in word], self.tag_vocab[tag]) for word, tag in sent]
                        sents.append(sent)
        return sents

    def char_to_int(self, char):
        if char in self.char_vocab:
            return self.char_vocab[char]
        else:
            return self.char_vocab[UNK]

    def lookup_tag(self, tag_id):
        return self.tag_lookup[tag_id]

    def get_output(self, sents):
        dy.renew_cg()
        tagged_sents = []

        for sent in sents:
            cur_tags = []
            char_embeds = [[self.char_embeds[c] for c in word] for word, tag in sent]
            word_reps = [dy.concatenate([self.char_lstm_fwd.initial_state().transduce(emb)[-1], self.char_lstm_bwd.initial_state().transduce(reversed(emb))[-1]]) for emb in char_embeds]
            contexts = self.word_lstm.transduce(word_reps)
            for context in contexts:
                probs = dy.softmax(self.feedforward.forward(context)).vec_value()
                pred_tag = probs.index(max(probs))
                cur_tags.append(pred_tag)
            tagged_sents.append(cur_tags)
        return tagged_sents
    
    def calculate_loss(self, sents):
        dy.renew_cg()
        losses = []

        for sent in sents:
            char_embeds = [[self.char_embeds[c] for c in word] for word, tag in sent]
            word_reps = [dy.concatenate([self.char_lstm_fwd.initial_state().transduce(emb)[-1], self.char_lstm_bwd.initial_state().transduce(reversed(emb))[-1]]) for emb in char_embeds]
            contexts = self.word_lstm.transduce(word_reps)
            probs = dy.concatenate_to_batch([self.feedforward.forward(context) for context in contexts])

            if DROPOUT > 0.:
                probs = dy.dropout_batch(probs, DROPOUT)

            cur_losses = dy.pickneglogsoftmax_batch(probs, [tag for word,tag in sent])
            losses.append(dy.sum_batches(cur_losses)) 

        return dy.esum(losses)

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
                if num_batches % 50 == 0:
                    f_out.write('Validation loss: %f\n' % self.get_loss(self.dev_data))
                    f_out.write('Validation accuracy: %f\n' % self.get_accuracy(self.dev_data))
                    f_out.write('Test accuracy: %f\n' % self.get_accuracy(self.test_data))
                cur_size = min(BATCH_SIZE, len(self.training_data)-i)
                loss = self.calculate_loss(self.training_data[i:i+cur_size])            
                ep_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
                num_batches += 1
            f_out.write('Training loss: %f\n' % ep_loss)
            f_out.write('Training accuracy: %f\n' % self.get_accuracy(self.training_data))
            f_out.write('\n')

    def get_loss(self, sents):
        val_loss = 0
        for i in range(0, len(sents), BATCH_SIZE):
            cur_size = min(BATCH_SIZE, len(sents)-i)
            loss = self.calculate_loss(sents[i:i+cur_size])
            val_loss += loss.scalar_value()       
        return val_loss

    def get_accuracy(self, sents):
        outputs = []
        for i in range(0, len(sents), BATCH_SIZE):
            cur_size = min(BATCH_SIZE, len(sents)-i)
            outputs += self.get_output(sents[i:i+cur_size])
        corr_tags = 0.0
        total_tags = 0

        assert len(sents) == len(outputs)
        
        for sent, output in zip(sents, outputs):
            for (chars, tag), pred_tag in zip(sent, output):
                if tag == pred_tag:
                    corr_tags += 1
                total_tags += 1
        return corr_tags/total_tags                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='baseline')
    parser.add_argument('--char', default='256')
    parser.add_argument('--embed', default='256')
    parser.add_argument('--word', default='256')
    parser.add_argument('--mlp', default='512')
    parser.add_argument('--train', default='0,18')
    parser.add_argument('--dev', default='19,21')
    parser.add_argument('--test', default='22,24')
    parser.add_argument('--filetype', default='.txt')
    parser.add_argument('--trainer', default='sgd')
    args, unknown = parser.parse_known_args()

    f_out = open(args.output + '.log', 'w', 0)
    f_out.write(str(args) + '\n')

    tagger_model = BiLSTMTagger(int(args.embed), int(args.char), int(args.word), int(args.mlp), args.train.split(','), args.dev.split(','),args.test.split(','), args.filetype)
    tagger_model.train(1000)
    
    f_out.close()
