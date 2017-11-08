from __future__ import print_function
import dynet as dy
import codecs
from collections import defaultdict
import mlp

BATCH_SIZE = 32

class BiLSTMTagger(object):
    def __init__(self, embed_size, char_hidden_size, word_hidden_size, training_file):
        self.training_data, self.char_vocab, self.tag_vocab = self.read(training_file)
        self.model = dy.Model()

        self.char_embeds = self.model.add_lookup_parameters((len(self.char_vocab), embed_size))
        self.char_lstm = dy.BiRNNBuilder(1, embed_size, char_hidden_size, self.model, dy.LSTMBuilder)
        self.word_lstm = dy.BiRNNBuilder(1, char_hidden_size, word_hidden_size, self.model, dy.LSTMBuilder)

    def read(self, filename):
        train_sents = []
        char_vocab = defaultdict(lambda: len(char_vocab))
        tags = defaultdict(lambda: len(tags))

        with codecs.open(filename, 'r', 'utf8') as fh:
            for line in fh:
                line = line.strip().split()
                sent = [tuple(x.rsplit("/",1)) for x in line]
                sent = [([char_vocab[c] for c in word], tags[tag]) for word, tag in sent]
                train_sents.append(sent)
        return train_sents, char_vocab, tags
    
    def calculate_loss(self, sents):
        dy.renew_cg()
        losses = []

        for sent in sents:
            word_reps = [self.char_lstm.transduce([self.char_embeds[c] for c in word])[-1] for word, tag in sent]
            contexts = self.word_lstm.transduce(word_reps)
        pass

    def train(self, epochs):
        trainer = dy.AdamTrainer(self.model)

        for ep in range(epochs):
            print('Epoch: %d' % ep)
            ep_loss = 0
            for i in range(0, len(self.training_data), BATCH_SIZE):
                cur_size = min(BATCH_SIZE, len(self.training_data)-i)
                loss = self.calculate_loss(self.training_data[i:i+cur_size])
                ep_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
            print(ep_loss/len(self.training_data))

if __name__ == '__main__':
    b = BiLSTMTagger(2,2,2,'test.txt')
    b.train(1)
