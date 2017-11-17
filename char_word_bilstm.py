from __future__ import print_function
import dynet as dy
import codecs
from collections import defaultdict
import mlp
import random

BATCH_SIZE = 32
DROPOUT = 0.5
UNK = '$unk'    # Do we still need UNK for char lstm?

f_out = open('char_word_full.log', 'w', 0)

class BiLSTMTagger(object):
    def __init__(self, embed_size, char_hidden_size, word_hidden_size, mlp_layer_size, training_file, dev_file, test_file):
        self.training_data, self.char_vocab, self.tag_vocab = self.read(training_file)
        self.tag_lookup = dict((v,k) for k,v in self.tag_vocab.iteritems())
        self.dev_data = self.read_unk(dev_file)
        self.test_data = self.read_unk(test_file)

        self.model = dy.Model()

        self.char_embeds = self.model.add_lookup_parameters((len(self.char_vocab), embed_size))
        self.char_lstm = dy.BiRNNBuilder(1, embed_size, char_hidden_size, self.model, dy.LSTMBuilder)
        self.word_lstm = dy.BiRNNBuilder(1, embed_size, word_hidden_size, self.model, dy.LSTMBuilder)
        self.feedforward = mlp.MLP(self.model, 2, [(word_hidden_size,mlp_layer_size), (mlp_layer_size,len(self.tag_vocab))], 'tanh', 0.0)
        
        if DROPOUT > 0.:
            self.char_lstm.set_dropout(DROPOUT)
            self.word_lstm.set_dropout(DROPOUT)

    def read(self, filename):
        train_sents = []
        char_vocab = defaultdict(lambda: len(char_vocab))
        tags = defaultdict(lambda: len(tags))

        with codecs.open(filename, 'r', 'utf8') as fh:
            for line in fh:
                line = line.strip().split()
                sent = [tuple(x.rsplit("/",1)) for x in line]
                #sent = [(word_vocab[word], tags[tag]) for word, tag in sent]
                sent = [([char_vocab[c] for c in word], tags[tag]) for word, tag in sent]
                train_sents.append(sent)        
        return train_sents, char_vocab, tags
    
    def read_unk(self, filename):
        sents = []

        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                line = line.strip().split()
                sent = [tuple(x.rsplit("/",1)) for x in line]
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
        tagged_sents = []

        for sent in sents:
            dy.renew_cg()
            cur_tags = []
            #contexts = self.word_lstm.transduce([self.word_embeds[word] for word, tag in sent])
            word_reps = [self.char_lstm.transduce([self.char_embeds[c] for c in word])[-1] for word, tag in sent]
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
        words = 0

        for sent in sents:
            #contexts = self.word_lstm.transduce([self.word_embeds[word] for word, tag in sent])
            word_reps = [self.char_lstm.transduce([self.char_embeds[c] for c in word])[-1] for word, tag in sent]
            contexts = self.word_lstm.transduce(word_reps)
            probs = dy.concatenate_to_batch([self.feedforward.forward(context) for context in contexts])

            if DROPOUT > 0.:
                probs = dy.dropout_batch(probs, DROPOUT)

            cur_losses = dy.pickneglogsoftmax_batch(probs, [tag for word,tag in sent])
            losses.append(dy.sum_batches(cur_losses)) 

        return dy.esum(losses)

    def train(self, epochs):
        trainer = dy.SimpleSGDTrainer(self.model)

        for ep in range(epochs):
            f_out.write('Epoch: %d\n' % ep)
            ep_loss = 0
            num_batches = 0
            random.shuffle(self.training_data)
            for i in range(0, len(self.training_data), BATCH_SIZE):
                cur_size = min(BATCH_SIZE, len(self.training_data)-i)
                loss = self.calculate_loss(self.training_data[i:i+cur_size])            
                ep_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
                num_batches += 1

                if num_batches % 50 == 0:
                    f_out.write('Validation loss: %f\n' % self.get_loss(self.dev_data))
                    f_out.write('Validation accuracy: %f\n' % self.get_accuracy(self.dev_data))
                    f_out.write('Test accuracy: %f\n' % self.get_accuracy(self.test_data))
            f_out.write('Training loss: %f\n' % ep_loss)
            f_out.write('Training accuracy: %f\n' % self.get_accuracy(self.training_data))
            f_out.write('\n')

    def get_loss(self, sents):
        val_loss = 0
        val_words = 0
        loss = self.calculate_loss(sents)          
        return loss.scalar_value()

    def get_accuracy(self, sents):
        outputs = self.get_output(sents)
        corr_tags = 0.0
        total_tags = 0
        for sent, output in zip(sents, outputs):
            for (chars, tag), pred_tag in zip(sent, output):
                if tag == pred_tag:
                    corr_tags += 1
                total_tags += 1
        return corr_tags/total_tags                


if __name__ == '__main__':
    tagger_model = BiLSTMTagger(256, 256, 512, 256, './data/3-way/train_data.txt','./data/3-way/dev_data.txt','./data/3-way/test_data.txt')
    tagger_model.train(1000)
    # tagger_model = BiLSTMTagger(16, 16, 16, 8, './data/small_data.txt','./data/small_data.txt','./data/small_data.txt')
    # tagger_model.train(100)
    