from collections import defaultdict
import dynet as dy
import numpy as np
import random
import datetime
import codecs

train_file = "train_clean.txt"
dev_file = "dev_clean.txt"

# train_file = "one_tagged.txt"
# dev_file = "one_tagged.txt"

W_INPUT = 1024
W_HIDDEN = 1024
BATCH_SIZE = 32
unk = '$UNK'
ss = '<s>'

class Vocab:
    def __init__(self, words):
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.get_dict(words)
        self.id_lookup = {i:w for w,i in self.vocab.iteritems()}

    def get_dict(self, words):
        wc = defaultdict(lambda: 0)

        for w in words:
            wc[w] += 1

        for w,c in wc.iteritems():
            if c >= 0:
                self.vocab[w]

    def sents_to_int(self, sents):
        corpus = []
        for sent in sents:
            corpus.append([self.vocab[word] for word in sent])
        return corpus

    def word_to_int(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab[unk]

    def size(self):
        return len(self.vocab.keys())

def read(fname):
    train_sents = []
    with codecs.open(fname, 'r', 'utf8') as fh:
        for line in fh:
            line = line.strip().lower()
            chars = []

            if len(line) < 2:
                continue

            for c in line:
                chars.append(c)

            train_sents.append(chars)

    return train_sents

def make_batches(sents):
    sents.sort(key=lambda x: -len(x))
    batches = []
    cur_batch = []
    cur_len = len(sents[0])

    for s in sents:
        s_len = len(s)
        if len(cur_batch) >= BATCH_SIZE:  # or s_len != cur_len:
            batches.append(cur_batch)
            cur_batch = []
            cur_len = s_len

        cur_batch.append([wv.word_to_int(x) for x in s])

    if len(cur_batch) > 0:
        batches.append(cur_batch)

    return batches

def batch_loss(batch):
    dy.renew_cg()
    W_l2r = dy.parameter(params['W_l2r'])
    b_l2r = dy.parameter(params['b_l2r'])
    w_lookup = params['w_lookup']

    tot_words = 0
    wids = []
    masks = []
    for i in range(len(batch[0])):
        wids.append([(sent[i] if len(sent)>i else wv.word_to_int(ss)) for sent in batch])
        mask = [(1 if len(sent)>i else 0) for sent in batch]
        masks.append(mask)
        tot_words += sum(mask)

    s = w_l2r.initial_state()

    # start the rnn by inputting "<s>"
    init_ids = [wv.word_to_int(ss)] * len(batch)
    s = s.add_input(dy.lookup_batch(w_lookup,init_ids))

    # feed word vectors into the RNN and predict the next word
    losses = []
    for wid, mask in zip(wids, masks):
        # calculate the softmax and loss
        score = W_l2r * s.output() + b_l2r
        loss = dy.pickneglogsoftmax_batch(score, wid)
        # mask the loss if at least one sentence is shorter
        if mask[-1] != 1:
            mask_expr = dy.inputVector(mask)
            mask_expr = dy.reshape(mask_expr, (1,), len(batch))
            loss = loss * mask_expr
        losses.append(loss)
        # update the state of the RNN
        wemb = dy.lookup_batch(w_lookup, wid)
        s = s.add_input(wemb)

    return dy.sum_batches(dy.esum(losses)), tot_words


train = list(read(train_file))
dev = list(read(dev_file))
words = []

for sent in train:
    for w in sent:
        words.append(w)

wv = Vocab(words)
wv.vocab[unk]
wv.vocab[ss]

model = dy.Model()
trainer = dy.AdamTrainer(model)

params = {}
params['w_lookup'] = model.add_lookup_parameters((wv.size(), W_INPUT))
params['W_l2r'] = model.add_parameters((wv.size(), W_HIDDEN))
params['b_l2r'] = model.add_parameters(wv.size())

w_l2r = dy.LSTMBuilder(2, W_INPUT, W_HIDDEN, model)
w_l2r.set_dropout(0.5)

batches = make_batches(train)
dev_batches = make_batches(dev)

print len(batches)
print len(dev_batches)
print wv.size()

with codecs.open('c_w_vocab', 'w', 'utf8') as out:
    for k,v in wv.vocab.iteritems():
        out.write(k + '\t' + str(v) + '\n')

to_check = int(len(batches)/10.0)

prev_best = 100000000.0

for iteration in xrange(1000):
    random.shuffle(batches)
    cum_loss = 0
    num_tagged = 0
    count_b = 0
    print "Epoch: " + str(iteration)
    for batch in batches:
        count_b += 1
        if count_b % 50 == 0:
            print count_b
        loss_exp, words = batch_loss(batch)
        cum_loss += loss_exp.scalar_value()
        num_tagged += words
        loss_exp.backward()
        trainer.update()

        if count_b == to_check:
            count_b = 0
            dev_loss = 0
            dev_words = 0
            for b in dev_batches:
                loss_exp, words = batch_loss(b)
                dev_loss += loss_exp.scalar_value()
                dev_words += words
            print dev_loss/dev_words

            if dev_loss < prev_best:
                model.save('char_lstm', [w_l2r, params['w_lookup'], params['W_l2r'], params['b_l2r']])
                prev_best = dev_loss

    print datetime.datetime.now()
    print cum_loss/num_tagged
