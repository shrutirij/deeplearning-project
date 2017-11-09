import codecs
import random

en = {}
es = {}
cs = {}
other = {}

count = 0

with codecs.open('train_data.txt', 'r', 'utf8') as f:
    for line in f:
        if '/en' in line and '/es' in line:
            cs[line.strip()] = True
        elif '/en' in line:
            en[line.strip()] = True
        elif '/es' in line:
            es[line.strip()] = True
        else:
            other[line.strip()] = True
        count +=1

    print count

dev_count = 1020

with codecs.open('dev_data.txt', 'w', 'utf8') as out:
    for cur_lang in [en, es, cs, other]:
        cur_count = dev_count*len(cur_lang)/count
        
        while cur_count > 0:
            to_write = random.choice(cur_lang.keys())
            if cur_lang[to_write]:
                out.write(to_write + '\n')
                cur_lang[to_write] = False
                cur_count -= 1

with codecs.open('train_data_pruned.txt', 'w', 'utf8') as out:
    for cur_lang in [en, es, cs, other]:
        for k, v in cur_lang.iteritems():
            if v:
                out.write(k + '\n')
