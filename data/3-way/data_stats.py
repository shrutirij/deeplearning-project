import numpy as np
import math
import codecs

# Train Data
f = codecs.open('train_data.txt', 'r', 'utf8')
train_en = 0
train_es = 0
train_other = 0

for line in f:
    cur_en = line.strip().count('/en')
    cur_es = line.strip().count('/es')
    cur_other = line.strip().count('/other')
    train_en += cur_en
    train_es += cur_es
    train_other += cur_other
f.close()

#Dev data
f = codecs.open('dev_data.txt', 'r', 'utf8')
dev_en = 0
dev_es = 0
dev_other = 0

for line in f:
    cur_en = line.strip().count('/en')
    cur_es = line.strip().count('/es')
    cur_other = line.strip().count('/other')
    dev_en += cur_en
    dev_es += cur_es
    dev_other += cur_other
f.close()

# Test
f = codecs.open('test_data.txt', 'r', 'utf8')
test_en = 0
test_es = 0
test_other = 0

for line in f:
    cur_en = line.strip().count('/en')
    cur_es = line.strip().count('/es')
    cur_other = line.strip().count('/other')
    test_en += cur_en
    test_es += cur_es
    test_other += cur_other
f.close()

print "Training english words = ", train_en
print "Training spanish words = ", train_es
print "Training other words = ", train_other
print "Validation english words = ", dev_en
print "Validation spanish words = ", dev_es
print "Validation other words = ", dev_other
print "Test english words = ", test_en
print "Test spanish words = ", test_es
print "Test other words = ", test_other











