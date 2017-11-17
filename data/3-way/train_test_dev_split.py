import numpy as np
import math

f = open('combined_data_no_dup.txt', 'rb')
lines = f.readlines()
np.random.shuffle(lines)
total = len(lines)
train_idx = int(math.ceil(0.8 * total))
idx = int(math.ceil(0.1 * total))
dev_idx = train_idx+idx

# Train Data
w = open('train_all_classes_data.txt', 'wb')
for line in lines[0:train_idx]:
    w.write(line)
w.close()

# Dev Data
w = open('dev_all_classes_data.txt', 'wb')
for line in lines[train_idx:dev_idx]:
    w.write(line)
w.close()

# Test Data
w = open('test_all_classes_data.txt', 'wb')
for line in lines[dev_idx:]:
    w.write(line)
w.close()



