'''
Prepares the train/dev/test sets for the PennTreeBank POS tagged data.
Taken the entore dataset and does a 70:10:20 split
'''

import os
import sys
import glob
from collections import defaultdict

input_dir = sys.argv[1]
output_dir = sys.argv[2]

train_sections = ["%02d" % x for x in range(19)]
dev_sections = ["%02d" % x for x in range(19, 22)]
test_sections = ["%02d" % x for x in range(22,25)]

def ParseData(subfolder_range, subdir_name):
    subdir = os.path.join(output_dir, subdir_name)
    try:
        os.makedirs(subdir)
    except:
        pass

    for i in subfolder_range:
        dir_to_save = os.path.join(subdir, i)
        try:
            os.makedirs(dir_to_save)
        except:
            pass

        pattern = input_dir + i + "/*.pos"
        #print "pattern = ", pattern
        files = glob.glob(pattern)
        for file in files:
            filename = os.path.basename(file)
            out_file = filename.replace('.pos', '.txt')
            #print "outfile = ", out_file
            w = open(os.path.join(dir_to_save, out_file), 'w')
            f = open(file, 'rb')
            lines = f.readlines()
            temp = defaultdict(list)
            sent_num = 1
            next_line = ""
            i = 0
            while i < len(lines):
                if '[' in lines[i]:
                    next_line = lines[i]
                    j = i+1
                    while ('./.' not in next_line) and (j < len(lines)):
                        temp[sent_num].append(next_line)
                        next_line = lines[j]
                        j += 1
                    i = j
                    temp[sent_num].append(next_line)
                    sent_num += 1
                else:
                    i += 1

            lines = []
            for sent in temp:
                parts = [line.strip() for line in temp[sent]]
                parts = [line.strip('[') for line in parts]
                parts = [line.strip(']') for line in parts]
                lines.append(''.join(parts))
            lines = [line.lstrip() for line in lines]
            for line in lines:
                w.write(line + '\n')
            w.close()

# Create training data
print "Parsing train data"
ParseData(train_sections, "Train")
print "Parsing dev data"
ParseData(dev_sections, "Dev")
print "Parsing test data"
ParseData(test_sections, "Test")






