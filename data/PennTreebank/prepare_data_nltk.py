
import os
import sys
from nltk.corpus import ptb


file_range = ["%02d" % x for x in range(25)]
mrg_path = 'treebank_3/parsed/mrg/WSJ'
output_dir = 'pos_parsed_nltk/'

for subdir in file_range:
    mrg_subdir = os.path.join(mrg_path, subdir)
    out_subdir = os.path.join(output_dir, subdir)
    
    try:
        os.makedirs(out_subdir)
    except:
        pass

    for filename in os.listdir(mrg_subdir):
        if '.MRG' not in filename:
            continue

        nltk_ptb_path = 'WSJ/' + subdir + '/' + filename
        sents = ptb.tagged_sents(nltk_ptb_path)
        w = open(os.path.join(out_subdir, filename), 'w')

        for sent in sents:
            line = ""
            for word,tag in sent:
                line = line + word + '/' + tag + ' '
            line = line.strip()
            w.write(line + '\n')

        w.close()





