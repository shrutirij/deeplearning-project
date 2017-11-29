import glob
import re
from collections import defaultdict
import os

pos_folder = 'PennTreebank/pos_parsed/'
pos_files = glob.glob(pos_folder + '*/*')
ner_folder = 'PennTreebank/wsj/'

regex = re.compile("<[^<]+>")

for file_name in pos_files:
    ner_file = ner_folder + file_name.replace(pos_folder, '').replace('.txt', '')
    print ner_file
    try:
        if not os.path.exists('tagged_data/' + file_name.replace(pos_folder, '')):
            os.makedirs('tagged_data/' + file_name.replace(pos_folder, ''))

        with open(ner_file + '.name', 'r') as ner, open(file_name, 'r') as pos, open('tagged_data/' + file_name.replace(pos_folder, '').replace('.txt', '') + '.tagged', 'w') as out:
            lines = defaultdict(lambda: len(lines))
            pos_lines = []

            for line in pos:
                spl = line.strip().split()
                cur_line = " ".join([word.rsplit('/',1)[0] for word in spl])
                if cur_line not in lines:
                    lines[cur_line]              
                    pos_lines.append(spl)
            
            assert len(lines) == len(pos_lines)

            count = 0
            for line in ner:
                if '<DOC DOCNO' in line or '</DOC>' in line:
                    continue
                clean_line = regex.sub("", line.strip())
                if clean_line in lines:
                    out_line = ''
                    cur_pos = pos_lines[lines[clean_line]]
                    spl = line.strip().replace('<ENAMEX ', '').split()

                    assert len(cur_pos) == len(spl)

                    print spl

                    in_ne = False
                    cur_type = 'O'

                    for token, ner_token in zip(cur_pos, spl):
                        if in_ne:
                            out_line += token + '/I-' + cur_type + ' '
                            if '</ENAMEX>' in ner_token:
                                in_ne = False
                                cur_type = 'O'
                        elif 'TYPE=' in ner_token:
                            cur_type = ner_token.split('">')[0].replace('TYPE="', '')
                            in_ne = True
                            out_line += token + '/B-' + cur_type + ' '
                        else:
                            out_line += token + '/' + cur_type + ' '
                    
                    out.write(out_line.strip() + '\n')
                else:
                    print clean_line
                    exit()
    except IOError:
        raise
