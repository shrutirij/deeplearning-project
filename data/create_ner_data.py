import glob
import re
from collections import defaultdict
import os

pos_folder = 'PennTreebank/pos_parsed/'
pos_files = glob.glob(pos_folder + '*/*')
ner_folder = 'PennTreebank/wsj/'

regex = re.compile("<[^<]+>")
split_regex = re.compile('\s(?![^<>]*>)')

count_of_data = 0

types = {'LOC': True, 'GPE': True, 'PERSON': True, 'FAC': True, 'ORG': True}

for file_name in pos_files:
    ner_file = ner_folder + file_name.replace(pos_folder, '').replace('.txt', '')
    print ner_file
    print file_name
    try:
        if not os.path.exists('tagged_data/' + ner_file.split('/')[2]):
            os.makedirs('tagged_data/' + ner_file.split('/')[2])

        with open(ner_file + '.name', 'r') as ner, open(file_name, 'r') as pos, open('tagged_data/' + ner_file.split('/')[2] + '/' + file_name.split('/')[-1].replace('.txt', '') + '.tagged', 'w') as out, open('tagged_data/' + ner_file.split('/')[2] + '/' + file_name.split('/')[-1].replace('.txt', '') + '.ner', 'w') as out_ner:
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
                    spl = split_regex.split(line.strip())

                    assert len(cur_pos) == len(spl)

                    in_ne = False
                    cur_type = 'O'

                    for token, ner_token in zip(cur_pos, spl):
                        if in_ne:
                            out_line += token + '/I-' + cur_type + ' '
                            if '</ENAMEX>' in ner_token:
                                in_ne = False
                                cur_type = 'O'
                        elif 'TYPE=' in ner_token:
                            cur_type = ner_token.split('">')[0].replace('<ENAMEX TYPE="', '').split('"')[0]
                            if cur_type in types:
                                in_ne = True
                                out_line += token + '/B-' + cur_type + ' '
                                if '</ENAMEX>' in ner_token:
                                    in_ne = False
                                    cur_type = 'O'
                            else:
                                cur_type = 'O'
                                out_line += token + '/O' + ' '
                        else:
                            out_line += token.replace('</ENAMEX>','') + '/' + cur_type + ' '
                    
                    out.write(out_line.strip() + '\n')
                    
                out_line = ''
                spl = split_regex.split(line.strip())

                in_ne = False
                cur_type = 'O'

                for ner_token in spl:
                    if in_ne:
                        out_line += ner_token.replace('</ENAMEX>','') + '/I-' + cur_type + ' '
                        if '</ENAMEX>' in ner_token:
                            in_ne = False
                            cur_type = 'O'
                    elif 'TYPE=' in ner_token:
                        cur_type = ner_token.split('">')[0].replace('<ENAMEX TYPE="', '').split('"')[0]
                        if cur_type in types:
                            in_ne = True
                            out_line += ner_token.split('">')[1].replace('</ENAMEX>','') + '/B-' + cur_type + ' '
                            if '</ENAMEX>' in ner_token:
                                in_ne = False
                                cur_type = 'O'
                        else:
                            cur_type = 'O'
                            out_line += ner_token.split('">')[1].replace('</ENAMEX>','') + '/O' + ' '
                    else:
                        out_line += ner_token.replace('</ENAMEX>','') + '/' + cur_type + ' '
                
                out_ner.write(out_line.strip() + '\n')
                count_of_data += 1

    except IOError:
        continue

print(count_of_data)
print(types)
