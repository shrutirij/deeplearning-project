'''
Changes all the tags in the file that are not /es or /en to other
'''

input_file = 'combined_data_no_dup.txt'
output_file = 'combined_data_three_class.txt'

f = open(input_file, 'rb')
w = open(output_file, 'wb')

lines = f.readlines()

for line in lines:
    parts = line.strip().split(" ")
    temp = []
    for p in parts:
        t = p.split("/")
        if t[-1] != 'en' and t[-1] != 'es':
            t[-1] = 'other'
        c = '/'.join(t)
        temp.append(c)
    to_write = ' '.join(temp) + '\n'
    w.write(to_write)
f.close()
w.close()