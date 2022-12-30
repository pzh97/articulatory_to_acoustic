import os
import re
import sys

def transcription_generator(filename):
    order = r'[0-9]'
    with open(filename) as t:
        lines = []
        for line in t:
            new_line = re.sub(order, '', line).replace('. ', '').replace('.', '').replace('?', '').replace('\'', '').replace(',', '').replace('\"', '').replace('-', ' ')
            new_line = new_line.strip()
            if new_line:
                lines.append(new_line)
    path = r'./corpus'
    if not os.path.exists(path):
        os.makedirs(path)
                
    for i in range(len(lines)):
        with open("./corpus/" + sys.argv[1] + "_%03d.txt"%(i+1), 'w') as w:
            w.write(lines[i])

transcription_generator('./mocha-timit.txt')
