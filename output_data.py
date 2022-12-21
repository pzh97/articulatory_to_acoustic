import textgrids
import sys
import numpy as np
from input_data import time_vec

def output_builder(textgrid):
    phon_max = []
    labels = ['aj', 'aw', 'b', 'bʲ', 'c', 'cʰ', 'd', 'dʒ', 'dʲ', 'd̪', 'ej', 'f', 'fʲ', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'l', 'm', 'mʲ', 'm̩', 'n', 'n̩', 'p', 'pʰ', 'pʲ', 's', 't', 'tʃ', 'tʰ', 'tʲ', 't̪', 'v', 'vʲ', 'w', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔj', 'ə', 'əw', 'ɛ', 'ɛː', 'ɜ', 'ɜː', 'ɟ', 'ɡ', 'ɪ', 'ɫ', 'ɫ̩', 'ɱ', 'ɲ', 'ɹ', 'ʃ', 'ʉ', 'ʉː', 'ʊ', 'ʎ', 'ʒ', 'ʔ', 'θ', '']

    grid = textgrids.TextGrid(textgrid)


    for phon in grid['phones']:
        label = phon.text.transcode()
        print('"{}";{};{}'.format(label, phon.xmin, phon.xmax))
        phon_max.append(phon.xmax)

    print(phon_max)
    output_array = np.zeros(len(time_vec))
    phone_value = {}
    print(labels)

    for i in range(len(labels)):
        phone_value[labels[i]] = i
    print(phone_value)
    index = []
    for i in range(count-2):
        result = next(k for k, value in enumerate(time_vec) if phon_max[i]<value)
        index.append(result)

    print(index)

    for n in range(index[0]):
        output_array[0:index[0]] = phone_value[labels[0]]
    print(output_array[354])

    n = 1
    for n in range(index[n+1]-index[n]+1):
        if n < 12:
            output_array[index[n]:index[n+1]] = phone_value[labels[n]]
    print(output_array[421])

    for n in range(len(time_vec)-index[-1]+1):
        output_array[index[-1]:] = phone_value[labels[-1]]
    print(output_array[-1])
    print(len(output_array))
    print(len(time_vec))

#output_builder('./test.TextGrid')
