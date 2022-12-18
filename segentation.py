import textgrids
import sys
import numpy as np
from data import time_vec

def output_builder(testgrid):
    phon_max = []
    count = 0
    labels = []

    grid = textgrids.TextGrid('./test.TextGrid')


    for phon in grid['phones']:
        count+=1
        label = phon.text.transcode()
        labels.append(label)
        print('"{}";{};{}'.format(label, phon.xmin, phon.xmax))
        phon_max.append(phon.xmax)

    print(phon_max)
    print(count)
    output_array = np.zeros(len(time_vec))
    phone_value = {}
    print(labels)

    for i in range(count):
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

output_builder('./test.TextGrid')
