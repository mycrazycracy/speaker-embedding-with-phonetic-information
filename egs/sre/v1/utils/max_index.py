#!/usr/bin/env python

import sys
import numpy as np

try:
    ark = sys.argv[1]
    with open(ark, 'r') as f:
        for line in f.readlines():
            [utt, score] = line.split('[')
            utt = utt.strip(' ')
            score = [ float(s) for s in score.strip(' ').split(' ')[:-1] ]
            index = np.argmax(score)
            print('{0} {1}'.format(utt, index))
except:
    print('input text ark as the first argument')
