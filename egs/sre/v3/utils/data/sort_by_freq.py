#!/usr/bin/env python

import sys
import operator

def sort_text(text, sorted_text):
    freq = {}
    with open(text, 'r') as fp_in:
        for line in fp_in.readlines():
            s = " ".join(line.strip().split(' ')[1:])
            if s not in freq:
                freq[s] = 0
            freq[s] += 1
    sorted_freq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)

    with open(sorted_text, 'w') as fp_out:
        for v in sorted_freq:
            fp_out.write('%s %d\n' % (v[0], v[1]))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s text sorted_text' % sys.argv[0])
        quit()

    text = sys.argv[1]
    sorted_text = sys.argv[2]

    sort_text(text, sorted_text)
