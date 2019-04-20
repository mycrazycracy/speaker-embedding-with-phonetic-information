#!/usr/bin/env python

import sys
import os 

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('%s ali.raw' % sys.argv[0])
        quit()
    ali = sys.argv[1]
    if not os.path.isfile(ali):
        print('[ERROR] Cannot find ali.raw %s' % ali)
        quit()

    spk = set()
    with open(ali, 'r') as f:
        for line in f.readlines():
            spk = spk | set(line.strip().split(' ')[1:])

    print(len(spk))
