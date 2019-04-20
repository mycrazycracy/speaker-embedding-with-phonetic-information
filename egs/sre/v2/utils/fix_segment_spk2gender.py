#!/usr/bin/env python

import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: %s spk2gender_old utt2spk spk2gender_new' % sys.argv[0])
        quit()

    f_spk2gender_old = sys.argv[1]
    f_utt2spk = sys.argv[2]
    f_spk2gender_new = sys.argv[3]

    spk2gender = {}
    with open(f_spk2gender_old, 'r') as f:
        for line in f.readlines():
            [spk, gender] = line.strip().split(' ')
            spk2gender[spk] = gender

    fid_spk2gender = open(f_spk2gender_new, 'w')

    with open(f_utt2spk, 'r') as f:
        for line in f.readlines():
            [utt, spk] = line.strip().split(' ')
            fid_spk2gender.write('%s %s\n' % (utt, spk2gender[spk]))

    fid_spk2gender.close()

