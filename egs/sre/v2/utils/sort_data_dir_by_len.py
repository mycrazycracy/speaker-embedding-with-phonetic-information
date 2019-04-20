#!/usr/bin/env python
import sys
import os
import shutil


def sort_by_len(file_utt2num_frames, file_spk2utt, file_spk2utt_new):
    utt2num_frames = {}
    with open(file_utt2num_frames, 'r') as f:
        for line in f.readlines():
            [utt, num] = line.strip().split()
            num = int(num)
            utt2num_frames[utt] = num

    spk2utt_new = open(file_spk2utt_new, 'w')
    with open(file_spk2utt, 'r') as f:
        for line in f.readlines():
            s = line.strip().split()
            spk2utt_frames = []
            for utt in s[1:]:
                spk2utt_frames.append([utt, utt2num_frames[utt]])
            new_spk2utt_frames = sorted(spk2utt_frames, key=lambda x:x[1], reverse=True)
            spk2utt_new.write('%s' % s[0])
            for utt_frames in new_spk2utt_frames:
                spk2utt_new.write(' %s' % utt_frames[0])
            spk2utt_new.write('\n')

    spk2utt_new.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s data' % sys.argv[0])
        quit()
    utt2num_frames = os.path.join(sys.argv[1], 'utt2num_frames')
    spk2utt = os.path.join(sys.argv[1], 'spk2utt')
    spk2utt_new = os.path.join(sys.argv[1], 'spk2utt.new')

    sort_by_len(utt2num_frames, spk2utt, spk2utt_new)
    shutil.move(spk2utt_new, spk2utt)
 
