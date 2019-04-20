#/usr/bin/env python
import sys
import os

def print_max_len(file_utt2num_frames, file_spk2utt, file_spk2max_len):
    utt2num_frames = {}
    with open(file_utt2num_frames, 'r') as f:
        for line in f.readlines():
            [utt, num] = line.strip().split(' ')
            utt2num_frames[utt] = int(num)
    
    spk2max_len = open(file_spk2max_len, 'w')
    with open(file_spk2utt, 'r') as f:
        for line in f.readlines():
            s = line.strip().split(' ')
            max_len = 0
            for utt in s[1:]:
                if utt2num_frames[utt] > max_len:
                    max_len = utt2num_frames[utt]
            spk2max_len.write('%s %d\n' % (s[0], max_len))
    spk2max_len.close()
        

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s data' % sys.argv[0])
        quit()

    data_dir = sys.argv[1]
    utt2num_frames = os.path.join(data_dir, 'utt2num_frames')
    spk2utt = os.path.join(data_dir, 'spk2utt')
    spk2max_len = os.path.join(data_dir, 'spk2max_frames')
    print_max_len(utt2num_frames, spk2utt, spk2max_len)

