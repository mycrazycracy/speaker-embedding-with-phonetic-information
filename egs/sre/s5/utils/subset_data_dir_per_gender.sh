#!/bin/bash

data_in=$1
per_gender=$2
data_out=$3

utils/copy_data_dir_new.sh $data_in $data_out
python -c "
import sys
utt2num_frames = {}
with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        [utt, num] = line.strip().split(' ')
        utt2num_frames[utt] = int(num)

spk2num_frames = {}
spk2utt = {}
with open(sys.argv[3], 'r') as f:
    for line in f.readlines():
        s = line.strip().split(' ')
        num = 0
        spk2utt[s[0]] = []
        for utt in s[1:]:
            num += utt2num_frames[utt]
            spk2utt[s[0]].append(utt)
        spk2num_frames[s[0]] = num

gender2spk = {}
with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        [spk, gender] = line.strip().split(' ')
        if gender not in gender2spk:
            gender2spk[gender] = []
        gender2spk[gender].append([spk, spk2num_frames[spk]])

spk2utt_new = open(sys.argv[5], 'w')
per_gender = int(sys.argv[4])
for gender in gender2spk:
    sorted_spk = sorted(gender2spk[gender], reverse=True, key=lambda x:x[1])
    for spk in sorted_spk[:per_gender]:
        spk2utt_new.write('%s' % spk[0])
        for utt in spk2utt[spk[0]]:
            spk2utt_new.write(' %s' % utt)
        spk2utt_new.write('\n')
spk2utt_new.close()
" $data_in/utt2num_frames $data_in/spk2gender $data_in/spk2utt $per_gender $data_out/spk2utt
utils/spk2utt_to_utt2spk.pl $data_out/spk2utt > $data_out/utt2spk
utils/fix_data_dir.sh $data_out

# utils/filter_scp.pl $data_out/utt2spk $data_in/utt2num_frames > $data_out/utt2num_frames
awk -v num=0 '{num+=$2} END {print num}' $data_out/utt2num_frames

