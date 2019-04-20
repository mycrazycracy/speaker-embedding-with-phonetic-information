#!/bin/bash

data_in=$1
per_spk=$2
data_out=$3

utils/copy_data_dir_new.sh $data_in $data_out
python utils/sort_data_dir_by_len.py $data_out
cut -d ' ' -f 1-$(($per_spk+1)) $data_out/spk2utt > $data_out/spk2utt.new
mv $data_out/spk2utt.new $data_out/spk2utt
utils/spk2utt_to_utt2spk.pl $data_out/spk2utt > $data_out/utt2spk
utils/fix_data_dir.sh $data_out

# utils/filter_scp.pl $data_out/utt2spk $data_in/utt2num_frames > $data_out/utt2num_frames
awk -v num=0 '{num+=$2} END {print num}' $data_out/utt2num_frames
