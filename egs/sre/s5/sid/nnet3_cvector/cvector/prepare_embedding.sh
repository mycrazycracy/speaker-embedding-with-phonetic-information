#!/bin/bash

cmd="run.pl"
nj=16
compress=true
cmn_window=300

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <ref-dir> <data-dir> <target-dir>"
  echo ""
fi

refdir=$1
data=$2
dir=$3

mkdir -p $dir/temp $dir/log
temp=$dir/temp

utils/filter_scp.pl $refdir/feats.scp $data/feats.scp > $temp/feats.scp
cp $refdir/vad.scp $temp/vad.scp
cp $refdir/utt2spk $temp/utt2spk
cp $refdir/spk2utt $temp/spk2utt
cp $refdir/wav.scp $temp/wav.scp
[ -f $refdir/segments ] && cp $refdir/segments $temp/segments

# check the order
cut -d ' ' -f 1 $temp/feats.scp > $temp/feats.scp.bak
cut -d ' ' -f 1 $temp/vad.scp > $temp/vad.scp.bak
[ `diff $temp/feats.scp.bak $temp/vad.scp.bak | wc -l` -gt 0 ] && echo "Not all features in $temp have VAD info" && exit 1
rm -f $temp/feats.scp.bak $temp/vad.scp.bak 

sdata=$temp/split$nj/JOB
utils/split_data.sh $temp $nj || exit 1;

name=`basename $data`
write_num_frames_opt="--write-num-frames=ark,t:$dir/log/utt2num_frames.JOB"

$cmd JOB=1:$nj $dir/log/create_embedding_${name}.JOB.log \
  select-voiced-frames scp:$sdata/feats.scp scp,s,cs:$sdata/vad.scp ark:- \| \
  copy-feats --compress=$compress $write_num_frames_opt ark:- \
  ark,scp:$dir/am_embedding_${name}.JOB.ark,$dir/am_embedding_${name}.JOB.scp || exit 1;

# $cmd JOB=1:$nj $dir/log/create_embedding_${name}.JOB.log \
#   apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window \
#   scp:${sdata}/feats.scp ark:- \| \
#   select-voiced-frames ark:- scp,s,cs:$sdata/vad.scp ark:- \| \
#   copy-feats --compress=$compress $write_num_frames_opt ark:- \
#   ark,scp:$dir/am_embedding_${name}.JOB.ark,$dir/am_embedding_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $dir/am_embedding_${name}.$n.scp || exit 1;
done > ${dir}/feats.scp || exit 1

for n in $(seq $nj); do
  cat $dir/log/utt2num_frames.$n || exit 1;
done > $dir/utt2num_frames || exit 1
rm $dir/log/utt2num_frames.*




