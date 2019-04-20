#!/bin/bash

# Copyright     2017  David Snyder
#               2017  Johns Hopkins University (Author: Daniel Povey)
#               2017  Johns Hopkins University (Author: Daniel Garcia Romero)
# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.  The purpose of this script
# is analogous to sid/extract_ivectors.sh: it creates archives of
# vectors that are used in speaker recognition.  Like ivectors, xvectors can
# be used in PLDA or a similar backend for scoring.

# Begin configuration section.
nj=30
cmd="run.pl"
chunk_size=-1 # The chunk size over which the embedding is extracted.
              # If left unspecified, it uses the max_chunk_size in the nnet
              # directory.
use_gpu=false
stage=0
input_name=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <nnet-dir> <output-node> <data> <data-am> <cvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/embedding exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
  echo "  --input-name <input,input_xvec>"  
fi

srcdir=$1
output_node=$2
data=$3
data_am=$4
dir=$5

for f in $srcdir/am/cmvn_opts $srcdir/final.raw $srcdir/min_chunk_size $srcdir/max_chunk_size $data/feats.scp $data/vad.scp $data_am/feats.scp $data_am/cmvn.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

min_chunk_size=`cat $srcdir/min_chunk_size 2>/dev/null`
max_chunk_size=`cat $srcdir/max_chunk_size 2>/dev/null`

nnet=$srcdir/final.raw


if [ $chunk_size -le 0 ]; then
  chunk_size=$max_chunk_size
fi

if [ $max_chunk_size -lt $chunk_size ]; then
  echo "$0: specified chunk size of $chunk_size is larger than the maximum chunk size, $max_chunk_size" && exit 1;
fi

mkdir -p $dir/log

am_cmvn_opts=$(cat $srcdir/am/cmvn_opts)

cut -d ' ' -f 1 $data/feats.scp > $data/feats.scp.new
cut -d ' ' -f 1 $data_am/feats.scp > $data_am/feats.scp.new
[ `diff $data/feats.scp.new $data_am/feats.scp.new | wc -l` -ne 0 ] && echo "am and xvec data mismatch" && exit 1

cp $data/vad.scp $data_am/

echo "$0: extracting cvectors for $data"
utils/split_data.sh $data $nj
sdata=$data/split$nj/JOB

utils/split_data.sh $data_am $nj
sdata_am=$data_am/split$nj/JOB

name=`basename $data`

# Set up the features
feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"

feat_am="ark:apply-cmvn $am_cmvn_opts --utt2spk='ark:${sdata_am}/utt2spk' scp:${sdata_am}/cmvn.scp scp:${sdata_am}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata_am}/vad.scp ark:- |"


if [ $stage -le 0 ]; then
  echo "$0: extracting cvectors from nnet"
  if $use_gpu; then
    for g in $(seq $nj); do
      $cmd --gpu 1 ${dir}/log/extract.$g.log \
        nnet3-cvector-compute-multiple-input --use-gpu=yes --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --name="$input_name" \
        2 "$nnet" "`echo $feat | sed s/JOB/$g/g`" "`echo $feat_am | sed s/JOB/$g/g`" ark,t:${dir}/xvector_$name.$g.ark || exit 1 &
    done
    wait
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      nnet3-cvector-compute-multiple-input --use-gpu=no --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --name="$input_name" \
      2 "$nnet" "$feat" "$feat_am" ark,t:${dir}/xvector_$name.JOB.ark || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining cvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector_$name.$j.ark; done >$dir/xvector_$name.ark || exit 1;
fi

