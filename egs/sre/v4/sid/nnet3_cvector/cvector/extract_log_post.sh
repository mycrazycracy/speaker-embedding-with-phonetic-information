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

# Features with wcmvn
remove_silence=false
norm_vars=false
center=true
compress=true
cmn_window=300

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-in> <data> <log-post-dir>"
  echo " e.g.: $0 exp/nnet/final.raw data/train exp/log_post"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
  exit 1
fi

nnet=$1
data=$2
dir=$3

for f in $nnet $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting log-posteriors for $data"
sdata=$data/split$nj/JOB

name=`basename $data`

# Set up the features
if $remove_silence; then
  [ ! -f $data/vad.scp ] && echo "No such file $f but silence is requested to remove" && exit 1;
  feats="ark:apply-cmvn-sliding --norm-vars=$norm_vars --center=$center --cmn-window=$cmn_window scp:$sdata/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"
else
  feats="ark:apply-cmvn-sliding --norm-vars=$norm_vars --center=$center --cmn-window=$cmn_window scp:$sdata/feats.scp ark:- |"
fi

if [ $stage -le 0 ]; then
  if $use_gpu; then
    for g in $(seq $nj); do
      $cmd --gpu 1 ${dir}/log/extract.$g.log \
        nnet3-compute --use-gpu=yes \
        "$nnet" "`echo $feats | sed s/JOB/$g/g`" ark,scp:${dir}/log_post_$name.$g.ark,${dir}/log_post_$name.$g.scp || exit 1 &
    done
    wait
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      nnet3-compute --use-gpu=no \
      "$nnet" "$feats" ark,scp:${dir}/log_post_$name.JOB.ark,${dir}/log_post_$name.JOB.scp || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/log_post_$name.$j.scp; done >$dir/log_post_$name.scp || exit 1;
fi

