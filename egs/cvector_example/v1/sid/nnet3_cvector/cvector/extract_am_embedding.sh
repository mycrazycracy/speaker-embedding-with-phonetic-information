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
compress=true

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <nnet-dir> <output-node> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --compress"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
output_node=$2
data=$3
dir=$4

for f in $srcdir/final.mdl $data/feats.scp $data/vad.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

echo "$0: extract output for node $output_node"
echo "output-node name=output input=$output_node" > $dir/extract.config
nnet="nnet3-am-copy --raw=true $srcdir/final.mdl - | nnet3-copy --nnet-config=$dir/extract.config - - |"

utils/split_data.sh $data $nj
sdata=$data/split$nj/JOB

name=`basename $data`

echo "$0: extracting embeddings for $data"

# Set up the features
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
feats="ark:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/utt2spk scp:$sdata/cmvn.scp scp:$sdata/feats.scp ark:- |"

if $use_gpu; then
  compute_queue_opt="--gpu 1"
  compute_gpu_opt="--use-gpu=yes"
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  echo "$0: without using a GPU this will be very slow.  nnet3 does not yet support multiple threads."
  compute_gpu_opt="--use-gpu=no"
fi

$cmd $compute_queue_opt JOB=1:$nj ${dir}/log/extract_embedding_$name.JOB.log \
  nnet3-compute $compute_gpu_opt "$nnet" "$feats" ark:- \| \
  copy-feats --compress=$compress ark:- ark,scp:${dir}/am_embedding_$name.JOB.ark,${dir}/am_embedding_$name.JOB.scp || exit 1;

N0=$(cat $data/feats.scp | wc -l)
N1=$(cat $dir/am_embedding_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: Error happens when generating BNF for $name (Original:$N0  BNF:$N1)"
  exit 1;
fi

for j in $(seq $nj); do cat $dir/am_embedding_$name.$j.scp; done >$dir/feats.scp || exit 1;

