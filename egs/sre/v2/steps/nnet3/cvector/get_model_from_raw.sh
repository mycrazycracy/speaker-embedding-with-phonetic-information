#!/bin/bash

cmd=run.pl
prior_subset_size=20000
use_gpu=no
nj=10
prior=

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <ali-dir> <egs-dir> <nnet-dir> "
  echo "main options (for others, see top of script file)"
  echo "  --prior-subset-size <20000>              # #frames "
  echo "  --use-gpu <false>"
  exit 1;
fi

alidir=$1
egsdir=$2
dir=$3
  
# If the raw model is provided, additional steps are needed to convert raw nnet to am nnet..
# 1. Set the transition model
# 2. Add priors (estimated from the egs)

# Add transition model
nnet3-am-init ${alidir}/final.mdl ${dir}/final.raw - | \
nnet3-am-train-transitions - "ark:gunzip -c ${alidir}/ali.*.gz|" ${dir}/final.mdl

if [ -z $prior ]; then
  if [ $nj -gt `ls $egsdir/egs.*.ark | wc -l` ]; then
    echo "Set the posterior computation to only 1 egs."
    nj=1
  fi
  
  # Adjust the prior using the posterior
  $cmd JOB=1:$nj ${dir}/log/get_post.JOB.log \
    nnet3-copy-egs ark:$egsdir/egs.JOB.ark ark:- \| \
    nnet3-subset-egs --srand=JOB --n=$prior_subset_size \
      ark:- ark:- \| \
    nnet3-merge-egs --minibatch-size=128 ark:- ark:- \| \
    nnet3-compute-from-egs --use-gpu=$use_gpu --apply-exp=true \
      $dir/final.mdl ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.JOB.vec || exit 1
  
  sleep 5
  
  $cmd $dir/log/vector_sum.log \
    vector-sum $dir/post.*.vec $dir/post.vec
  rm -f $dir/post.*.vec
  prior=$dir/post.vec
fi

cp $dir/final.mdl $dir/final.mdl.bak
$cmd $dir/log/adjust_priors.log \
  nnet3-am-adjust-priors $dir/final.mdl $prior $dir/final.mdl

