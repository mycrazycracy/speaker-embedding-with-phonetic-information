#!/bin/bash
# Copyright 2015-2017   David Snyder
#           2015        Johns Hopkins University (Author: Daniel Garcia-Romero)
#           2015        Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script derives a full-covariance UBM from DNN posteriors and
# speaker recognition features.

# This script have been changed to adapt a nnet3 recipe

echo "$0 $@"  # Print the command line for logging

# Begin configuration section.
nj=16
cmd="run.pl"
use_gpu=false
stage=0
delta_window=3
delta_order=2
cleanup=true
# End configuration section.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ] && [ $# != 5 ]; then
  echo "Usage: steps/init_full_ubm_from_nnet3.sh <data-speaker-id> <data-dnn> <dnn-dir> <new-ubm-dir> [dnn-model]"
  echo "Initializes a full-covariance UBM from nnet3 DNN posteriors and speaker recognition features."
  echo " e.g.: steps/init_full_ubm_from_nnet3.sh data/train data/train_dnn exp/dnn/ exp/full_ubm"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|16>                                      # number of parallel training jobs"
  echo "  --delta-window <n|3>                             # delta window size"
  echo "  --delta-order <n|2>                              # delta order"
  echo "  --use-gpu <true/false>                           # Use GPU to extract DNN posteriors"
  echo "                                                   # replace those specified by --cmd"
  exit 1;
fi


data=$1     # Features for the GMM
data_dnn=$2 # Features for the DNN
nnet=$3
dir=$4

nnet_mdl=$nnet/final.mdl
if [ $# == 5 ]; then
  nnet_mdl=$5
fi

nnet_raw="nnet3-am-copy --raw=true $nnet_mdl - |"

for f in $data/feats.scp $data/vad.scp ${data_dnn}/feats.scp \
    ${data_dnn}/vad.scp $nnet_mdl; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

logdir=$dir/log
mkdir -p $dir/log

echo $nj > $dir/num_jobs
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;
sdata_dnn=$data_dnn/split$nj;
utils/split_data.sh $data_dnn $nj || exit 1;


delta_opts="--delta-window=$delta_window --delta-order=$delta_order"
echo $delta_opts > $dir/delta_opts

feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | \
apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | \
select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

cmvn_opts=`cat $nnet/cmvn_opts 2>/dev/null`
nnet_feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_dnn/JOB/utt2spk scp:$sdata_dnn/JOB/cmvn.scp scp:$sdata_dnn/JOB/feats.scp ark:- |"

if [ -f $nnet/trans.scp ]; then
  # TODO: this may be incorrect currently
  echo "No transform expected, stop."
  nnet_feats="$nnet_feats transform-feats --utt2spk=ark:$sdata_dnn/JOB/utt2spk scp:$nnet/egs/trans.scp ark:- ark:- |"
  exit 1
fi

# Parse the output of nnet-am-info to find the size of the output layer
# of the TDNN.  This will also correspond to the number of components
# in the ancillary GMM.
num_components=`nnet3-info $nnet_raw | grep 'output-node' | grep -oP 'dim=\K[0-9]+'`
echo "num_component: $num_components"

if [ $stage -le 0 ]; then
  echo "$0: accumulating stats from DNN posteriors and speaker ID features"

  if $use_gpu; then
    if ! cuda-compiled; then
      echo "$0: WARNING: you are trying to use the GPU but you have not compiled"
      echo "   for CUDA.  If you have GPUs and have nvcc installed, go to src/"
      echo "   and do ./configure; make"
      exit 1
    fi
    for g in $(seq $nj); do
      $cmd --gpu 1 $dir/log/make_stats.$g.log \
        nnet3-compute --use-gpu=yes $nnet_raw \
          "`echo $nnet_feats | sed s/JOB/$g/g`" \
          ark:- \
          \| select-voiced-frames ark:- scp,s,cs:$sdata/$g/vad.scp ark:- \
          \| logprob-to-post ark:- ark:- \| \
          fgmm-global-acc-stats-post ark:- $num_components \
          "`echo $feats | sed s/JOB/$g/g`" \
          $dir/stats.$g.acc || exit 1 &
    done
    wait
  else
    echo "$0: without using a GPU this will be slow."
    $cmd JOB=1:$nj $dir/log/make_stats.JOB.log \
      nnet3-compute --use-gpu=no $nnet_raw \
        "$nnet_feats" \
        ark:- \
        \| select-voiced-frames ark:- "scp,s,cs:$sdata/JOB/vad.scp" ark:- \
        \| logprob-to-post ark:- ark:- \| \
        fgmm-global-acc-stats-post ark:- $num_components \
        "$feats" \
        "$dir/stats.JOB.acc" || exit 1 
  fi
fi


if [ $stage -le 1 ]; then
  echo "$0: initializing GMM from stats"
  $cmd $dir/log/init.log \
    fgmm-global-init-from-accs --verbose=2 \
    "fgmm-global-sum-accs - $dir/stats.*.acc |" $num_components \
    $dir/final.ubm || exit 1;
fi

if $cleanup; then
  echo "$0: removing stats"
  for g in $(seq $nj); do
    rm $dir/stats.$g.acc || exit 1
  done
fi













