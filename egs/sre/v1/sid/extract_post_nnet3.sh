#!/bin/bash

# This script extract floored posteriors for i-vector training


# Begin configuration section.
nj=10
cmd="run.pl"
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3
use_gpu=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ] && [ $# != 5 ]; then
  echo "Usage: $0 <dnn-dir> <data-speaker-id> <data-dnn> <post-dir> [dnn-model]"
  echo " e.g.: $0 exp/dnn data/train data/train_dnn exp/post_male"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|-4>                               # To control partial reruns"
  echo "  --use-gpu <false> "
  echo "  --posterior-scale <1.0>"
  echo "  --min-post <0.025>"
  exit 1;
fi

nnet=$1
data=$2
data_dnn=$3
dir=$4

nnet_mdl=$nnet/final.mdl
if [ $# == 5 ]; then
  nnet_mdl=$5
fi

nnet_raw="nnet3-am-copy --raw=true $nnet_mdl - |"

for f in $data/feats.scp $data_dnn/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

sdata_dnn=$data_dnn/split$nj;
utils/split_data.sh $data_dnn $nj || exit 1;

## Set up features.
cmvn_opts=`cat $nnet/cmvn_opts 2>/dev/null`
nnet_feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_dnn/JOB/utt2spk scp:$sdata_dnn/JOB/cmvn.scp scp:$sdata_dnn/JOB/feats.scp ark:- |"

echo $nj > $dir/num_jobs


if $use_gpu; then
  if ! cuda-compiled; then
    echo "$0: WARNING: you are trying to use the GPU but you have not compiled"
    echo "   for CUDA.  If you have GPUs and have nvcc installed, go to src/"
    echo "   and do ./configure; make"
    exit 1
  fi
  for g in $(seq $nj); do
    $cmd --gpu 1 $dir/log/post.$g.log \
      nnet3-compute --use-gpu=yes "$nnet_raw" \
        "`echo $nnet_feats | sed s/JOB/$g/g`" \
        ark:- \
        \| select-voiced-frames ark:- scp,s,cs:$sdata/$g/vad.scp ark:- \
        \| logprob-to-post --min-post=$min_post ark:- ark:- \
        \| scale-post ark:- $posterior_scale "ark:|gzip -c >$dir/post.$g.gz" || exit 1 &
  done
  wait
else
  echo "$0: without using a GPU this will be slow."
  $cmd JOB=1:$nj $dir/log/post.JOB.log \
    nnet3-compute --use-gpu=no "$nnet_raw" \
      "$nnet_feats" \
      ark:- \
      \| select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- \
      \| logprob-to-post --min-post=$min_post ark:- ark:- \
      \| scale-post ark:- $posterior_scale "ark:|gzip -c >$dir/post.JOB.gz" || exit 1
fi



