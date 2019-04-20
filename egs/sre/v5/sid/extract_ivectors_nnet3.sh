#!/bin/bash

# Copyright     2013  Daniel Povey
#          2014-2017  David Snyder
#               2015  Johns Hopkins University (Author: Daniel Garcia-Romero)
#               2015  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained DNN-based iVector extractor.

# Begin configuration section.
cmd="run.pl"
stage=0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 <extractor-dir> <data-speaker-id> <post-dir> <ivector-dir>"
  echo " e.g.: $0 exp/extractor_2048_male data/train_male exp/post_male exp/ivectors_male"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --stage <stage|0>                                # To control partial reruns"
  exit 1;
fi

srcdir=$1
data=$2
post_dir=$3
dir=$4

for f in $srcdir/final.ie $srcdir/final.ubm $data/feats.scp $post_dir/num_jobs ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

nj=$(cat $post_dir/num_jobs)

# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

## Set up features.
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
      ivector-extract --verbose=2 $srcdir/final.ie \
      "$feats" "ark,s,cs:gunzip -c $post_dir/post.JOB.gz|" \
      ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp || exit 1
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector.$j.scp; done >$dir/ivector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Be careful here: the speaker-level iVectors are now length-normalized,
  # even if they are otherwise the same as the utterance-level ones.
  echo "$0: computing mean of iVectors for each speaker and length-normalizing"
  $cmd $dir/log/speaker_mean.log \
    ivector-normalize-length scp:$dir/ivector.scp  ark:- \| \
    ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
    ivector-normalize-length ark:- ark,scp:$dir/spk_ivector.ark,$dir/spk_ivector.scp || exit 1;
fi
