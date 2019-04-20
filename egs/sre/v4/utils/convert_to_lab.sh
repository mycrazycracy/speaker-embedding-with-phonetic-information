#!/bin/bash

nj=4
cmd=run.pl
delta_window=3
delta_order=2

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

data=$1
dir=$2

sdata=$data/split$nj
mkdir -p $dir/log
utils/split_data.sh $data $nj || exit 1;

delta_opts="--delta-window=$delta_window --delta-order=$delta_order"
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

$cmd JOB=1:$nj $dir/log/convert.JOB.log \
  copy-feats-to-lab --output-dir=$dir --output-ext=mfc "$feats" || exit 1;

