#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e 

# Begin configuration section.
stage=0
nj=30
cmd="run.pl"
use_gpu=false
compress=true
# End configuration options.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <nnet-dir> <output-node> <input-data> <output-data> <dir>"
  echo " e.g.: $0 exp/nnet data/train data/train_bn exp/train_bn"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --compress <true>"
  exit 1
fi

srcdir=$1
output_node=$2
data=$3
bnf_data=$4
dir=$5

for f in $srcdir/final.mdl $data/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

cmvn_opts=`cat $srcdir/cmvn_opts`
name=`basename $data`
sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log 
mkdir -p $bnf_data

echo "$0: extracting bottleneck features for $data"

echo "$0: Generating bottleneck features using $srcdir/final.mdl as output of "
echo "    component-node with name $output_node."
echo "output-node name=output input=$output_node" > $dir/extract.config

raw_nnet="nnet3-am-copy --raw=true $srcdir/final.mdl - | nnet3-copy --nnet-config=$dir/extract.config - - |"
# Set up the features
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  if $use_gpu; then
    echo "Set use_gpu=false"
    exit 1
  else
    $cmd JOB=1:$nj $dir/log/extract.JOB.log \
      nnet3-compute --use-gpu=no "$raw_nnet" "$feats" ark:- \| \
        copy-feats --compress=$compress ark:- ark,scp:$dir/raw_bnfeat_$name.JOB.ark,$dir/raw_bnfeat_$name.JOB.scp || exit 1;
  fi
fi

N0=$(cat $data/feats.scp | wc -l)
N1=$(cat $dir/raw_bnfeat_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: Error happens when generating bottleneck features for $name (Original:$N0  BNF:$N1)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in $(seq $nj); do  
  cat $dir/raw_bnfeat_$name.$n.scp
done > $bnf_data/feats.scp

for f in segments spk2utt spk2gender text utt2spk wav.scp vad.scp utt2num_frames char.stm glm kws reco2file_and_channel stm; do
  [ -e $data/$f ] && cp -r $data/$f $bnf_data/$f
done

if [ $stage -le 1 ]; then
  echo "$0: computing CMVN stats."
  steps/compute_cmvn_stats.sh $bnf_data $dir/log $dir
fi

echo "$0: done making bottleneck features."

exit 0;
