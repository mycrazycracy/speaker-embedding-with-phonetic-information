#!/bin/bash

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <info-dir> <nnet-dir> <output-node> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
fi

infodir=$1
srcdir=$2
output_node=$3
data=$4
dir=$5

for f in $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir

echo "$0: extract output for node $output_node"

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting xvectors for $data"
sdata=$data/split$nj/JOB

name=`basename $data`

# disable GPUs
export CUDA_VISIBLE_DEVICES=""

if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
    python $srcdir/config/activation_extract.py --info_dir $infodir ${sdata}/feats.scp $dir $srcdir $output_node || exit 1;
fi

