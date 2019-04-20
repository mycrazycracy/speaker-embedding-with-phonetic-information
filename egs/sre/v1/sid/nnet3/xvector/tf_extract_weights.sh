#!/bin/bash

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <info-dir> <nnet-file> <data> <weights-dir>"
  echo " e.g.: $0 exp/xvector_egs exp/xvector_nnet/nnet data/train exp/weights_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
fi

infodir=$1
nnet_file=$2
data=$3
dir=$4

for f in $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

utils/split_data.sh $data $nj
sdata=$data/split$nj/JOB

name=`basename $data`

# disable GPUs
export CUDA_VISIBLE_DEVICES=""

srcdir=$(dirname $nnet_file)

if [ $stage -le 0 ]; then
  echo "$0: extracting weights from nnet"
  $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
    python $srcdir/config/extract_weights.py --info_dir $infodir ${sdata}/feats.scp ${dir}/tf_weights_$name.JOB.ark $nnet_file || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining weights across jobs"
  $cmd JOB=1:$nj ${dir}/log/generate_scp.JOB.log \
    copy-vector ark:${dir}/tf_weights_$name.JOB.ark ark,scp:${dir}/weights_$name.JOB.ark,${dir}/weights_$name.JOB.scp || exit 1;
  for j in $(seq $nj); do cat $dir/weights_$name.$j.scp; done >$dir/weights_$name.scp || exit 1;
  for j in $(seq $nj); do rm -f ${dir}/tf_weights_$name.$j.ark; done || exit 1
fi

