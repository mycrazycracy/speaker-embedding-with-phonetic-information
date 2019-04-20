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
    python $srcdir/config/extract_embedding.py --info_dir $infodir ${sdata}/feats.scp ${dir}/xvector_$name.JOB.ark $srcdir $output_node || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  $cmd JOB=1:$nj ${dir}/log/generate_scp.JOB.log \
    copy-vector ark:${dir}/xvector_$name.JOB.ark ark,scp:${dir}/tf_xvector_$name.JOB.ark,${dir}/tf_xvector_$name.JOB.scp || exit 1;
  for j in $(seq $nj); do cat $dir/tf_xvector_$name.$j.scp; done >$dir/tf_xvector_$name.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors.
  echo "$0: computing mean of xvectors for each speaker"
  $cmd $dir/log/speaker_mean.log \
    ivector-mean ark:$data/spk2utt scp:$dir/tf_xvector_$name.scp \
      ark,scp:$dir/tf_spk_xvector_$name.ark,$dir/tf_spk_xvector_$name.scp ark,t:$dir/num_utts.ark || exit 1;
  for j in $(seq $nj); do rm -f ${dir}/xvector_$name.$j.ark; done || exit 1
fi

