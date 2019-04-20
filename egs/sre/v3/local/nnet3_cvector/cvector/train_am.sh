#!/bin/bash

. ./cmd.sh
set -e

cmd=run.pl
stage=0
train_stage=-10
use_gpu=true

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

data=$1
alidir=$2
lang=$3
dir=$4

if [ $stage -le 0 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";
  
  feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1
  num_targets=$(tree-info $alidir/tree | grep num-pdfs | awk '{print $2}') || exit 1
  
  mkdir -p $dir/configs

  # classic speech recognition setup
  cat <<EOF > $dir/configs/network.xconfig
    input dim=$feat_dim name=input
    relu-batchnorm-layer name=tdnn1 dim=650 input=Append(-2,-1,0,1,2)
    relu-batchnorm-layer name=tdnn2 dim=650 input=Append(-1,0,1)
    relu-batchnorm-layer name=tdnn3 dim=650 input=Append(-1,0,1)
    relu-batchnorm-layer name=tdnn4 dim=650 input=Append(-3,0,3)
    relu-batchnorm-layer name=tdnn5 dim=128 input=Append(-6,-3,0)
    output-layer name=output dim=$num_targets max-change=1.5
EOF
  
  steps/nnet3/xconfig_to_configs_new.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 1 ]; then
  am_left_context=$(grep 'model_left_context' $dir/configs/vars | cut -d '=' -f 2)
  am_right_context=$(grep 'model_right_context' $dir/configs/vars | cut -d '=' -f 2)
  sid/nnet3_cvector/cvector/get_egs_am.sh --cmd "$cmd" \
    --nj 6 \
    --left-context $am_left_context \
    --right-context $am_right_context \
    --stage 0 \
    $data $alidir $dir/egs
  
  # To indicate training without multitask, delete valid_diagnostic.scp
  rm -f $dir/egs/valid_diagnostic.scp
fi

if [ $stage -le 2 ]; then
  srand=123
  steps/nnet3/train_raw_dnn_new.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=3 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --egs.dir=$dir/egs \
    --cleanup.remove-egs=false \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=$use_gpu \
    --dir=$dir  || exit 1;
fi 

