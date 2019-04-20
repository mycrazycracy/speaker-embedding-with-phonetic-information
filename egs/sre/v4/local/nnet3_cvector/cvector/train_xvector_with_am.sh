#!/bin/bash

. ./cmd.sh
set -e

train_cmd=run.pl
fea_nj=16
stage=0
train_stage=-10
am_lr_factor=0
use_gpu=true
compress=true
egs=

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ $# != 4 ]; then
  echo "$0 <am-embedding-node> <am-nnet> <data> <dir>"
  echo ""
  echo "Options:"
  echo "  stage, use-gpu"
  exit 1
fi

am_node=$1
am_mdl=$2
data=$3
dir=$4

num_speakers=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

echo "am_lr_factor=${am_lr_factor}"

if [ $stage -le 0 ]; then
  echo "$0: Create neural net configs using the xconfig parser for"
  echo "    generating new layers. The am nnet is used to feed the embedding"
  echo "    to different layer of xvector nnet."
  
  max_chunk_size=10000
  min_chunk_size=25
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  relu-batchnorm-layer name=tdnn1_xvec input=Append(input@-2,input@-1,input@0,input@1,input@2) dim=512
  relu-batchnorm-layer name=tdnn2_xvec input=Append(tdnn1_xvec@-2,tdnn1_xvec@0,tdnn1_xvec@2) dim=512
  relu-batchnorm-layer name=tdnn3_xvec input=Append(tdnn2_xvec@-3,tdnn2_xvec@0,tdnn2_xvec@3) dim=512
  relu-batchnorm-layer name=tdnn4_xvec dim=512 input=tdnn3_xvec
  relu-batchnorm-layer name=tdnn5_xvec dim=1500 input=Append(tdnn4_xvec, $am_node)

  stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size}) input=tdnn5_xvec

  relu-batchnorm-layer name=tdnn6_xvec dim=512 input=stats
  relu-batchnorm-layer name=tdnn7_xvec dim=512 input=tdnn6_xvec
  output-layer name=output include-log-softmax=true dim=${num_speakers} input=tdnn7_xvec
EOF

  steps/nnet3/xconfig_to_configs.py --existing-model $am_mdl \
    --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ 
  
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$am_lr_factor" $am_mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
  
  cp $dir/configs/final.config $dir/nnet.config
  echo "$max_chunk_size" > $dir/max_chunk_size
  echo "$min_chunk_size" > $dir/min_chunk_size
fi

if [ $stage -le 1 ]; then
  if [ -z $egs ]; then
    sid/nnet3_cvector/cvector/get_egs_xvec.sh --nj 6 \
      --stage 0 \
      --min-frames-per-chunk 200 \
      --max-frames-per-chunk 400 \
      --repeats-per-spk 5000 \
      --num-heldout-utts 1000 \
      --num-train-archives 107 \
      $data \
      $dir/egs
  fi
fi

if [ -z $egs ]; then
  egs=$dir/egs
fi

if [ $stage -le 2 ]; then
  dropout_schedule='0,0@0.20,0.1@0.50,0'
  srand=123
  steps/nnet3/train_raw_dnn_new.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.input-model=$dir/input.raw \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=64 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=3 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir=$egs \
    --cleanup.remove-egs=false \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --dir=$dir  || exit 1;
fi



