#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.


. ./cmd.sh
set -e

stage=-10
use_gpu=true

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

nnet_dir=$1

am_egs_dir=$nnet_dir/egs_am
xvec_egs_dir=$nnet_dir/egs_xvec

minibatch_size='256;64'  # minibatch_size for am and xvector
dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123

steps/nnet3/train_cvector_dnn.py --stage=$stage \
  --cmd="$train_cmd" \
  --trainer.optimization.proportional-shrink 10 \
  --trainer.optimization.momentum=0.5 \
  --trainer.optimization.num-jobs-initial=2 \
  --trainer.optimization.num-jobs-final=4 \
  --trainer.optimization.initial-effective-lrate=0.001 \
  --trainer.optimization.final-effective-lrate=0.0001 \
  --trainer.optimization.minibatch-size="$minibatch_size" \
  --trainer.srand=$srand \
  --trainer.max-param-change=2 \
  --trainer.num-epochs=3 \
  --trainer.dropout-schedule="$dropout_schedule" \
  --trainer.shuffle-buffer-size=1000 \
  --cleanup.remove-egs=false \
  --cleanup.preserve-model-interval=10 \
  --use-gpu=true \
  --am-output-name="output_am" \
  --am-weight=1.0 \
  --am-egs-dir=$nnet_dir/egs_am \
  --xvec-output-name="output" \
  --xvec-weight=1.0 \
  --xvec-egs-dir=$nnet_dir/egs_xvec \
  --dir=$nnet_dir  || exit 1;



