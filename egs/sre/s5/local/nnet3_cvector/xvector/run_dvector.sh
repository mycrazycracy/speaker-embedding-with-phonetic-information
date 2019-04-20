#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This script trains a DNN similar to the recipe described in
# http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf .

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true
remove_egs=false

data=data
nnet_dir=exp/dvector_nnet_1a
egs_dir=exp/dvector_nnet_1a/egs

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

if [ $stage -le 0 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$num_pdfs
  feat_dim=$(feat-to-dim scp:$data/feats.scp -)

  mkdir -p $nnet_dir/configs
  cat <<EOF > $nnet_dir/configs/network.xconfig
  # please note that it is important to have input layer with the name=input

  # The frame-level layers
  input dim=${feat_dim} name=input
  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
  relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn4 input=Append(-4,0,4) dim=512
  relu-batchnorm-layer name=tdnn5 input=Append(-5,0,5) dim=512
  relu-batchnorm-layer name=tdnn6 dim=512
  output-layer name=output include-log-softmax=true dim=${num_targets}
EOF

  steps/nnet3/xconfig_to_configs.py \
      --xconfig-file $nnet_dir/configs/network.xconfig \
      --config-dir $nnet_dir/configs/
  cp $nnet_dir/configs/final.config $nnet_dir/nnet.config
  echo "output-node name=output input=tdnn6.affine" > $nnet_dir/extract.config

fi

# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <spk-id>
#
# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# You may need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
if [ $stage -le 1 ]; then
  echo "$0: Getting neural network training egs";
  
  left_context=`grep 'model_left_context' $nnet_dir/configs/vars | awk -F '=' '{print $NF}'`
  right_context=`grep 'model_right_context' $nnet_dir/configs/vars | awk -F '=' '{print $NF}'`

  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  sid/nnet3/xvector/get_egs_dvector.sh --cmd "$train_cmd" \
    --nj 6 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 300 \
    --max-frames-per-chunk 500 \
    --num-diagnostic-archives 3 \
    --num-repeats 35 \
    --left-context $left_context \
    --right-context $right_context \
    "$data" $egs_dir
fi
num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')

#dropout_schedule='0,0@0.20,0.1@0.50,0'
#srand=123
#if [ $stage -le 6 ]; then
#  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
#    --cmd="$train_cmd" \
#    --trainer.optimization.proportional-shrink 10 \
#    --trainer.optimization.momentum=0.5 \
#    --trainer.optimization.num-jobs-initial=3 \
#    --trainer.optimization.num-jobs-final=6 \
#    --trainer.optimization.initial-effective-lrate=0.001 \
#    --trainer.optimization.final-effective-lrate=0.0001 \
#    --trainer.optimization.minibatch-size=64 \
#    --trainer.srand=$srand \
#    --trainer.max-param-change=2 \
#    --trainer.num-epochs=3 \
#    --trainer.dropout-schedule="$dropout_schedule" \
#    --trainer.shuffle-buffer-size=1000 \
#    --egs.frames-per-eg=1 \
#    --egs.dir="$egs_dir" \
#    --cleanup.remove-egs $remove_egs \
#    --cleanup.preserve-model-interval=10 \
#    --use-gpu=true \
#    --dir=$nnet_dir  || exit 1;
#fi
#
#exit 0;
