#!/bin/bash

# This script generate the config files,
# and other information for AM and xvector net training

stage=0
train_stage=-10
num_senones=
num_speakers=
feat_dim=
am_lr_factor=0.0
am_weight=1.0
xvec_weight=1.0
max_chunk_size=10000
min_chunk_size=25


if [ -f path.sh ]; then . ./path.sh; fi
. ./cmd.sh
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: $0 <am-node> <am-mdl> <am-egs-dir> <xvec-egs-dir> <nnet-dir>"
  echo " e.g.: $0 tdnn6_batchnorm am/ am_egs/ xvec_egs/ nnet3"
  echo "  "
  echo "  --num-senones"
  echo "  --num-speakers"
  echo "  --feat-dim"
  echo "  --stage"
  echo "  --train-stage"
  echo "  --am-lr-factor"
  echo "  --am-weight"
  echo "  --xvec-weight"
  echo "  --max-chunk-size <10000>"
  echo "  --min-chunk-size <25>"
  exit 1
fi

am_node=$1
am_mdl=$2
am_egs_dir=$3
xvec_egs_dir=$4
nnetdir=$5

mkdir -p $nnetdir

[ -z $num_senones ] && echo "num-senones must be specified" && exit 1
[ -z $num_speakers ] && echo "num-speakers must be specified" && exit 1
[ -z $feat_dim ] && echo "feat-dim must be specified" && exit 1

echo "am_lr_factor=${am_lr_factor}"


if [ $stage -le 0 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  echo "num of senones: $num_senones and num of speakers: $num_speakers"

  echo "$0: Create neural net configs using the xconfig parser for"
  echo "    generating new layers. The am nnet is used to feed the embedding"
  echo "    to different layer of xvector nnet."
  
  mkdir -p $nnetdir/configs

  cat <<EOF > $nnetdir/configs/network.xconfig.old
  input dim=$feat_dim name=input

  # shared part
  relu-batchnorm-layer name=tdnn1_share dim=512 input=Append(input@-2,input@-1,input@0,input@1,input@2)
  relu-batchnorm-layer name=tdnn2_share dim=512 input=Append(tdnn1_share@-2,tdnn1_share@0,tdnn1_share@2)

  # am part
  relu-batchnorm-layer name=tdnn3_am dim=512 input=Append(tdnn2_share@-3,tdnn2_share@0,tdnn2_share@3)
  relu-batchnorm-layer name=tdnn4_am dim=512 input=tdnn3_am
  relu-batchnorm-layer name=tdnn5_am dim=512 input=tdnn4_am
  relu-batchnorm-layer name=tdnn6_am dim=512 input=tdnn5_am
  relu-batchnorm-layer name=tdnn7_am dim=512 input=tdnn6_am
  output-layer name=output_am dim=$num_senones max-change=1.5 input=tdnn7_am

  # xvector part
  relu-batchnorm-layer name=tdnn3_xvec dim=512 input=Append(tdnn2_share@-3,tdnn2_share@0,tdnn2_share@3)
  relu-batchnorm-layer name=tdnn4_xvec dim=512 input=tdnn3_xvec
  relu-batchnorm-layer name=tdnn5_xvec dim=1500 input=Append(tdnn4_xvec,$am_node)
  stats-layer name=stats_xvec config=mean+stddev(0:1:1:${max_chunk_size}) input=tdnn5_xvec
  relu-batchnorm-layer name=tdnn6_xvec dim=512 input=stats_xvec
  relu-batchnorm-layer name=tdnn7_xvec dim=512 input=tdnn6_xvec
  output-layer name=output_xvec include-log-softmax=true dim=$num_speakers input=tdnn7_xvec
EOF

  mkdir -p $nnetdir/configs_am $nnetdir/configs_xvec

  # For two different tasks, the context is different. We create two branches and initialize 
  # both of them to get their context respectively.
  echo "processing Xvector nnet"
  sed "s/output_xvec/output/g" $nnetdir/configs/network.xconfig.old > $nnetdir/configs_xvec/network.xconfig
  steps/nnet3/xconfig_to_configs.py --existing-model $am_mdl \
    --xconfig-file $nnetdir/configs_xvec/network.xconfig \
    --config-dir $nnetdir/configs_xvec
  
  echo "processing AM nnet"
  sed "s/output_am/output/g" $nnetdir/configs/network.xconfig.old > $nnetdir/configs_am/network.xconfig
  steps/nnet3/xconfig_to_configs.py --existing-model $am_mdl \
    --xconfig-file $nnetdir/configs_am/network.xconfig \
    --config-dir $nnetdir/configs_am

  echo "process Cvector nnet"
  sed "s/output_xvec/output/g" $nnetdir/configs/network.xconfig.old > $nnetdir/configs/network.xconfig
  steps/nnet3/xconfig_to_configs.py --existing-model $am_mdl \
    --xconfig-file $nnetdir/configs/network.xconfig \
    --config-dir $nnetdir/configs

  cp $nnetdir/configs_xvec/vars $nnetdir/configs/vars_xvec
  cp $nnetdir/configs_am/vars $nnetdir/configs/vars_am
  cp $nnetdir/configs/final.config $nnetdir/nnet.config
  echo "$max_chunk_size" > $nnetdir/max_chunk_size
  echo "$min_chunk_size" > $nnetdir/min_chunk_size

  $train_cmd $nnetdir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$am_lr_factor" $am_mdl - \| \
      nnet3-init --srand=1 - $nnetdir/configs/final.config $nnetdir/input.raw  || exit 1;
fi


if [ $stage -le 1 ]; then
  minibatch_size='256;64'  # minibatch_size for am and xvector
  dropout_schedule='0,0@0.20,0.1@0.50,0'
  srand=123

  steps/nnet3/train_cvector_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.input-model=$nnetdir/input.raw \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
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
    --am-weight=$am_weight \
    --am-egs-dir=$am_egs_dir \
    --xvec-output-name="output" \
    --xvec-weight=$xvec_weight \
    --xvec-egs-dir=$xvec_egs_dir \
    --dir=$nnetdir  || exit 1;
fi



