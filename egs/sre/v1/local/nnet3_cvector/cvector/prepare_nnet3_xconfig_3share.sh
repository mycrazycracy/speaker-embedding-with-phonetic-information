#!/bin/bash

# This script generate the config files,
# and other information for AM and xvector net training

num_senones=
num_speakers=
feat_dim=
max_chunk_size=10000
min_chunk_size=25


if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 1 ]; then
  echo "Usage: $0 <nnet-dir>"
  echo " e.g.: $0 nnet3"
  echo "  "
  echo "  --num-senones"
  echo "  --num-speakers"
  echo "  --feat-dim"
  echo "  --max-chunk-size <10000>"
  echo "  --min-chunk-size <25>"
  exit 1
fi

nnetdir=$1

mkdir -p $nnetdir

[ -z $num_senones ] && echo "num-senones must be specified" && exit 1
[ -z $num_speakers ] && echo "num-speakers must be specified" && exit 1
[ -z $feat_dim ] && echo "feat-dim must be specified" && exit 1



echo "$0: creating neural net configs using the xconfig parser";
echo "num of senones: $num_senones and num of speakers: $num_speakers"

mkdir -p $nnetdir/configs


# Configure 1: 
cat <<EOF > $nnetdir/configs/network.xconfig.old
  input dim=$feat_dim name=input

  # shared part
  relu-batchnorm-layer name=tdnn1 dim=512 input=Append(input@-2,input@-1,input@0,input@1,input@2)
  relu-batchnorm-layer name=tdnn2 dim=512 input=Append(tdnn1@-2,tdnn1@0,tdnn1@2)
  relu-batchnorm-layer name=tdnn3 dim=512 input=Append(tdnn2@-3,tdnn2@0,tdnn2@3)

  # am part
  relu-batchnorm-layer name=tdnn4_am dim=512 input=tdnn3
  relu-batchnorm-layer name=tdnn5_am dim=512 input=tdnn4_am
  relu-batchnorm-layer name=tdnn6_am dim=512 input=tdnn5_am
  relu-batchnorm-layer name=tdnn7_am dim=512 input=tdnn6_am
  output-layer name=output_am dim=$num_senones max-change=1.5 input=tdnn7_am

  # xvector part
  relu-batchnorm-layer name=tdnn4_xvec dim=512 input=tdnn3
  relu-batchnorm-layer name=tdnn5_xvec dim=1500 input=tdnn4_xvec
  stats-layer name=stats_xvec config=mean+stddev(0:1:1:${max_chunk_size}) input=tdnn5_xvec
  relu-batchnorm-layer name=tdnn6_xvec dim=512 input=stats_xvec
  relu-batchnorm-layer name=tdnn7_xvec dim=512 input=tdnn6_xvec
  output-layer name=output_xvec include-log-softmax=true dim=$num_speakers input=tdnn7_xvec
EOF

# # Configure 2: 
# cat <<EOF > $nnetdir/configs/network.xconfig.old
#   input dim=$feat_dim name=input
# 
#   # shared part
#   relu-batchnorm-layer name=tdnn1 dim=512 input=Append(input@-2,input@-1,input@0,input@1,input@2)
#   relu-batchnorm-layer name=tdnn2 dim=512 input=Append(tdnn1@-2,tdnn1@0,tdnn1@2)
#   relu-batchnorm-layer name=tdnn3 dim=512 input=Append(tdnn2@-3,tdnn2@0,tdnn2@3)
#   relu-batchnorm-layer name=tdnn4 dim=512 input=tdnn3
# 
#   # am part
#   relu-batchnorm-layer name=tdnn5_am dim=512 input=tdnn4
#   relu-batchnorm-layer name=tdnn6_am dim=512 input=tdnn5_am
#   relu-batchnorm-layer name=tdnn7_am dim=128 input=tdnn6_am
#   output-layer name=output_am dim=$num_senones max-change=1.5 input=tdnn7_am
# 
#   # xvector part
#   relu-batchnorm-layer name=tdnn5_xvec dim=1500 input=Append(tdnn4,tdnn7_am)
#   stats-layer name=stats_xvec config=mean+stddev(0:1:1:${max_chunk_size}) input=tdnn5_xvec
#   relu-batchnorm-layer name=tdnn6_xvec dim=512 input=stats_xvec
#   relu-batchnorm-layer name=tdnn7_xvec dim=512 input=tdnn6_xvec
#   output-layer name=output_xvec include-log-softmax=true dim=$num_speakers input=tdnn7_xvec
# EOF

mkdir -p $nnetdir/configs_am $nnetdir/configs_xvec
# For two different tasks, the context is different. We create two branches and initialize 
# both of them to get their context respectively.
echo "processing Xvector nnet"
sed "s/output_xvec/output/g" $nnetdir/configs/network.xconfig.old > $nnetdir/configs_xvec/network.xconfig
steps/nnet3/xconfig_to_configs.py --xconfig-file $nnetdir/configs_xvec/network.xconfig --config-dir $nnetdir/configs_xvec

echo "processing AM nnet"
sed "s/output_am/output/g" $nnetdir/configs/network.xconfig.old > $nnetdir/configs_am/network.xconfig
steps/nnet3/xconfig_to_configs.py --xconfig-file $nnetdir/configs_am/network.xconfig --config-dir $nnetdir/configs_am

echo "process Cvector nnet"
sed "s/output_xvec/output/g" $nnetdir/configs/network.xconfig.old > $nnetdir/configs/network.xconfig
steps/nnet3/xconfig_to_configs.py --xconfig-file $nnetdir/configs/network.xconfig --config-dir $nnetdir/configs

cp $nnetdir/configs_xvec/vars $nnetdir/configs/vars_xvec
cp $nnetdir/configs_am/vars $nnetdir/configs/vars_am
cp $nnetdir/configs/final.config $nnetdir/nnet.config
echo "$max_chunk_size" > $nnetdir/max_chunk_size
echo "$min_chunk_size" > $nnetdir/min_chunk_size




