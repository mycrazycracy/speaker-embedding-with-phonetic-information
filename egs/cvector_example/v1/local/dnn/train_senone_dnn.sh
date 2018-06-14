#!/bin/bash

# This script is based on egs/fisher_english/s5/run.sh. It trains a
# multisplice time-delay neural network used in the DNN-based speaker
# recognition recipes.

# It's best to run the commands in this one by one.
stage=0
train_stage=-10
srand=0

. cmd.sh
. path.sh
. utils/parse_options.sh


data=$1
exp=$2
mfccdir=$3
nnetdir=$4

export tmpdir=$data/local/data
export data_links=$data/local/data/links
set -e

mkdir -p $tmpdir

# Prepare fisher data for ASR training
if [ $stage -le 0 ]; then
  local/dnn/fisher_data_prep.sh --calldata --stage 0 --data $data /mnt/lv10/person/sre16/data/fisher 
  
  local/dnn/fisher_prepare_dict.sh $data
  
  utils/prepare_lang.sh $data/local/dict "<unk>" $data/local/lang $data/lang
  
  local/dnn/fisher_train_lms.sh $data
  local/dnn/fisher_create_test_lang.sh $data

  utils/fix_data_dir.sh $data/train_all_asr
fi

# Extract features for ASR system training
if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --nj $fea_nj --cmd "$train_cmd" --mfcc-config conf/mfcc_asr.conf \
     $data/train_all_asr $exp/make_mfcc/train_all_asr $mfccdir || exit 1;
  
  utils/fix_data_dir.sh $data/train_all_asr
  utils/validate_data_dir.sh $data/train_all_asr
fi

# Change the format to be consistence with the other data,
# And filter out the training subset.
if [ $stage -le 2 ]; then
  # Some errors (i.e. genders) contains in the original dataset, so the filter operation
  # would lead to smaller subsets, i.e. num of speakers less than 5000 in the training set.

  local/fix_fisher_dir.sh $data/train_all_asr
  local/filter_fisher_dir.sh $data/train $data/train_all_asr $data/train_asr
  utils/fix_data_dir.sh $data/train_asr
#  local/filter_fisher_dir.sh $data/enroll $data/train_all_asr $data/enroll_asr
#  utils/fix_data_dir.sh $data/enroll_asr
#  local/filter_fisher_dir.sh $data/test $data/train_all_asr $data/test_asr
#  utils/fix_data_dir.sh $data/test_asr
fi

if [ $stage -le 3 ]; then
  steps/compute_cmvn_stats.sh $data/train_asr $exp/make_mfcc/train_asr $mfccdir
#  steps/compute_cmvn_stats.sh $data/enroll_asr $exp/make_mfcc/enroll_asr $mfccdir
#  steps/compute_cmvn_stats.sh $data/test_asr $exp/make_mfcc/test_asr $mfccdir
fi

# Split some subset to acclerate the train
if [ $stage -le 4 ]; then
  utils/subset_data_dir.sh $data/train_asr 10000 $data/train_asr_10k
  local/dnn/remove_dup_utts.sh 100 $data/train_asr_10k $data/train_asr_10k_nodup
  utils/subset_data_dir.sh --speakers $data/train_asr 30000 $data/train_asr_30k
fi

# Train from mono-phone to LDA+fMLLR
if [ $stage -le 5 ]; then
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
    $data/train_asr_10k_nodup $data/lang $exp/mono0a
  
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr_30k $data/lang $exp/mono0a $exp/mono0a_ali || exit 1;
  
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 $data/train_asr_30k $data/lang $exp/mono0a_ali $exp/tri1 || exit 1;
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr_30k $data/lang $exp/tri1 $exp/tri1_ali || exit 1;
  
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 $data/train_asr_30k $data/lang $exp/tri1_ali $exp/tri2 || exit 1;
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr $data/lang $exp/tri2 $exp/tri2_ali || exit 1;
  
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 40000 $data/train_asr $data/lang $exp/tri2_ali $exp/tri3a || exit 1;
fi

if [ $stage -le 8 ]; then
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr $data/lang $exp/tri3a $exp/tri3a_ali || exit 1;
  
  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 100000 $data/train_asr $data/lang $exp/tri3a_ali $exp/tri4a || exit 1;
  
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr $data/lang $exp/tri4a $exp/tri4a_ali || exit 1;

#  # No need to train more
#  steps/train_sat.sh  --cmd "$train_cmd" \
#    7000 300000 $data/train_asr $data/lang $exp/tri4a_ali $exp/tri5a || exit 1;
fi

# To control the number of senones, we have to train a new HMM with proper leaves
if [ $stage -le 9 ]; then
  steps/train_sat.sh  --cmd "$train_cmd" \
    3000 40000 $data/train_asr $data/lang $exp/tri4a_ali $exp/tri5 || exit 1;
  
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr $data/lang $exp/tri5 $exp/tri5_ali || exit 1;
fi

# Extract high-resolution MFCCs 
if [ $stage -le 10 ]; then
  utils/copy_data_dir.sh $data/train_asr $data/train_hires_asr
  steps/make_mfcc.sh --nj $fea_nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" $data/train_hires_asr $exp/make_hires/train_asr $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh $data/train_hires_asr $exp/make_hires/train_asr $mfccdir
fi


# We have around 4000 senones until here
# Train TDNN using the lastest system
gmm_dir=$exp/tri5
if [ $stage -le 11 ]; then
  mkdir -p $nnetdir
  echo "$0: creating neural net configs using the xconfig parser";

  feat_dim=$(feat-to-dim scp:$data/train_hires_asr/feats.scp -) || exit 1
  num_targets=$(tree-info $gmm_dir/tree |grep num-pdfs|awk '{print $2}') || exit 1
  echo "num of targets: $num_targets"

  mkdir -p $nnetdir/configs
  cat <<EOF > $nnetdir/configs/network.xconfig
  input dim=$feat_dim name=input

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=512 input=Append(-2,-1,0,1,2)
  relu-renorm-layer name=tdnn2 dim=512 input=Append(-2,0,2)
  relu-renorm-layer name=tdnn3 dim=512 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn4 dim=512 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=512 input=Append(-6,-3,0)
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs_new.py --xconfig-file $nnetdir/configs/network.xconfig --config-dir $nnetdir/configs/
fi

# check the num of parameters before continue...
# exit 1

if [ $stage -le 12 ]; then
  # We should not give the egs_dir for the first time we run it
  # it does not exist
  # egs_dir=$nnetdir/egs
  steps/nnet3/train_dnn_new.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=3 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --egs.dir=$egs_dir \
    --cleanup.remove-egs=false \
    --cleanup.preserve-model-interval=50 \
    --use-gpu=true \
    --feat-dir=$data/train_hires_asr \
    --ali-dir=$exp/tri5_ali \
    --lang=$data/lang \
    --dir=$nnetdir  || exit 1;

#  # Here is another version (with cmvn)
#  mkdir -p ${nnetdir}_cmvn/configs
#  steps/nnet3/xconfig_to_configs_new.py --xconfig-file $nnetdir/configs/network.xconfig --config-dir ${nnetdir}_cmvn/configs/
#  # egs_dir=${nnetdir}_cmvn/egs
#  steps/nnet3/train_dnn_new.py --stage=$train_stage \
#    --cmd="$train_cmd" \
#    --feat.cmvn-opts="--norm-means=true --norm-vars=false" \
#    --trainer.srand=$srand \
#    --trainer.max-param-change=2.0 \
#    --trainer.num-epochs=3 \
#    --trainer.samples-per-iter=400000 \
#    --trainer.optimization.num-jobs-initial=2 \
#    --trainer.optimization.num-jobs-final=4 \
#    --trainer.optimization.initial-effective-lrate=0.0015 \
#    --trainer.optimization.final-effective-lrate=0.00015 \
#    --trainer.optimization.minibatch-size=256,128 \
#    --egs.dir=$egs_dir \
#    --cleanup.remove-egs=false \
#    --cleanup.preserve-model-interval=50 \
#    --use-gpu=true \
#    --feat-dir=$data/train_hires_asr \
#    --ali-dir=$exp/tri4a_ali \
#    --lang=$data/lang \
#    --dir=${nnetdir}_cmvn  || exit 1;
fi


#  sid/nnet2/train_multisplice_accel2.sh --stage $train_stage \
#    --feat-type raw \
#    --splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2" \
#    --num-epochs 6 \
#    --num-hidden-layers 6 \
#    --num-jobs-initial 3 --num-jobs-final 8 \
#    --num-threads "$num_threads" \
#    --minibatch-size "$minibatch_size" \
#    --parallel-opts "$parallel_opts" \
#    --mix-up 10500 \
#    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
#    --cmd "$decode_cmd" \
#    --egs-dir "$common_egs_dir" \
#    --pnorm-input-dim 3500 \
#    --pnorm-output-dim 350 \
#    data/train_hires_asr data/lang exp/tri5a $dir  || exit 1;



