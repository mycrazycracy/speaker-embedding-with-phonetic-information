#!/bin/bash

# This script is based on egs/fisher_english/s5/run.sh. It trains a
# multisplice time-delay neural network used in the DNN-based speaker
# recognition recipes.

# It's best to run the commands in this one by one.

. cmd.sh
. path.sh
set -e


if [ $stage -le 0 ]; then
  # the next command produces the data in data/train_all, and we move it to train_all_asr
  local/fisher_data_prep.sh --calldata $data/train_fisher /mnt/lv10/person/sre16/data/fisher
  local/fisher_prepare_dict.sh
  utils/prepare_lang.sh $data/local/dict "<unk>" $data/local/lang $data/lang
  local/fisher_train_lms.sh $data
  local/fisher_create_test_lang.sh

  mv $data/train_all $data/train_all_asr
  utils/fix_data_dir.sh $data/train_all_asr
fi 

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --nj $fea_nj --cmd "$train_cmd" --mfcc-config conf/mfcc_asr.conf \
     $data/train_all_asr $exp/make_mfcc/train_all_asr $mfccdir || exit 1;
  utils/fix_data_dir.sh $data/train_all_asr
  utils/validate_data_dir.sh $data/train_all_asr
fi 

if [ $stage -le 2 ]; then
  utils/subset_data_dir.sh --first $data/train_all_asr 10000 $data/dev_and_test_asr
  utils/subset_data_dir.sh --first $data/dev_and_test_asr 5000 $data/dev_asr
  utils/subset_data_dir.sh --last $data/dev_and_test_asr 5000 $data/test_asr
  rm -r $data/dev_and_test_asr
fi

if [ $stage -le 3 ]; then
  steps/compute_cmvn_stats.sh $data/dev_asr $exp/make_mfcc/dev_asr $mfccdir
  steps/compute_cmvn_stats.sh $data/test_asr $exp/make_mfcc/test_asr $mfccdir
  
  n=$[`cat $data/train_all_asr/segments | wc -l` - 10000]
  utils/subset_data_dir.sh --last $data/train_all_asr $n $data/train_asr
  steps/compute_cmvn_stats.sh $data/train_asr $exp/make_mfcc/train_asr $mfccdir
fi

# Now-- there are 1.6 million utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.
if [ $stage -le 4 ]; then
  utils/subset_data_dir.sh --shortest $data/train_asr 100000 $data/train_asr_100kshort
  utils/subset_data_dir.sh  $data/train_asr_100kshort 10000 $data/train_asr_10k
  utils/data/remove_dup_utts.sh 100 $data/train_asr_10k $data/train_asr_10k_nodup
  utils/subset_data_dir.sh --speakers $data/train_asr 30000 $data/train_asr_30k
  utils/subset_data_dir.sh --speakers $data/train_asr 100000 $data/train_asr_100k
fi 

if [ $stage -le 5 ]; then
  steps/train_mono.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr_10k_nodup $data/lang $exp/mono0a
  
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr_30k $data/lang $exp/mono0a $exp/mono0a_ali || exit 1;
fi

if [ $stage -le 6 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
      2500 20000 $data/train_asr_30k $data/lang $exp/mono0a_ali $exp/tri1 || exit 1;
  
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr_30k $data/lang $exp/tri1 $exp/tri1_ali || exit 1;
  
  steps/train_deltas.sh --cmd "$train_cmd" \
      2500 20000 $data/train_asr_30k $data/lang $exp/tri1_ali $exp/tri2 || exit 1;
  
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr_100k $data/lang $exp/tri2 $exp/tri2_ali || exit 1;
fi

if [ $stage -le 7 ]; then
  # Train tri3a, which is LDA+MLLT, on 100k data.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5000 40000 $data/train_asr_100k $data/lang $exp/tri2_ali $exp/tri3a || exit 1;
  
  # Next we'll use fMLLR and train with SAT (i.e. on
  # fMLLR features)
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr_100k $data/lang $exp/tri3a $exp/tri3a_ali || exit 1;
fi 

if [ $stage -le 8 ]; then
  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 100000 $data/train_asr_100k $data/lang $exp/tri3a_ali $exp/tri4a || exit 1;
  
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_asr $data/lang $exp/tri4a $exp/tri4a_ali || exit 1;
  
  steps/train_sat.sh  --cmd "$train_cmd" --stage 20 \
    3000 100000 $data/train_asr $data/lang $exp/tri4a_ali $exp/tri5a || exit 1;
fi

# The following is based on an older nnet2 recipe.
local/dnn/run_nnet2_multisplice.sh --stage -10 --train-stage -10
