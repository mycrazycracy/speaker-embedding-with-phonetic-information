#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.
#
# Pretrained models are available for this recipe.
# See http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

. ./cmd.sh
. ./path.sh
set -e

fea_nj=16
train_nj=16

nnet_extract_gpu=false
nnet_nj=16

root=/mnt/lv10/person/liuyi/fisher
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc
nnet_dir=$exp/cvector_nnet_1a
amdir=$nnetdir/am

trials=$data/test/trials

stage=0

if [ $stage -le 0 ]; then
  # Extract features
  steps/make_mfcc.sh --nj $fea_nj --cmd "$train_cmd" --mfcc-config conf/mfcc.conf \
    $data/train_background $exp/make_mfcc/train_background $mfccdir || exit 1;
  utils/fix_data_dir.sh $data/train_background
  sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" --vad-config conf/vad.conf \
    $data/train_background $exp/make_vad/train_background $vaddir
  utils/fix_data_dir.sh $data/train_background

  steps/make_mfcc.sh --nj $fea_nj --cmd "$train_cmd" --mfcc-config conf/mfcc.conf \
    $data/enroll $exp/make_mfcc/enroll $mfccdir || exit 1;
  utils/fix_data_dir.sh $data/enroll
  sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" --vad-config conf/vad.conf \
    $data/enroll $exp/make_vad/enroll $vaddir
  utils/fix_data_dir.sh $data/enroll

  steps/make_mfcc.sh --nj $fea_nj --cmd "$train_cmd" --mfcc-config conf/mfcc.conf \
    $data/test $exp/make_mfcc/test $mfccdir || exit 1;
  utils/fix_data_dir.sh $data/test
  sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" --vad-config conf/vad.conf \
    $data/test $exp/make_vad/test $vaddir
  utils/fix_data_dir.sh $data/test
fi

if [ $stage -le 1 ]; then
  # Train ASR system that generate the frame alignments
  # For demonstration, we use SWBD-1 as the source of phonetic information 
  # The asr model will be in $exp/tri4_swbd and alignments in exp/tri4_swbd_ali_nodup
  local/dnn/train_asr_swbd.sh --stage 0 $data $exp $mfccdir

  # Extract speaker features for data/train_swbd_asr_nodup
  utils/copy_data_dir_new.sh $data/train_swbd_asr_nodup $data/train_swbd_nodup
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $fea_nj --cmd "$train_cmd" \
    $data/train_swbd_nodup $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/train_swbd_nodup
  sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" \
    $data/train_swbd_nodup $exp/make_vad $vaddir
  utils/fix_data_dir.sh $data/train_swbd_nodup
fi


if [ $stage -le 2 ]; then
  # Prepare features for xvector egs
  utils/copy_data_dir_new.sh $data/train_background $data/train_background_vad

  # This script applies cvmn to the original data, and remove the silence in the feature (using vad.scp).
  # If ali-dir is given, the silence is also removed from the alignments (using vad.scp as well).
  # Speakers with too few segments (min-num-utts) and too short segments (min-len) is also removed,
  # since they are not suitable for training as indicated in egs/SRE16/V2 
  # Note that this takes extra space than we need, since we explore other setup in the previous experiments.
  # You may want to rewrite the script or delete unused files when you know what you are doing.
  sid/nnet3_cvector/cvector/prepare_feats.sh --cmd "$train_cmd" \
    --nj $fea_nj \
    --vad-dir $data/train_background_vad \
    --min-len 200 \
    --min-num-utts 8 \
    $data/train_background \
    $data/train_background_cmvn \
    $exp/train_background_cmvn \
    $data/train_background_cmvn_nosil \
    $exp/train_background_cmvn_nosil

  # Prepare features for phonetic egs (with ali)
  utils/copy_data_dir_new.sh $data/train_swbd_nodup $data/train_swbd_nodup_vad
  sid/nnet3_cvector/cvector/prepare_feats.sh --cmd "$train_cmd" \
    --nj $fea_nj \
    --vad-dir $data/train_swbd_nodup_vad \
    --ali-dir $exp/tri4_swbd_ali_nodup \
    --min-len 100 \
    --min-num-utts 3 \
    $data/train_swbd_nodup \
    $data/train_swbd_nodup_cmvn \
    $exp/train_swbd_nodup_cmvn \
    $data/train_swbd_nodup_cmvn_nosil \
    $exp/train_swbd_nodup_cmvn_nosil
fi


if [ $stage -le 3 ]; then
  # Use the speaker features to train phonetic-discriminant network 
  # Silence is removed from the training data
  # As stated in V1, I'm not sure whether it is better to use train_swbd_nodup_cmvn_nosil to train this network. The result in the paper using the settings below.
  local/nnet3_cvector/cvector/train_am.sh --stage 0 --train_stage -10 \
    $data/train_swbd_nodup_cmvn_nosil \
    $exp/train_swbd_nodup_cmvn_nosil \
    $data/lang_swbd \
    $amdir
fi
 

if [ $stage -le 4 ]; then
  # Train the x-vector network using phonetic bottleneck features extracted from phonetic network
  local/nnet3_cvector/cvector/train_xvector_with_am.sh --stage 0 --train-stage -10 \
    --am-lr-factor 0.2 \
    tdnn5.batchnorm $amdir/final.raw \
    $data/train_background_cmvn_nosil \
    $nnetdir
fi


if [ $stage -le 5 ]; then
  # Extract cvectors
  sid/nnet3_cvector/cvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu $nnet_extract_gpu --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/train_background \
    $exp/cvectors_train_background
  
  sid/nnet3_cvector/cvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu $nnet_extract_gpu --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/enroll \
    $exp/cvectors_enroll
  
  sid/nnet3_cvector/cvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu $nnet_extract_gpu --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/test \
    $exp/cvectors_test
fi


if [ $stage -le 6 ]; then
  # The last thing is scoring
  # LDA + PLDA
  lda_dim=150
  
  $train_cmd $exp/cvectors_train_background/log/compute_mean.log \
    ivector-mean scp:$exp/cvectors_train_background/xvector_train_background.scp $exp/cvectors_train_background/mean.vec || exit 1;
  
  $train_cmd $exp/cvectors_train_background/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$exp/cvectors_train_background/xvector_train_background.scp ark:- |" \
    ark:$data/train/utt2spk $exp/cvectors_train_background/transform.mat || exit 1;
  
  #  Train the PLDA model.
  $train_cmd $exp/cvectors_train_background/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$exp/cvectors_train_background/xvector_train_background.scp ark:- | transform-vec $exp/cvectors_train_background/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $exp/cvectors_train_background/plda_lda${lda_dim} || exit 1;
  
  $train_cmd $exp/cvector_scores/log/fisher_test.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/cvectors_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/cvectors_train_background/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/enroll/spk2utt scp:$exp/cvectors_enroll/xvector_enroll.scp ark:- | ivector-subtract-global-mean $exp/cvectors_train_background/mean.vec ark:- ark:- | transform-vec $exp/cvectors_train_background/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/cvectors_train_background/mean.vec scp:$exp/cvectors_test/xvector_test.scp ark:- | transform-vec $exp/cvectors_train_background/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $exp/cvector_scores/fisher_test || exit 1;
   
  python utils/recover_scores.py $trials $exp/cvector_scores/fisher_test > $exp/cvector_scores/fisher_test.rec
  eer=$(paste $trials $exp/cvector_scores/fisher_test.rec | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: ${eer}%"
  
  
  # Cosine scoring
  $train_cmd $exp/cvector_scores/log/fisher_test_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll/spk2utt scp:$exp/cvectors_enroll/xvector_enroll.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$exp/cvectors_test/xvector_test.scp ark:- |" \
    $exp/cvector_scores/fisher_test_cos
  
  python utils/recover_scores.py $trials $exp/cvector_scores/fisher_test_cos > $exp/cvector_scores/fisher_test_cos.rec
  eer=$(paste $trials $exp/cvector_scores/fisher_test_cos.rec | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: ${eer}%"
  
  # LDA + Cosine scoring
  $train_cmd $exp/cvector_scores/log/fisher_test_lda_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll/spk2utt scp:$exp/cvectors_enroll/xvector_enroll.scp ark:- | ivector-subtract-global-mean $exp/cvectors_train_background/mean.vec ark:- ark:- | transform-vec $exp/cvectors_train_background/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/cvectors_train_background/mean.vec scp:$exp/cvectors_test/xvector_test.scp ark:- | transform-vec $exp/cvectors_train_background/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $exp/cvector_scores/fisher_test_lda_cos
  
  python utils/recover_scores.py $trials $exp/cvector_scores/fisher_test_lda_cos > $exp/cvector_scores/fisher_test_lda_cos.rec
  eer=$(paste $trials $exp/cvector_scores/fisher_test_lda_cos.rec | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: ${eer}%"
fi


