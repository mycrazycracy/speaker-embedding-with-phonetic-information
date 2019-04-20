#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using ivectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.

. ./cmd.sh
. ./path.sh
set -e

fea_nj=16
train_nj=24

root=/mnt/lv10/person/liuyi/sre/
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc

# we won't use the noise data to do data augmentation
rirs_noises=/mnt/lv10/person/liuyi/ly_database/RIRS_NOISES/
musan=/mnt/lv10/person/liuyi/ly_database/musan/

data_root=/mnt/lv10/person/liuyi/ly_list/sre16_kaldi_list/
sre16_trials=/mnt/lv10/person/liuyi/sre16/data/sre16_eval_test/trials
sre16_trials_tgl=/mnt/lv10/person/liuyi/sre16/data/sre16_eval_test/trials_tgl
sre16_trials_yue=/mnt/lv10/person/liuyi/sre16/data/sre16_eval_test/trials_yue

stage=0

# Train UBM and i-vector extractor using recipe v1/run_sre10.sh

if [ $stage -le 0 ]; then
  for name in sre16_major sre16_minor sre16_eval_enroll sre16_eval_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/$name
    sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/$name
  done
fi 

if [ $stage -le 1 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre16_major \
    $exp/ivectors_sre16_major
  
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre16_minor \
    $exp/ivectors_sre16_minor
  
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre16_eval_enroll \
    $exp/ivectors_sre16_eval_enroll
  
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre16_eval_test \
    $exp/ivectors_sre16_eval_test
fi

if [ $stage -le 2 ]; then
  # LDA + PLDA
  lda_dim=200
  
  $train_cmd $exp/ivectors_sre16_major/log/compute_mean.log \
    ivector-mean scp:$exp/ivectors_sre16_major/ivector.scp \
    $exp/ivectors_sre16_major/mean.vec || exit 1;
  
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $exp/ivectors_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre/ivector.scp ark:- |" \
    ark:$data/sre/utt2spk $exp/ivectors_sre/transform.mat || exit 1;
  
  #  Train the PLDA model.
  $train_cmd $exp/ivectors_sre/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre/ivector.scp ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $exp/ivectors_sre/plda_lda${lda_dim} || exit 1;
  
  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation.
  $train_cmd $exp/ivectors_sre16_major/log/plda_lda${lda_dim}_sre16_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $exp/ivectors_sre/plda_lda${lda_dim} \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre16_major/ivector.scp ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $exp/ivectors_sre16_major/plda_lda${lda_dim}_sre16_adapt || exit 1;
  
  
  # Get results using the out-of-domain PLDA model
  $train_cmd $exp/ivector_scores/log/sre16_eval.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre16_eval_enroll/spk2utt scp:$exp/ivectors_sre16_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre16_major/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre16_major/mean.vec scp:$exp/ivectors_sre16_eval_test/ivector.scp ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores/sre16_eval_scores || exit 1;
  
  utils/filter_scp.pl $sre16_trials_tgl $exp/ivector_scores/sre16_eval_scores > $exp/ivector_scores/sre16_eval_tgl_scores
  utils/filter_scp.pl $sre16_trials_yue $exp/ivector_scores/sre16_eval_scores > $exp/ivector_scores/sre16_eval_yue_scores
  pooled_eer=$(paste $sre16_trials $exp/ivector_scores/sre16_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl $exp/ivector_scores/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue $exp/ivector_scores/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"
  
  paste $sre16_trials $exp/ivector_scores/sre16_eval_scores | awk '{print $6, $3}' > $exp/ivector_scores/sre16_eval_scores.new
  grep ' target$' $exp/ivector_scores/sre16_eval_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_scores.target
  grep ' nontarget$' $exp/ivector_scores/sre16_eval_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_scores.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre16_eval_scores.target', '$exp/ivector_scores/sre16_eval_scores.nontarget', '$exp/ivector_scores/sre16_eval_scores.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre16_eval_scores.new $exp/ivector_scores/sre16_eval_scores.target $exp/ivector_scores/sre16_eval_scores.nontarget
  tail -n 1 $exp/ivector_scores/sre16_eval_scores.result
  
  paste $sre16_trials_tgl $exp/ivector_scores/sre16_eval_tgl_scores | awk '{print $6, $3}' > $exp/ivector_scores/sre16_eval_tgl_scores.new
  grep ' target$' $exp/ivector_scores/sre16_eval_tgl_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_tgl_scores.target
  grep ' nontarget$' $exp/ivector_scores/sre16_eval_tgl_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_tgl_scores.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre16_eval_tgl_scores.target', '$exp/ivector_scores/sre16_eval_tgl_scores.nontarget', '$exp/ivector_scores/sre16_eval_tgl_scores.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre16_eval_tgl_scores.new $exp/ivector_scores/sre16_eval_tgl_scores.target $exp/ivector_scores/sre16_eval_tgl_scores.nontarget
  tail -n 1 $exp/ivector_scores/sre16_eval_tgl_scores.result
  
  paste $sre16_trials_yue $exp/ivector_scores/sre16_eval_yue_scores | awk '{print $6, $3}' > $exp/ivector_scores/sre16_eval_yue_scores.new
  grep ' target$' $exp/ivector_scores/sre16_eval_yue_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_yue_scores.target
  grep ' nontarget$' $exp/ivector_scores/sre16_eval_yue_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_yue_scores.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre16_eval_yue_scores.target', '$exp/ivector_scores/sre16_eval_yue_scores.nontarget', '$exp/ivector_scores/sre16_eval_yue_scores.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre16_eval_yue_scores.new $exp/ivector_scores/sre16_eval_yue_scores.target $exp/ivector_scores/sre16_eval_yue_scores.nontarget
  tail -n 1 $exp/ivector_scores/sre16_eval_yue_scores.result
  
  
  # Get results using an adapted PLDA model. In the future we'll replace
  # this (or add to this) with a clustering based approach to PLDA adaptation.
  $train_cmd $exp/ivector_scores/log/sre16_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre16_major/plda_lda${lda_dim}_sre16_adapt - |" \
    "ark:ivector-mean ark:$data/sre16_eval_enroll/spk2utt scp:$exp/ivectors_sre16_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre16_major/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre16_major/mean.vec scp:$exp/ivectors_sre16_eval_test/ivector.scp ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores/sre16_eval_scores_adapt || exit 1;
  
  
  utils/filter_scp.pl $sre16_trials_tgl $exp/ivector_scores/sre16_eval_scores_adapt > $exp/ivector_scores/sre16_eval_tgl_scores_adapt
  utils/filter_scp.pl $sre16_trials_yue $exp/ivector_scores/sre16_eval_scores_adapt > $exp/ivector_scores/sre16_eval_yue_scores_adapt
  pooled_eer=$(paste $sre16_trials $exp/ivector_scores/sre16_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl $exp/ivector_scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue $exp/ivector_scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"
  
  
  paste $sre16_trials $exp/ivector_scores/sre16_eval_scores_adapt | awk '{print $6, $3}' > $exp/ivector_scores/sre16_eval_scores_adapt.new
  grep ' target$' $exp/ivector_scores/sre16_eval_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_scores_adapt.target
  grep ' nontarget$' $exp/ivector_scores/sre16_eval_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_scores_adapt.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre16_eval_scores_adapt.target', '$exp/ivector_scores/sre16_eval_scores_adapt.nontarget', '$exp/ivector_scores/sre16_eval_scores_adapt.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre16_eval_scores_adapt.new $exp/ivector_scores/sre16_eval_scores_adapt.target $exp/ivector_scores/sre16_eval_scores_adapt.nontarget
  tail -n 1 $exp/ivector_scores/sre16_eval_scores_adapt.result
  
  paste $sre16_trials_tgl $exp/ivector_scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' > $exp/ivector_scores/sre16_eval_tgl_scores_adapt.new
  grep ' target$' $exp/ivector_scores/sre16_eval_tgl_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_tgl_scores_adapt.target
  grep ' nontarget$' $exp/ivector_scores/sre16_eval_tgl_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_tgl_scores_adapt.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre16_eval_tgl_scores_adapt.target', '$exp/ivector_scores/sre16_eval_tgl_scores_adapt.nontarget', '$exp/ivector_scores/sre16_eval_tgl_scores_adapt.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre16_eval_tgl_scores_adapt.new $exp/ivector_scores/sre16_eval_tgl_scores_adapt.target $exp/ivector_scores/sre16_eval_tgl_scores_adapt.nontarget
  tail -n 1 $exp/ivector_scores/sre16_eval_tgl_scores_adapt.result
  
  paste $sre16_trials_yue $exp/ivector_scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' > $exp/ivector_scores/sre16_eval_yue_scores_adapt.new
  grep ' target$' $exp/ivector_scores/sre16_eval_yue_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_yue_scores_adapt.target
  grep ' nontarget$' $exp/ivector_scores/sre16_eval_yue_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre16_eval_yue_scores_adapt.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre16_eval_yue_scores_adapt.target', '$exp/ivector_scores/sre16_eval_yue_scores_adapt.nontarget', '$exp/ivector_scores/sre16_eval_yue_scores_adapt.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre16_eval_yue_scores_adapt.new $exp/ivector_scores/sre16_eval_yue_scores_adapt.target $exp/ivector_scores/sre16_eval_yue_scores_adapt.nontarget
  tail -n 1 $exp/ivector_scores/sre16_eval_yue_scores_adapt.result
fi

