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

nnet=$exp/nnet2_online/nnet_ms_a/final.mdl

stage=0

if [ $stage -le 0 ]; then
  # Extract DNN features.
  cp -r $data/sre16_major $data/sre16_major_dnn
  cp -r $data/sre16_minor $data/sre16_minor_dnn
  cp -r $data/sre16_eval_enroll $data/sre16_eval_enroll_dnn
  cp -r $data/sre16_eval_test $data/sre16_eval_test_dnn
  
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre16_major_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre16_minor_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre16_eval_enroll_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre16_eval_test_dnn $exp/make_mfcc $mfccdir
  
  for name in sre16_major_dnn sre16_minor_dnn sre16_eval_enroll_dnn sre16_eval_test_dnn; do
      utils/fix_data_dir.sh $data/${name}
  done
  
  for name in sre16_major sre16_minor sre16_eval_enroll sre16_eval_test; do
    cp $data/${name}/vad.scp $data/${name}_dnn/vad.scp
    cp $data/${name}/utt2spk $data/${name}_dnn/utt2spk
    cp $data/${name}/spk2utt $data/${name}_dnn/spk2utt
    utils/fix_data_dir.sh $data/${name}_dnn
  done
fi

if [ $stage -le 2 ]; then
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre16_major \
    $data/sre16_major_dnn \
    $exp/ivectors_sre16_major_nnet2
  
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre16_minor \
    $data/sre16_minor_dnn \
    $exp/ivectors_sre16_minor_nnet2
  
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre16_eval_enroll \
    $data/sre16_eval_enroll_dnn \
    $exp/ivectors_sre16_eval_enroll_nnet2
  
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre16_eval_test \
    $data/sre16_eval_test_dnn \
    $exp/ivectors_sre16_eval_test_nnet2
fi 

if [ $stage -le 3 ]; then
  # LDA + PLDA
  lda_dim=200
  
  $train_cmd $exp/ivectors_sre16_major_nnet2/log/compute_mean.log \
    ivector-mean scp:$exp/ivectors_sre16_major_nnet2/ivector.scp \
    $exp/ivectors_sre16_major_nnet2/mean.vec || exit 1;
  
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $exp/ivectors_sre_nnet2/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre_nnet2/ivector.scp ark:- |" \
    ark:$data/sre/utt2spk $exp/ivectors_sre_nnet2/transform.mat || exit 1;
  
  #  Train the PLDA model.
  $train_cmd $exp/ivectors_sre_nnet2/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre_nnet2/ivector.scp ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $exp/ivectors_sre_nnet2/plda_lda${lda_dim} || exit 1;
  
  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation.
  $train_cmd $exp/ivectors_sre16_major_nnet2/log/plda_lda${lda_dim}_sre16_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $exp/ivectors_sre_nnet2/plda_lda${lda_dim} \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre16_major_nnet2/ivector.scp ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $exp/ivectors_sre16_major_nnet2/plda_lda${lda_dim}_sre16_adapt || exit 1;
  
  
  # Get results using the out-of-domain PLDA model
  $train_cmd $exp/ivector_scores_nnet2/log/sre16_eval.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre16_eval_enroll_nnet2/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre_nnet2/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre16_eval_enroll/spk2utt scp:$exp/ivectors_sre16_eval_enroll_nnet2/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre16_major_nnet2/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre16_major_nnet2/mean.vec scp:$exp/ivectors_sre16_eval_test_nnet2/ivector.scp ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores_nnet2/sre16_eval_scores || exit 1;
  
  utils/filter_scp.pl $sre16_trials_tgl $exp/ivector_scores_nnet2/sre16_eval_scores > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores
  utils/filter_scp.pl $sre16_trials_yue $exp/ivector_scores_nnet2/sre16_eval_scores > $exp/ivector_scores_nnet2/sre16_eval_yue_scores
  pooled_eer=$(paste $sre16_trials $exp/ivector_scores_nnet2/sre16_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl $exp/ivector_scores_nnet2/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue $exp/ivector_scores_nnet2/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"
  
  paste $sre16_trials $exp/ivector_scores_nnet2/sre16_eval_scores | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre16_eval_scores.new
  grep ' target$' $exp/ivector_scores_nnet2/sre16_eval_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_scores.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre16_eval_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_scores.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre16_eval_scores.target', '$exp/ivector_scores_nnet2/sre16_eval_scores.nontarget', '$exp/ivector_scores_nnet2/sre16_eval_scores.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindow > /dev/nulls
  cd -
  rm -f $exp/ivector_scores_nnet2/sre16_eval_scores.new $exp/ivector_scores_nnet2/sre16_eval_scores.target $exp/ivector_scores_nnet2/sre16_eval_scores.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre16_eval_scores.result
  
  paste $sre16_trials_tgl $exp/ivector_scores_nnet2/sre16_eval_tgl_scores | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.new
  grep ' target$' $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre16_eval_tgl_scores.target', '$exp/ivector_scores_nnet2/sre16_eval_tgl_scores.nontarget', '$exp/ivector_scores_nnet2/sre16_eval_tgl_scores.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindow > /dev/nulls
  cd -
  rm -f $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.new $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.target $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre16_eval_tgl_scores.result
  
  paste $sre16_trials_yue $exp/ivector_scores_nnet2/sre16_eval_yue_scores | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre16_eval_yue_scores.new
  grep ' target$' $exp/ivector_scores_nnet2/sre16_eval_yue_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_yue_scores.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre16_eval_yue_scores.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_yue_scores.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre16_eval_yue_scores.target', '$exp/ivector_scores_nnet2/sre16_eval_yue_scores.nontarget', '$exp/ivector_scores_nnet2/sre16_eval_yue_scores.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindow > /dev/nulls
  cd -
  rm -f $exp/ivector_scores_nnet2/sre16_eval_yue_scores.new $exp/ivector_scores_nnet2/sre16_eval_yue_scores.target $exp/ivector_scores_nnet2/sre16_eval_yue_scores.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre16_eval_yue_scores.result
  
  # Get results using an adapted PLDA model. In the future we'll replace
  # this (or add to this) with a clustering based approach to PLDA adaptation.
  $train_cmd $exp/ivector_scores_nnet2/log/sre16_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre16_eval_enroll_nnet2/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre16_major_nnet2/plda_lda${lda_dim}_sre16_adapt - |" \
    "ark:ivector-mean ark:$data/sre16_eval_enroll/spk2utt scp:$exp/ivectors_sre16_eval_enroll_nnet2/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre16_major_nnet2/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre16_major_nnet2/mean.vec scp:$exp/ivectors_sre16_eval_test_nnet2/ivector.scp ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores_nnet2/sre16_eval_scores_adapt || exit 1;
  
  utils/filter_scp.pl $sre16_trials_tgl $exp/ivector_scores_nnet2/sre16_eval_scores_adapt > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt
  utils/filter_scp.pl $sre16_trials_yue $exp/ivector_scores_nnet2/sre16_eval_scores_adapt > $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt
  pooled_eer=$(paste $sre16_trials $exp/ivector_scores_nnet2/sre16_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"
  
  paste $sre16_trials $exp/ivector_scores_nnet2/sre16_eval_scores_adapt | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.new
  grep ' target$' $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre16_eval_scores_adapt.target', '$exp/ivector_scores_nnet2/sre16_eval_scores_adapt.nontarget', '$exp/ivector_scores_nnet2/sre16_eval_scores_adapt.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.new $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.target $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre16_eval_scores_adapt.result
  
  paste $sre16_trials_tgl $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.new
  grep ' target$' $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.target', '$exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.nontarget', '$exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.new $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.target $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre16_eval_tgl_scores_adapt.result
  
  paste $sre16_trials_yue $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.new
  grep ' target$' $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.target', '$exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.nontarget', '$exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.new $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.target $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre16_eval_yue_scores_adapt.result
fi

