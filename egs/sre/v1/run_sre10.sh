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

rirs_noises=/mnt/lv10/person/liuyi/ly_database/RIRS_NOISES/
musan=/mnt/lv10/person/liuyi/ly_database/musan/

gender=pool

data_root=/mnt/lv10/person/liuyi/ly_list/sre16_kaldi_list/
sre10_dir=$data_root/sre10_eval/
sre10_train_c5_ext=$sre10_dir/coreext_c5/enroll/$gender/
sre10_trials_c5_ext=$sre10_dir/coreext_c5/test/$gender/
sre10_train_10s=$sre10_dir/10sec/enroll/$gender/
sre10_trials_10s=$sre10_dir/10sec/test/$gender/

stage=0

## Data preparation
if [ $stage -le 0 ]; then
  # The data preparation is done using the script in egs/sre10 and egs/sre16.
  # You should finish the data preparation step in egs/sre10/v1/run.sh and egs/sre16/v2/run.sh 
  # before running this script.
  # After that, the SRE and SWBD data directories will be ready. We simply copy these directories and combine them as needed.
  # We exclude Mixer6, since Mixer6 is used in SRE10.

  # combine all sre data (04-08) (no Mixer6).
  utils/combine_data.sh $data/sre \
      $data_root/sre2004 $data_root/sre2005_train $data_root/sre2005_test \
      $data_root/sre2006_train $data_root/sre2006_test $data_root/sre08
  utils/validate_data_dir.sh --no-text --no-feats $data/sre
  utils/fix_data_dir.sh $data/sre
  
  # combine all swbd data
  utils/combine_data.sh $data/swbd \
      $data_root/swbd2_phase1_train $data_root/swbd2_phase2_train $data_root/swbd2_phase3_train \
      $data_root/swbd_cellular1_train $data_root/swbd_cellular2_train
  utils/validate_data_dir.sh --no-text --no-feats $data/swbd
  utils/fix_data_dir.sh $data/swbd
  
  # prepare sre10 evaluation data
  rm -rf $data/sre10_enroll_coreext_c5_$gender && cp -r $sre10_train_c5_ext $data/sre10_enroll_coreext_c5_$gender
  rm -rf $data/sre10_test_coreext_c5_$gender && cp -r $sre10_trials_c5_ext $data/sre10_test_coreext_c5_$gender
  rm -rf $data/sre10_enroll_10s_$gender && cp -r $sre10_train_10s $data/sre10_enroll_10s_$gender
  rm -rf $data/sre10_test_10s_$gender && cp -r $sre10_trials_10s $data/sre10_test_10s_$gender

  # prepare unlabeled Cantonese and Tagalog development data.
  rm -rf $data/sre16_major && cp -r $data_root/sre16_major $data/sre16_major
  rm -rf $data/sre16_minor && cp -r $data_root/sre16_minor $data/sre16_minor
fi

if [ $stage -le 1 ]; then
  # Make filterbanks and compute the energy-based VAD for each dataset
  # Note: MFCCs here is 20-dim, different with 23-dim in x-vector.
  for name in sre swbd sre10_enroll_coreext_c5_$gender sre10_test_coreext_c5_$gender sre10_enroll_10s_$gender sre10_test_10s_$gender; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/$name
    sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/$name
  done
  utils/combine_data.sh --extra-files "utt2num_frames" $data/swbd_sre $data/swbd $data/sre
  utils/fix_data_dir.sh $data/swbd_sre
fi 

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj $train_nj --num-threads 1  --subsample 1 \
    $data/swbd_sre 2048 \
    $exp/diag_ubm
  
  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj $train_nj --remove-low-count-gaussians false --subsample 1 \
    $data/swbd_sre \
    $exp/diag_ubm $exp/full_ubm
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
    --nj 2 --num-threads 1 --num-processes 16 \
    --ivector-dim 600 \
    --num-iters 5 \
    $exp/full_ubm/final.ubm $data/swbd_sre \
    $exp/extractor
fi

if [ $stage -le 4 ]; then
  # Extract i-vectors for SRE data (includes Mixer 6).
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/swbd_sre \
    $exp/ivectors_swbd_sre
  
  # extract i-vectors for SRE data 
  mkdir -p $exp/ivectors_sre
  utils/filter_scp.pl $data/sre/utt2spk $exp/ivectors_swbd_sre/ivector.scp > $exp/ivectors_sre/ivector.scp
  
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre10_enroll_coreext_c5_$gender \
    $exp/ivectors_sre10_enroll_coreext_c5_$gender
  
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre10_test_coreext_c5_$gender \
    $exp/ivectors_sre10_test_coreext_c5_$gender
  
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre10_enroll_10s_$gender \
    $exp/ivectors_sre10_enroll_10s_$gender
  
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $train_nj \
    $exp/extractor $data/sre10_test_10s_$gender \
    $exp/ivectors_sre10_test_10s_$gender
fi

  
if [ $stage -le 5 ]; then 
  # LDA + PLDA
  lda_dim=200
  
  $train_cmd $exp/ivectors_sre/log/compute_mean.log \
    ivector-mean scp:$exp/ivectors_sre/ivector.scp $exp/ivectors_sre/mean.vec || exit 1;
  
  $train_cmd $exp/ivectors_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre/ivector.scp ark:- |" \
    ark:$data/sre/utt2spk $exp/ivectors_sre/transform.mat || exit 1;
  
  # Train the PLDA model.
  $train_cmd $exp/ivectors_sre/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre/ivector.scp ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $exp/ivectors_sre/plda_lda${lda_dim} || exit 1;
  
  # Coreext C5
  $train_cmd $exp/ivector_scores/log/sre10_coreext_c5_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre10_enroll_coreext_c5_$gender/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_coreext_c5_$gender/spk2utt scp:$exp/ivectors_sre10_enroll_coreext_c5_$gender/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre/mean.vec scp:$exp/ivectors_sre10_test_coreext_c5_$gender/ivector.scp ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_coreext_c5_$gender/trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores/sre10_coreext_c5_scores_$gender || exit 1;
  
  # Evalation using det_score
  cp $sre10_trials_c5_ext/../male/trials $data/sre10_test_coreext_c5_$gender/trials_male
  cp $sre10_trials_c5_ext/../female/trials $data/sre10_test_coreext_c5_$gender/trials_female
  utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_male $exp/ivector_scores/sre10_coreext_c5_scores_$gender > $exp/ivector_scores/sre10_coreext_c5_scores_male
  utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_female $exp/ivector_scores/sre10_coreext_c5_scores_$gender > $exp/ivector_scores/sre10_coreext_c5_scores_female
  pooled_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials $exp/ivector_scores/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/ivector_scores/sre10_coreext_c5_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/ivector_scores/sre10_coreext_c5_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"
  
  paste $data/sre10_test_coreext_c5_$gender/trials $exp/ivector_scores/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' > $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.new
  grep ' target$' $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.target
  grep ' nontarget$' $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre10_coreext_c5_scores_${gender}.target', '$exp/ivector_scores/sre10_coreext_c5_scores_${gender}.nontarget', '$exp/ivector_scores/sre10_coreext_c5_scores_${gender}.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.new $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.target $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.nontarget
  tail -n 1 $exp/ivector_scores/sre10_coreext_c5_scores_${gender}.result

  paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/ivector_scores/sre10_coreext_c5_scores_male | awk '{print $6, $3}' > $exp/ivector_scores/sre10_coreext_c5_scores_male.new
  grep ' target$' $exp/ivector_scores/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_coreext_c5_scores_male.target
  grep ' nontarget$' $exp/ivector_scores/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_coreext_c5_scores_male.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre10_coreext_c5_scores_male.target', '$exp/ivector_scores/sre10_coreext_c5_scores_male.nontarget', '$exp/ivector_scores/sre10_coreext_c5_scores_male.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre10_coreext_c5_scores_male.new $exp/ivector_scores/sre10_coreext_c5_scores_male.target $exp/ivector_scores/sre10_coreext_c5_scores_male.nontarget
  tail -n 1 $exp/ivector_scores/sre10_coreext_c5_scores_male.result

  paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/ivector_scores/sre10_coreext_c5_scores_female | awk '{print $6, $3}' > $exp/ivector_scores/sre10_coreext_c5_scores_female.new
  grep ' target$' $exp/ivector_scores/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_coreext_c5_scores_female.target
  grep ' nontarget$' $exp/ivector_scores/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_coreext_c5_scores_female.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre10_coreext_c5_scores_female.target', '$exp/ivector_scores/sre10_coreext_c5_scores_female.nontarget', '$exp/ivector_scores/sre10_coreext_c5_scores_female.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre10_coreext_c5_scores_female.new $exp/ivector_scores/sre10_coreext_c5_scores_female.target $exp/ivector_scores/sre10_coreext_c5_scores_female.nontarget
  tail -n 1 $exp/ivector_scores/sre10_coreext_c5_scores_female.result
  
  
  # 10s-10s
  $train_cmd $exp/ivector_scores/log/sre10_10s_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre10_enroll_10s_$gender/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_10s_$gender/spk2utt scp:$exp/ivectors_sre10_enroll_10s_$gender/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre/mean.vec scp:$exp/ivectors_sre10_test_10s_$gender/ivector.scp ark:- | transform-vec $exp/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_10s_$gender/trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores/sre10_10s_scores_$gender || exit 1;
  
  cp $sre10_trials_10s/../male/trials $data/sre10_test_10s_$gender/trials_male
  cp $sre10_trials_10s/../female/trials $data/sre10_test_10s_$gender/trials_female
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_male $exp/ivector_scores/sre10_10s_scores_$gender > $exp/ivector_scores/sre10_10s_scores_male
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_female $exp/ivector_scores/sre10_10s_scores_$gender > $exp/ivector_scores/sre10_10s_scores_female
  pooled_eer=$(paste $data/sre10_test_10s_$gender/trials $exp/ivector_scores/sre10_10s_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $data/sre10_test_10s_$gender/trials_male $exp/ivector_scores/sre10_10s_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $data/sre10_test_10s_$gender/trials_female $exp/ivector_scores/sre10_10s_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

  paste $data/sre10_test_10s_$gender/trials $exp/ivector_scores/sre10_10s_scores_$gender | awk '{print $6, $3}' > $exp/ivector_scores/sre10_10s_scores_${gender}.new
  grep ' target$' $exp/ivector_scores/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_10s_scores_${gender}.target
  grep ' nontarget$' $exp/ivector_scores/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_10s_scores_${gender}.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre10_10s_scores_${gender}.target', '$exp/ivector_scores/sre10_10s_scores_${gender}.nontarget', '$exp/ivector_scores/sre10_10s_scores_${gender}.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre10_10s_scores_${gender}.new $exp/ivector_scores/sre10_10s_scores_${gender}.target $exp/ivector_scores/sre10_10s_scores_${gender}.nontarget
  tail -n 1 $exp/ivector_scores/sre10_10s_scores_${gender}.result

  paste $data/sre10_test_10s_$gender/trials_male $exp/ivector_scores/sre10_10s_scores_male | awk '{print $6, $3}' > $exp/ivector_scores/sre10_10s_scores_male.new
  grep ' target$' $exp/ivector_scores/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_10s_scores_male.target
  grep ' nontarget$' $exp/ivector_scores/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_10s_scores_male.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre10_10s_scores_male.target', '$exp/ivector_scores/sre10_10s_scores_male.nontarget', '$exp/ivector_scores/sre10_10s_scores_male.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre10_10s_scores_male.new $exp/ivector_scores/sre10_10s_scores_male.target $exp/ivector_scores/sre10_10s_scores_male.nontarget
  tail -n 1 $exp/ivector_scores/sre10_10s_scores_male.result

  paste $data/sre10_test_10s_$gender/trials_female $exp/ivector_scores/sre10_10s_scores_female | awk '{print $6, $3}' > $exp/ivector_scores/sre10_10s_scores_female.new
  grep ' target$' $exp/ivector_scores/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_10s_scores_female.target
  grep ' nontarget$' $exp/ivector_scores/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores/sre10_10s_scores_female.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores/sre10_10s_scores_female.target', '$exp/ivector_scores/sre10_10s_scores_female.nontarget', '$exp/ivector_scores/sre10_10s_scores_female.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows
  cd -
  rm -f $exp/ivector_scores/sre10_10s_scores_female.new $exp/ivector_scores/sre10_10s_scores_female.target $exp/ivector_scores/sre10_10s_scores_female.nontarget
  tail -n 1 $exp/ivector_scores/sre10_10s_scores_female.result
fi  


