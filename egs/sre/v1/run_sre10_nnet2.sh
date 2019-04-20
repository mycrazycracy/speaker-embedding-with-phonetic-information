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

export fea_nj=16
export train_nj=24

root=/mnt/lv10/person/liuyi/sre/
export data=$root/data
export exp=$root/exp
export tmpdir=$data/local/data
export data_links=$data/local/data/links
export mfccdir=$root/mfcc
export vaddir=$root/mfcc

# we won't use the noise data to do data augmentation
rirs_noises=/mnt/lv10/person/liuyi/ly_database/RIRS_NOISES/
musan=/mnt/lv10/person/liuyi/ly_database/musan/

gender=pool

data_root=/mnt/lv10/person/liuyi/ly_list/sre16_kaldi_list/
sre10_dir=$data_root/sre10_eval/
sre10_train_c5_ext=$sre10_dir/coreext_c5/enroll/$gender/
sre10_trials_c5_ext=$sre10_dir/coreext_c5/test/$gender/
sre10_train_10s=$sre10_dir/10sec/enroll/$gender/
sre10_trials_10s=$sre10_dir/10sec/test/$gender/

nnet=$exp/nnet2_online/nnet_ms_a/final.mdl

stage=0

# Train a DNN on about 1800 hours of the english portion of Fisher.
if [ $stage -le 0 ]; then
  local/dnn/train_dnn.sh
fi

if [ $stage -le 1 ]; then
  # Extract DNN features. (40-dim hires)
  cp -r $data/swbd_sre $data/swbd_sre_dnn
  cp -r $data/sre $data/sre_dnn
  cp -r $data/sre10_enroll_coreext_c5_$gender $data/sre10_enroll_coreext_c5_${gender}_dnn
  cp -r $data/sre10_test_coreext_c5_$gender $data/sre10_test_coreext_c5_${gender}_dnn
  cp -r $data/sre10_enroll_10s_$gender $data/sre10_enroll_10s_${gender}_dnn
  cp -r $data/sre10_test_10s_$gender $data/sre10_test_10s_${gender}_dnn
  
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/swbd_sre_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre10_enroll_coreext_c5_${gender}_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre10_test_coreext_c5_${gender}_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre10_enroll_10s_${gender}_dnn $exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $fea_nj \
    --cmd "$train_cmd" $data/sre10_test_10s_${gender}_dnn $exp/make_mfcc $mfccdir
  
  for name in swbd_sre_dnn sre_dnn sre10_enroll_coreext_c5_${gender}_dnn \
      sre10_test_coreext_c5_${gender}_dnn sre10_enroll_10s_${gender}_dnn sre10_test_10s_${gender}_dnn; do
    utils/fix_data_dir.sh $data/${name}
  done
  
  for name in swbd_sre sre sre10_enroll_coreext_c5_${gender} \
      sre10_test_coreext_c5_${gender} sre10_enroll_10s_${gender} sre10_test_10s_${gender}; do
      cp $data/${name}/vad.scp $data/${name}_dnn/vad.scp
      cp $data/${name}/utt2spk $data/${name}_dnn/utt2spk
      cp $data/${name}/spk2utt $data/${name}_dnn/spk2utt
      utils/fix_data_dir.sh $data/${name}_dnn    
  done
fi

if [ $stage -le 2 ]; then
  # Subset training data for faster sup-GMM initialization.
  utils/subset_data_dir.sh $data/swbd_sre_dnn 32000 $data/swbd_sre_dnn_32k
  utils/fix_data_dir.sh $data/swbd_sre_dnn_32k
  utils/subset_data_dir.sh --utt-list $data/swbd_sre_dnn_32k/utt2spk \
    $data/swbd_sre $data/swbd_sre_32k
  utils/fix_data_dir.sh $data/swbd_sre_32k
  
  # Train the UBM.
  sid/init_full_ubm_from_dnn.sh --cmd "$train_cmd --mem 15G" \
    --use-gpu false --nj $train_nj \
    $data/swbd_sre_32k \
    $data/swbd_sre_dnn_32k $nnet $exp/full_ubm_nnet2
if

if [ $stage -le 3 ]; then
  # Train an i-vector extractor based on the DNN-UBM.
  sid/train_ivector_extractor_dnn.sh \
    --cmd "$train_cmd" --use-gpu false \
    --nj 2 --num-threads 1 --num-processes 12 \
    --min-post 0.015 --ivector-dim 600 --num-iters 5 \
    $exp/full_ubm_nnet2/final.ubm $nnet \
    $data/swbd_sre \
    $data/swbd_sre_dnn \
    $exp/extractor_nnet2
fi

if [ $stage -le 4 ]; then 
  # Extract i-vectors using the extractor with the DNN-UBM.
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre \
    $data/sre_dnn \
    $exp/ivectors_sre_nnet2
  
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre10_enroll_coreext_c5_$gender \
    $data/sre10_enroll_coreext_c5_${gender}_dnn \
    $exp/ivectors_sre10_enroll_coreext_c5_${gender}_nnet2
  
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre10_test_coreext_c5_$gender \
    $data/sre10_test_coreext_c5_${gender}_dnn \
    $exp/ivectors_sre10_test_coreext_c5_${gender}_nnet2
  
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre10_enroll_10s_$gender \
    $data/sre10_enroll_10s_${gender}_dnn \
    $exp/ivectors_sre10_enroll_10s_${gender}_nnet2
  
  sid/extract_ivectors_dnn.sh \
    --cmd "$train_cmd --mem 15G" --use-gpu false \
    --nj $train_nj \
    $exp/extractor_nnet2 \
    $nnet \
    $data/sre10_test_10s_$gender \
    $data/sre10_test_10s_${gender}_dnn \
    $exp/ivectors_sre10_test_10s_${gender}_nnet2
fi

if [ $stage -le 5 ]; then
  # LDA + PLDA
  lda_dim=200
  
  $train_cmd $exp/ivectors_sre_nnet2/log/compute_mean.log \
    ivector-mean scp:$exp/ivectors_sre_nnet2/ivector.scp $exp/ivectors_sre_nnet2/mean.vec || exit 1;
  
  $train_cmd $exp/ivectors_sre_nnet2/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre_nnet2/ivector.scp ark:- |" \
    ark:$data/sre/utt2spk $exp/ivectors_sre_nnet2/transform.mat || exit 1;
  
  #  Train the PLDA model.
  $train_cmd $exp/ivectors_sre_nnet2/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$exp/ivectors_sre_nnet2/ivector.scp ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $exp/ivectors_sre_nnet2/plda_lda${lda_dim} || exit 1;
  
  # Coreext C5
  $train_cmd $exp/ivector_scores_nnet2/log/sre10_coreext_c5_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre10_enroll_coreext_c5_${gender}_nnet2/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre_nnet2/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_coreext_c5_$gender/spk2utt scp:$exp/ivectors_sre10_enroll_coreext_c5_${gender}_nnet2/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre_nnet2/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre_nnet2/mean.vec scp:$exp/ivectors_sre10_test_coreext_c5_${gender}_nnet2/ivector.scp ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_coreext_c5_$gender/trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_$gender || exit 1;

    cp $sre10_trials_c5_ext/../male/trials $data/sre10_test_coreext_c5_$gender/trials_male
    cp $sre10_trials_c5_ext/../female/trials $data/sre10_test_coreext_c5_$gender/trials_female
    utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_male $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_$gender > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male
    utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_female $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_$gender > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female
    pooled_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    male_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    female_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

    paste $data/sre10_test_coreext_c5_$gender/trials $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.new
    grep ' target$' $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.target
    grep ' nontarget$' $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.nontarget
    cd ${KALDI_ROOT}/tools/det_score
    comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.target', '$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.nontarget', '$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.result')"`
    echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
    cd -
    rm -f $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.new $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.target $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.nontarget
    tail -n 1 $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_${gender}.result

    paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.new
    grep ' target$' $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.target
    grep ' nontarget$' $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.nontarget
    cd ${KALDI_ROOT}/tools/det_score
    comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.target', '$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.nontarget', '$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.result')"`
    echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
    cd -
    rm -f $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.new $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.target $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.nontarget
    tail -n 1 $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_male.result

    paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.new
    grep ' target$' $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.target
    grep ' nontarget$' $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.nontarget
    cd ${KALDI_ROOT}/tools/det_score
    comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.target', '$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.nontarget', '$exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.result')"`
    echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
    cd -
    rm -f $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.new $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.target $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.nontarget
    tail -n 1 $exp/ivector_scores_nnet2/sre10_coreext_c5_scores_female.result

  # 10s-10s
  $train_cmd $exp/ivector_scores_nnet2/log/sre10_10s_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/ivectors_sre10_enroll_10s_${gender}_nnet2/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/ivectors_sre_nnet2/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_10s_$gender/spk2utt scp:$exp/ivectors_sre10_enroll_10s_${gender}_nnet2/ivector.scp ark:- | ivector-subtract-global-mean $exp/ivectors_sre_nnet2/mean.vec ark:- ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/ivectors_sre_nnet2/mean.vec scp:$exp/ivectors_sre10_test_10s_${gender}_nnet2/ivector.scp ark:- | transform-vec $exp/ivectors_sre_nnet2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_10s_$gender/trials' | cut -d\  --fields=1,2 |" $exp/ivector_scores_nnet2/sre10_10s_scores_$gender || exit 1;

  cp $sre10_trials_10s/../male/trials $data/sre10_test_10s_$gender/trials_male
  cp $sre10_trials_10s/../female/trials $data/sre10_test_10s_$gender/trials_female
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_male $exp/ivector_scores_nnet2/sre10_10s_scores_$gender > $exp/ivector_scores_nnet2/sre10_10s_scores_male
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_female $exp/ivector_scores_nnet2/sre10_10s_scores_$gender > $exp/ivector_scores_nnet2/sre10_10s_scores_female
  pooled_eer=$(paste $data/sre10_test_10s_$gender/trials $exp/ivector_scores_nnet2/sre10_10s_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $data/sre10_test_10s_$gender/trials_male $exp/ivector_scores_nnet2/sre10_10s_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $data/sre10_test_10s_$gender/trials_female $exp/ivector_scores_nnet2/sre10_10s_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

  paste $data/sre10_test_10s_$gender/trials $exp/ivector_scores_nnet2/sre10_10s_scores_$gender | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.new
  grep ' target$' $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.target', '$exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.nontarget', '$exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.new $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.target $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre10_10s_scores_${gender}.result

  paste $data/sre10_test_10s_$gender/trials_male $exp/ivector_scores_nnet2/sre10_10s_scores_male | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre10_10s_scores_male.new
  grep ' target$' $exp/ivector_scores_nnet2/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_10s_scores_male.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_10s_scores_male.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre10_10s_scores_male.target', '$exp/ivector_scores_nnet2/sre10_10s_scores_male.nontarget', '$exp/ivector_scores_nnet2/sre10_10s_scores_male.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/ivector_scores_nnet2/sre10_10s_scores_male.new $exp/ivector_scores_nnet2/sre10_10s_scores_male.target $exp/ivector_scores_nnet2/sre10_10s_scores_male.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre10_10s_scores_male.result
  
  paste $data/sre10_test_10s_$gender/trials_female $exp/ivector_scores_nnet2/sre10_10s_scores_female | awk '{print $6, $3}' > $exp/ivector_scores_nnet2/sre10_10s_scores_female.new
  grep ' target$' $exp/ivector_scores_nnet2/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_10s_scores_female.target
  grep ' nontarget$' $exp/ivector_scores_nnet2/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/ivector_scores_nnet2/sre10_10s_scores_female.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/ivector_scores_nnet2/sre10_10s_scores_female.target', '$exp/ivector_scores_nnet2/sre10_10s_scores_female.nontarget', '$exp/ivector_scores_nnet2/sre10_10s_scores_female.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/ivector_scores_nnet2/sre10_10s_scores_female.new $exp/ivector_scores_nnet2/sre10_10s_scores_female.target $exp/ivector_scores_nnet2/sre10_10s_scores_female.nontarget
  tail -n 1 $exp/ivector_scores_nnet2/sre10_10s_scores_female.result
fi
