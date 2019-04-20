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

fea_nj=32
nnet_nj=32

root=/mnt/lv10/person/liuyi/sre.full/
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc
nnet_dir=$exp/xvector_nnet_1a

gender=pool

data_root=/mnt/lv10/person/liuyi/ly_list/sre16_kaldi_list/
sre10_dir=$data_root/sre10_eval/
sre10_train_c5_ext=$sre10_dir/coreext_c5/enroll/$gender/
sre10_trials_c5_ext=$sre10_dir/coreext_c5/test/$gender/
sre10_train_10s=$sre10_dir/10sec/enroll/$gender/
sre10_trials_10s=$sre10_dir/10sec/test/$gender/

rirs_noises=/mnt/lv10/person/liuyi/ly_database/RIRS_NOISES/
musan=/mnt/lv10/person/liuyi/ly_database/musan/

stage=0

## Data preparation
if [ $stage -le 0 ]; then
  # See v1/run_sre10.sh for more details.
#   # Exclude Mixer6, since Mixer6 is used in SRE10.
#   local/make_mx6.sh /mnt/lv10/person/liuyi/ly_database $data
  
  # combine all sre data (04-08) and Mixer6
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
  # Extract 23-dim MFCCs (which is different with the i-vector setup) following the Kaldi configuration.
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
  # Data augmentation
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data/swbd_sre/utt2num_frames > $data/swbd_sre/reco2dur
  
  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/mediumroom/rir_list")
  
  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    $data/swbd_sre $data/swbd_sre_reverb
  cp $data/swbd_sre/vad.scp $data/swbd_sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" $data/swbd_sre_reverb $data/swbd_sre_reverb.new
  rm -rf $data/swbd_sre_reverb
  mv $data/swbd_sre_reverb.new $data/swbd_sre_reverb
  
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan $data
  
  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh $data/musan_${name}
    mv $data/musan_${name}/utt2dur $data/musan_${name}/reco2dur
  done
  
  # Augment with musan_noise
  python steps/data/augment_data_dir_new.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$data/musan_noise" $data/swbd_sre $data/swbd_sre_noise
  # Augment with musan_music
  python steps/data/augment_data_dir_new.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$data/musan_music" $data/swbd_sre $data/swbd_sre_music
  # Augment with musan_speech
  python steps/data/augment_data_dir_new.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$data/musan_speech" $data/swbd_sre $data/swbd_sre_babble
  
  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh $data/swbd_sre_aug $data/swbd_sre_reverb $data/swbd_sre_noise $data/swbd_sre_music $data/swbd_sre_babble
  
  utils/subset_data_dir.sh $data/swbd_sre_aug 128000 $data/swbd_sre_aug_128k
  utils/filter_scp.pl $data/swbd_sre_aug_128k/utt2spk $data/swbd_sre_aug/utt2uniq > $data/swbd_sre_aug_128k/utt2uniq
  utils/fix_data_dir.sh $data/swbd_sre_aug_128k
  
  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $fea_nj --cmd "$train_cmd" \
    $data/swbd_sre_aug_128k $exp/make_mfcc $mfccdir
  
  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh $data/swbd_sre_combined $data/swbd_sre_aug_128k $data/swbd_sre
  utils/fix_data_dir.sh $data/swbd_sre_combined
  
  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  # Just use SRE data for PLDA training
  utils/copy_data_dir.sh $data/swbd_sre_combined $data/sre_combined
  utils/filter_scp.pl $data/sre/spk2utt $data/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > $data/sre_combined/utt2spk
  utils/fix_data_dir.sh $data/sre_combined
fi
  
if [ $stage -le 3 ]; then
  # Prepare the training features (WCMVN + VAD)
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj $fea_nj --cmd "$train_cmd" \
    $data/swbd_sre_combined $data/swbd_sre_combined_nosil $exp/swbd_sre_combined_nosil
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil
fi
 
if [ $stage -le 4 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want at least 5s (500 frames) per utterance.
  min_len=500
  mv $data/swbd_sre_combined_nosil/utt2num_frames $data/swbd_sre_combined_nosil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/swbd_sre_combined_nosil/utt2num_frames.bak > $data/swbd_sre_combined_nosil/utt2num_frames
  utils/filter_scp.pl $data/swbd_sre_combined_nosil/utt2num_frames $data/swbd_sre_combined_nosil/utt2spk > $data/swbd_sre_combined_nosil/utt2spk.new
  mv $data/swbd_sre_combined_nosil/utt2spk.new $data/swbd_sre_combined_nosil/utt2spk
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil
  
  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' $data/swbd_sre_combined_nosil/spk2utt > $data/swbd_sre_combined_nosil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/swbd_sre_combined_nosil/spk2num | utils/filter_scp.pl - $data/swbd_sre_combined_nosil/spk2utt > $data/swbd_sre_combined_nosil/spk2utt.new
  mv $data/swbd_sre_combined_nosil/spk2utt.new $data/swbd_sre_combined_nosil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/swbd_sre_combined_nosil/spk2utt > $data/swbd_sre_combined_nosil/utt2spk
  
  utils/filter_scp.pl $data/swbd_sre_combined_nosil/utt2spk $data/swbd_sre_combined_nosil/utt2num_frames > $data/swbd_sre_combined_nosil/utt2num_frames.new
  mv $data/swbd_sre_combined_nosil/utt2num_frames.new $data/swbd_sre_combined_nosil/utt2num_frames
  
  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil
fi

if [ $stage -le 5 ]; then
  local/nnet3/xvector/run_xvector_new.sh --stage 0 --train-stage -10 \
    --data $data/swbd_sre_combined_nosil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs
fi 

if [ $stage -le 6 ];then
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnet_dir "tdnn6.affine" $data/sre_combined \
    $exp/xvectors_sre_combined

  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnet_dir "tdnn6.affine" $data/sre10_enroll_coreext_c5_$gender \
    $exp/xvectors_sre10_enroll_coreext_c5_$gender
  
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnet_dir "tdnn6.affine" $data/sre10_test_coreext_c5_$gender \
    $exp/xvectors_sre10_test_coreext_c5_$gender
  
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnet_dir "tdnn6.affine" $data/sre10_enroll_10s_$gender \
    $exp/xvectors_sre10_enroll_10s_$gender
  
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnet_dir "tdnn6.affine" $data/sre10_test_10s_$gender \
    $exp/xvectors_sre10_test_10s_$gender
fi

if [ $stage -le 7 ]; then
  lda_dim=150
  
  $train_cmd $exp/xvectors_sre_combined/log/compute_mean.log \
    ivector-mean scp:$exp/xvectors_sre_combined/xvector_sre_combined.scp \
    $exp/xvectors_sre_combined/mean.vec || exit 1;
  
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $exp/xvectors_sre_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$exp/xvectors_sre_combined/xvector_sre_combined.scp ark:- |" \
    ark:$data/sre_combined/utt2spk $exp/xvectors_sre_combined/transform.mat || exit 1;
  
  $train_cmd $exp/xvectors_sre_combined/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:$exp/xvectors_sre_combined/xvector_sre_combined.scp ark:- | transform-vec $exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $exp/xvectors_sre_combined/plda_lda${lda_dim} || exit 1;
  
  # Coreext C5
  $train_cmd $exp/xvector_scores/log/sre10_coreext_c5_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/xvectors_sre10_enroll_coreext_c5_$gender/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/xvectors_sre_combined/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_coreext_c5_$gender/spk2utt scp:$exp/xvectors_sre10_enroll_coreext_c5_$gender/xvector_sre10_enroll_coreext_c5_${gender}.scp ark:- | ivector-subtract-global-mean $exp/xvectors_sre_combined/mean.vec ark:- ark:- | transform-vec $exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/xvectors_sre_combined/mean.vec scp:$exp/xvectors_sre10_test_coreext_c5_$gender/xvector_sre10_test_coreext_c5_${gender}.scp ark:- | transform-vec $exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_coreext_c5_$gender/trials' | cut -d\  --fields=1,2 |" $exp/xvector_scores/sre10_coreext_c5_scores_$gender || exit 1;
  
  cp $sre10_trials_c5_ext/../male/trials $data/sre10_test_coreext_c5_$gender/trials_male
  cp $sre10_trials_c5_ext/../female/trials $data/sre10_test_coreext_c5_$gender/trials_female
  utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_male $exp/xvector_scores/sre10_coreext_c5_scores_$gender > $exp/xvector_scores/sre10_coreext_c5_scores_male
  utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_female $exp/xvector_scores/sre10_coreext_c5_scores_$gender > $exp/xvector_scores/sre10_coreext_c5_scores_female
  pooled_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials $exp/xvector_scores/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/xvector_scores/sre10_coreext_c5_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/xvector_scores/sre10_coreext_c5_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

  paste $data/sre10_test_coreext_c5_$gender/trials $exp/xvector_scores/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' > $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.new
  grep ' target$' $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.target
  grep ' nontarget$' $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/xvector_scores/sre10_coreext_c5_scores_${gender}.target', '$exp/xvector_scores/sre10_coreext_c5_scores_${gender}.nontarget', '$exp/xvector_scores/sre10_coreext_c5_scores_${gender}.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.new $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.target $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.nontarget
  tail -n 1 $exp/xvector_scores/sre10_coreext_c5_scores_${gender}.result

  paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/xvector_scores/sre10_coreext_c5_scores_male | awk '{print $6, $3}' > $exp/xvector_scores/sre10_coreext_c5_scores_male.new
  grep ' target$' $exp/xvector_scores/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_coreext_c5_scores_male.target
  grep ' nontarget$' $exp/xvector_scores/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_coreext_c5_scores_male.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/xvector_scores/sre10_coreext_c5_scores_male.target', '$exp/xvector_scores/sre10_coreext_c5_scores_male.nontarget', '$exp/xvector_scores/sre10_coreext_c5_scores_male.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/xvector_scores/sre10_coreext_c5_scores_male.new $exp/xvector_scores/sre10_coreext_c5_scores_male.target $exp/xvector_scores/sre10_coreext_c5_scores_male.nontarget
  tail -n 1 $exp/xvector_scores/sre10_coreext_c5_scores_male.result

  paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/xvector_scores/sre10_coreext_c5_scores_female | awk '{print $6, $3}' > $exp/xvector_scores/sre10_coreext_c5_scores_female.new
  grep ' target$' $exp/xvector_scores/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_coreext_c5_scores_female.target
  grep ' nontarget$' $exp/xvector_scores/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_coreext_c5_scores_female.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/xvector_scores/sre10_coreext_c5_scores_female.target', '$exp/xvector_scores/sre10_coreext_c5_scores_female.nontarget', '$exp/xvector_scores/sre10_coreext_c5_scores_female.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/xvector_scores/sre10_coreext_c5_scores_female.new $exp/xvector_scores/sre10_coreext_c5_scores_female.target $exp/xvector_scores/sre10_coreext_c5_scores_female.nontarget
  tail -n 1 $exp/xvector_scores/sre10_coreext_c5_scores_female.result
  
  # 10s-10s
  $train_cmd $exp/xvector_scores/log/sre10_10s_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/xvectors_sre10_enroll_10s_$gender/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/xvectors_sre_combined/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_10s_$gender/spk2utt scp:$exp/xvectors_sre10_enroll_10s_$gender/xvector_sre10_enroll_10s_${gender}.scp ark:- | ivector-subtract-global-mean $exp/xvectors_sre_combined/mean.vec ark:- ark:- | transform-vec $exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/xvectors_sre_combined/mean.vec scp:$exp/xvectors_sre10_test_10s_$gender/xvector_sre10_test_10s_${gender}.scp ark:- | transform-vec $exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_10s_$gender/trials' | cut -d\  --fields=1,2 |" $exp/xvector_scores/sre10_10s_scores_$gender || exit 1;
  
  cp $sre10_trials_10s/../male/trials $data/sre10_test_10s_$gender/trials_male
  cp $sre10_trials_10s/../female/trials $data/sre10_test_10s_$gender/trials_female
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_male $exp/xvector_scores/sre10_10s_scores_$gender > $exp/xvector_scores/sre10_10s_scores_male
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_female $exp/xvector_scores/sre10_10s_scores_$gender > $exp/xvector_scores/sre10_10s_scores_female
  pooled_eer=$(paste $data/sre10_test_10s_$gender/trials $exp/xvector_scores/sre10_10s_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $data/sre10_test_10s_$gender/trials_male $exp/xvector_scores/sre10_10s_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $data/sre10_test_10s_$gender/trials_female $exp/xvector_scores/sre10_10s_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"
  
  paste $data/sre10_test_10s_$gender/trials $exp/xvector_scores/sre10_10s_scores_$gender | awk '{print $6, $3}' > $exp/xvector_scores/sre10_10s_scores_${gender}.new
  grep ' target$' $exp/xvector_scores/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_10s_scores_${gender}.target
  grep ' nontarget$' $exp/xvector_scores/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_10s_scores_${gender}.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/xvector_scores/sre10_10s_scores_${gender}.target', '$exp/xvector_scores/sre10_10s_scores_${gender}.nontarget', '$exp/xvector_scores/sre10_10s_scores_${gender}.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/xvector_scores/sre10_10s_scores_${gender}.new $exp/xvector_scores/sre10_10s_scores_${gender}.target $exp/xvector_scores/sre10_10s_scores_${gender}.nontarget
  tail -n 1 $exp/xvector_scores/sre10_10s_scores_${gender}.result

  paste $data/sre10_test_10s_$gender/trials_male $exp/xvector_scores/sre10_10s_scores_male | awk '{print $6, $3}' > $exp/xvector_scores/sre10_10s_scores_male.new
  grep ' target$' $exp/xvector_scores/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_10s_scores_male.target
  grep ' nontarget$' $exp/xvector_scores/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_10s_scores_male.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/xvector_scores/sre10_10s_scores_male.target', '$exp/xvector_scores/sre10_10s_scores_male.nontarget', '$exp/xvector_scores/sre10_10s_scores_male.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/xvector_scores/sre10_10s_scores_male.new $exp/xvector_scores/sre10_10s_scores_male.target $exp/xvector_scores/sre10_10s_scores_male.nontarget
  tail -n 1 $exp/xvector_scores/sre10_10s_scores_male.result

  paste $data/sre10_test_10s_$gender/trials_female $exp/xvector_scores/sre10_10s_scores_female | awk '{print $6, $3}' > $exp/xvector_scores/sre10_10s_scores_female.new
  grep ' target$' $exp/xvector_scores/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_10s_scores_female.target
  grep ' nontarget$' $exp/xvector_scores/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/xvector_scores/sre10_10s_scores_female.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/xvector_scores/sre10_10s_scores_female.target', '$exp/xvector_scores/sre10_10s_scores_female.nontarget', '$exp/xvector_scores/sre10_10s_scores_female.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/xvector_scores/sre10_10s_scores_female.new $exp/xvector_scores/sre10_10s_scores_female.target $exp/xvector_scores/sre10_10s_scores_female.nontarget
  tail -n 1 $exp/xvector_scores/sre10_10s_scores_female.result
fi  
 
 
