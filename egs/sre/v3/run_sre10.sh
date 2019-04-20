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

export fea_nj=32
export train_nj=32
nnet_nj=32

root=/mnt/lv10/person/liuyi/sre.full/
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc

gender=pool

data_root=/mnt/lv10/person/liuyi/ly_list/sre16_kaldi_list/
sre10_dir=$data_root/sre10_eval/
sre10_train_c5_ext=$sre10_dir/coreext_c5/enroll/$gender/
sre10_trials_c5_ext=$sre10_dir/coreext_c5/test/$gender/
sre10_train_10s=$sre10_dir/10sec/enroll/$gender/
sre10_trials_10s=$sre10_dir/10sec/test/$gender/

ali_suffix=4k_ali
egs_dir=$exp/cvector_nnet_1a_${ali_suffix}/egs

nnet_suffix=${ali_suffix}_1share
nnetdir=$exp/cvector_nnet_1a_$nnet_suffix

stage=0

# The ASR data is generated using s5/run.sh.
# Run that before this script.
# After running s5/run.sh, the asr training and alignments are prepared in data/train_nodup exp/tri6a_4k_ali or exp/tri6a_10k_ali
# We will use 4K alignments as the training data.

# You can modified the number of layers shared between the two tasks by editing the nnet config file in local/nnet3_cvector/cvector/prepare_nnet3_xconfig.sh
# and run from step 3.
# An example is included in local/nnet3_cvector/cvector/prepare_nnet3_xconfig_3share.sh

if [ $stage -le 0 ]; then
  # Note: Because the asr feature is extracted with snip-edge, the speaker feature (23-dim MFCCs) should also use snip-edge=true (which is default) to make #num of frames consistent. So we have to extract another version (but `almost the same`) of 23-dim MFCCs.
  utils/copy_data_dir_new.sh $data/train_nodup $data/train_asr_nodup
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_snip_edge.conf --nj $fea_nj --cmd "$train_cmd" \
    $data/train_asr_nodup $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/train_asr_nodup
  sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" \
    $data/train_asr_nodup $exp/make_vad $vaddir
  utils/fix_data_dir.sh $data/train_asr_nodup
fi

if [ $stage -le 1 ]; then
  # Prepare features for xvector egs and the alignments for the new data
  utils/copy_data_dir_new.sh $data/train_asr_nodup $data/train_asr_nodup_vad
  sid/nnet3_cvector/cvector/prepare_feats.sh --cmd "$train_cmd" \
    --nj 32 \
    --vad-dir $data/train_asr_nodup_vad \
    --ali-dir $exp/tri6a_$ali_suffix \
    --min-len 100 \
    --min-num-utts 8 \
    $data/train_asr_nodup \
    $data/train_asr_nodup_wcmvn \
    $exp/train_asr_nodup_wcmvn_$ali_suffix \
    $data/train_asr_nodup_wcmvn_nosil \
    $exp/train_asr_nodup_wcmvn_${ali_suffix}_nosil
fi

if [ $stage -le 2 ]; then
  # Write nnet3 config file and prepare egs for nnet3 training
  gmm_dir=$exp/tri6a_$ali_suffix
  num_senones=$(tree-info $gmm_dir/tree |grep num-pdfs|awk '{print $2}') || exit 1
  num_speakers=$(awk '{print $2}' $data/swbd_sre_combined_nosil/utt2spk | sort | uniq -c | wc -l)
  feat_dim=$(feat-to-dim scp:$data/swbd_sre_combined_nosil/feats.scp -) || exit 1
  local/nnet3_cvector/cvector/prepare_nnet3_xconfig.sh \
    --num-senones $num_senones \
    --num-speakers $num_speakers \
    --feat-dim $feat_dim \
    $egs_dir
  
  sid/nnet3_cvector/cvector/get_egs_cvector.sh --cmd "$train_cmd"\
    --stage 0 \
    --nj 6 \
    --repeats_per_spk 7500 \
    --num-train-archives 400 \
    --am-feat-dir $data/train_asr_nodup_wcmvn \
    --am-ali-dir $exp/train_asr_nodup_wcmvn_$ali_suffix \
    --xvec-feat-dir $data/swbd_sre_combined_nosil \
    $egs_dir
fi

if [ $stage -le 3 ]; then 
  # Train the phonetic adaptation model
  gmm_dir=$exp/tri6a_$ali_suffix
  num_senones=$(tree-info $gmm_dir/tree |grep num-pdfs|awk '{print $2}') || exit 1
  num_speakers=$(awk '{print $2}' $data/swbd_sre_combined_nosil/utt2spk | sort | uniq -c | wc -l)
  feat_dim=$(feat-to-dim scp:$data/swbd_sre_combined_nosil/feats.scp -) || exit 1
  local/nnet3_cvector/cvector/prepare_nnet3_xconfig.sh \
    --num-senones $num_senones \
    --num-speakers $num_speakers \
    --feat-dim $feat_dim \
    $nnetdir
  
  local/nnet3_cvector/cvector/train_cvector.sh --stage -10 1.0 $egs_dir/egs_am 1.0 $egs_dir/egs_xvec $nnetdir 
fi

if [ $stage -le 4 ]; then
  # Extract x-vector with phonetic adaptation
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/sre_combined \
    $exp/cvectors_sre_combined_$nnet_suffix

  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/sre10_enroll_coreext_c5_$gender \
    $exp/cvectors_sre10_enroll_coreext_c5_${gender}_$nnet_suffix
  
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/sre10_test_coreext_c5_$gender \
    $exp/cvectors_sre10_test_coreext_c5_${gender}_$nnet_suffix
  
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/sre10_enroll_10s_$gender \
    $exp/cvectors_sre10_enroll_10s_${gender}_$nnet_suffix
  
  sid/nnet3/xvector/extract_xvectors_new.sh --cmd "$train_cmd" --use-gpu false --nj $nnet_nj \
    $nnetdir "tdnn6_xvec.affine" $data/sre10_test_10s_$gender \
    $exp/cvectors_sre10_test_10s_${gender}_$nnet_suffix
fi

if [ $stage -le 5 ]; then
  lda_dim=150
  
  $train_cmd $exp/cvectors_sre_combined_${nnet_suffix}/log/compute_mean.log \
    ivector-mean scp:$exp/cvectors_sre_combined_${nnet_suffix}/xvector_sre_combined.scp \
    $exp/cvectors_sre_combined_${nnet_suffix}/mean.vec || exit 1;
  
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $exp/cvectors_sre_combined_${nnet_suffix}/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$exp/cvectors_sre_combined_${nnet_suffix}/xvector_sre_combined.scp ark:- |" \
    ark:$data/sre_combined/utt2spk $exp/cvectors_sre_combined_${nnet_suffix}/transform.mat || exit 1;
  
  $train_cmd $exp/cvectors_sre_combined_${nnet_suffix}/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:$exp/cvectors_sre_combined_${nnet_suffix}/xvector_sre_combined.scp ark:- | transform-vec $exp/cvectors_sre_combined_${nnet_suffix}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $exp/cvectors_sre_combined_${nnet_suffix}/plda_lda${lda_dim} || exit 1;
  
  
  # Coreext C5
  $train_cmd $exp/cvector_scores_${nnet_suffix}/log/sre10_coreext_c5_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/cvectors_sre10_enroll_coreext_c5_${gender}_${nnet_suffix}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/cvectors_sre_combined_${nnet_suffix}/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_coreext_c5_$gender/spk2utt scp:$exp/cvectors_sre10_enroll_coreext_c5_${gender}_${nnet_suffix}/xvector_sre10_enroll_coreext_c5_${gender}.scp ark:- | ivector-subtract-global-mean $exp/cvectors_sre_combined_${nnet_suffix}/mean.vec ark:- ark:- | transform-vec $exp/cvectors_sre_combined_${nnet_suffix}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/cvectors_sre_combined_${nnet_suffix}/mean.vec scp:$exp/cvectors_sre10_test_coreext_c5_${gender}_${nnet_suffix}/xvector_sre10_test_coreext_c5_${gender}.scp ark:- | transform-vec $exp/cvectors_sre_combined_${nnet_suffix}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_coreext_c5_$gender/trials' | cut -d\  --fields=1,2 |" $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_$gender || exit 1;
  
  cp $sre10_trials_c5_ext/../male/trials $data/sre10_test_coreext_c5_$gender/trials_male
  cp $sre10_trials_c5_ext/../female/trials $data/sre10_test_coreext_c5_$gender/trials_female
  utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_male $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_$gender > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male
  utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_female $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_$gender > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female
  pooled_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"
  
  paste $data/sre10_test_coreext_c5_$gender/trials $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.new
  grep ' target$' $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.target
  grep ' nontarget$' $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.target', '$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.nontarget', '$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.new $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.target $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.nontarget
  tail -n 1 $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_${gender}.result
  
  paste $data/sre10_test_coreext_c5_$gender/trials_male $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male | awk '{print $6, $3}' > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.new
  grep ' target$' $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.target
  grep ' nontarget$' $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.target', '$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.nontarget', '$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.new $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.target $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.nontarget
  tail -n 1 $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_male.result
  
  paste $data/sre10_test_coreext_c5_$gender/trials_female $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female | awk '{print $6, $3}' > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.new
  grep ' target$' $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.target
  grep ' nontarget$' $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.target', '$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.nontarget', '$exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.new $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.target $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.nontarget
  tail -n 1 $exp/cvector_scores_${nnet_suffix}/sre10_coreext_c5_scores_female.result
  
  # 10s-10s
  $train_cmd $exp/cvector_scores_${nnet_suffix}/log/sre10_10s_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$exp/cvectors_sre10_enroll_10s_${gender}_${nnet_suffix}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/cvectors_sre_combined_${nnet_suffix}/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_10s_$gender/spk2utt scp:$exp/cvectors_sre10_enroll_10s_${gender}_${nnet_suffix}/xvector_sre10_enroll_10s_${gender}.scp ark:- | ivector-subtract-global-mean $exp/cvectors_sre_combined_${nnet_suffix}/mean.vec ark:- ark:- | transform-vec $exp/cvectors_sre_combined_${nnet_suffix}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $exp/cvectors_sre_combined_${nnet_suffix}/mean.vec scp:$exp/cvectors_sre10_test_10s_${gender}_${nnet_suffix}/xvector_sre10_test_10s_${gender}.scp ark:- | transform-vec $exp/cvectors_sre_combined_${nnet_suffix}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_10s_$gender/trials' | cut -d\  --fields=1,2 |" $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_$gender || exit 1;
  
  cp $sre10_trials_10s/../male/trials $data/sre10_test_10s_$gender/trials_male
  cp $sre10_trials_10s/../female/trials $data/sre10_test_10s_$gender/trials_female
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_male $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_$gender > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male
  utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_female $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_$gender > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female
  pooled_eer=$(paste $data/sre10_test_10s_$gender/trials $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $data/sre10_test_10s_$gender/trials_male $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $data/sre10_test_10s_$gender/trials_female $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"
  
  paste $data/sre10_test_10s_$gender/trials $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_$gender | awk '{print $6, $3}' > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.new
  grep ' target$' $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.target
  grep ' nontarget$' $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.target', '$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.nontarget', '$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.new $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.target $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.nontarget
  tail -n 1 $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_${gender}.result

  paste $data/sre10_test_10s_$gender/trials_male $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male | awk '{print $6, $3}' > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.new
  grep ' target$' $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.target
  grep ' nontarget$' $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.target', '$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.nontarget', '$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.new $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.target $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.nontarget
  tail -n 1 $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_male.result

  paste $data/sre10_test_10s_$gender/trials_female $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female | awk '{print $6, $3}' > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.new
  grep ' target$' $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.target
  grep ' nontarget$' $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.target', '$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.nontarget', '$exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.new $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.target $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.nontarget
  tail -n 1 $exp/cvector_scores_${nnet_suffix}/sre10_10s_scores_female.result
fi 


