#!/bin/bash

# This script is based on egs/swbd/s5c/run.sh. It trains a
# multisplice time-delay neural network used in the DNN-based speaker
# recognition recipes.

# It's best to run the commands in this one by one.
stage=0
train_stage=-10
srand=0

. cmd.sh
. path.sh
. utils/parse_options.sh

train_nj=16

data=$1
exp=$2
mfccdir=$3

swbd_dir=/mnt/lv10/person/sre16/data/switchboard/Switchboard-P1

set -e # exit on error

# For swbd training: 
#   data/local --> data/local_swbd
#   data/train --> data/train_swbd_asr
#   data/lang_nosp --> data/lang_swb_nosp
#   data/lang  --> data/lang_swbd
#   exp/mono, mono_ali, tri1, tri1_ali, tri2, tri2_ali_100k_nodup, tri2_ali_nodup, tri3, tri3_ali_nodup, tri4, tri4_ali_nodup, tri4_denlats_nodup
#   --> exp/mono_swbd, mono_swbd_ali, tri1_swbd, tri1_swbd_ali, tri2_swbd, tri2_swbd_ali_100k_nodup, tri2_swbd_ali_nodup, tri3_swbd, tri3_swbd_ali_nodup, tri4_swbd, tri4_swbd_ali_nodup, tri4_swbd_denlats_nodup

# Data preparation
# Language model
if [ $stage -le 0 ]; then
  mkdir -p $data/local_swbd/train/
  [ -f $data/local_swbd/train/swb_ms98_transcriptions ] || ln -sf $swbd_dir/swb_ms98_transcriptions $data/local_swbd/train/
  local/swbd1_prepare_dict.sh $data/local_swbd/train $data/local_swbd/dict_nosp

  # Prepare Switchboard data. This command can also take a second optional argument
  # which specifies the directory to Switchboard documentations. Specifically, if
  # this argument is given, the script will look for the conv.tab file and correct
  # speaker IDs to the actual speaker personal identification numbers released in
  # the documentations. The documentations can be found here:
  # https://catalog.ldc.upenn.edu/docs/LDC97S62/
  # Note: if you are using this link, make sure you rename conv_tab.csv to conv.tab
  # after downloading.
  local/swbd1_data_prep.sh $data/local_swbd/train $data/local_swbd/dict_nosp $data/train_swbd_asr/ $swbd_dir/data
  
  utils/prepare_lang.sh $data/local_swbd/dict_nosp \
    "<unk>"  $data/local_swbd/lang_nosp $data/lang_swbd_nosp

  # Now train the language models. We are using SRILM and interpolating with an
  # LM trained on the Fisher transcripts (part 2 disk is currently missing; so
  # only part 1 transcripts ~700hr are used)
  
  # If you have the Fisher data, you can set this "fisher_dir" variable.
  fisher_dirs="/mnt/lv10/person/sre16/data/fisher/fe_03_tran"
  local/swbd1_train_lms.sh $data/local_swbd/train/text \
    $data/local_swbd/dict_nosp/lexicon.txt $data/local_swbd/lm $fisher_dirs
  
  # Compiles G for swbd trigram LM
  LM=$data/local_swbd/lm/sw1.o3g.kn.gz
  srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    $data/lang_swbd_nosp $LM $data/local_swbd/dict_nosp/lexicon.txt $data/lang_swbd_nosp_sw1_tg
 
  # Compiles const G for swbd+fisher 4gram LM, if it exists.
  LM=$data/local_swbd/lm/sw1_fsh.o4g.kn.gz
  has_fisher=true
  [ -f $LM ] || has_fisher=false
  if $has_fisher; then
    utils/build_const_arpa_lm.sh $LM $data/lang_swbd_nosp $data/lang_swbd_nosp_sw1_fsh_fg
  fi
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --nj $fea_nj --cmd "$train_cmd" --mfcc-config conf/mfcc_asr.conf \
    $data/train_swbd_asr $exp/make_mfcc/train_swbd_asr $mfccdir
  steps/compute_cmvn_stats.sh $data/train_swbd_asr $exp/make_mfcc/train_swbd_asr $mfccdir
  utils/fix_data_dir.sh $data/train_swbd_asr
fi


if [ $stage -le 2 ]; then
  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
  # the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
  # LM training data.   However, they will be in the lexicon, plus speakers
  # may overlap, so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first $data/train_swbd_asr 4000 $data/train_swbd_asr_dev # 5hr 6min
  n=$[`cat $data/train_swbd_asr/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $data/train_swbd_asr $n $data/train_swbd_asr_nodev
  
  # Now-- there are 260k utterances (313hr 23min), and we want to start the
  # monophone training on relatively short utterances (easier to align), but not
  # only the shortest ones (mostly uh-huh).  So take the 100k shortest ones, and
  # then take 30k random utterances from those (about 12hr)
  utils/subset_data_dir.sh --shortest $data/train_swbd_asr_nodev 100000 $data/train_swbd_asr_100kshort
  utils/subset_data_dir.sh $data/train_swbd_asr_100kshort 30000 $data/train_swbd_asr_30kshort
  
  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $data/train_swbd_asr_nodev 100000 $data/train_swbd_asr_100k
  utils/data/remove_dup_utts.sh 200 $data/train_swbd_asr_100k $data/train_swbd_asr_100k_nodup  # 110hr
 
  # Finally, the full training set:
  utils/data/remove_dup_utts.sh 300 $data/train_swbd_asr_nodev $data/train_swbd_asr_nodup  # 286hr
fi


if [ $stage -le 3 ]; then
  ## Starting basic training on MFCC features
  steps/train_mono.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_swbd_asr_30kshort $data/lang_swbd_nosp $exp/mono_swbd
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_swbd_asr_100k_nodup $data/lang_swbd_nosp $exp/mono_swbd $exp/mono_swbd_ali
fi


if [ $stage -le 4 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    3200 30000 $data/train_swbd_asr_100k_nodup $data/lang_swbd_nosp $exp/mono_swbd_ali $exp/tri1_swbd
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_swbd_asr_100k_nodup $data/lang_swbd_nosp $exp/tri1_swbd $exp/tri1_swbd_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
    4000 70000 $data/train_swbd_asr_100k_nodup $data/lang_swbd_nosp $exp/tri1_swbd_ali $exp/tri2_swbd
#   # The 100k_nodup data is used in neural net training.
#   steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
#     $data/train_swbd_asr_100k_nodup $data/lang_swbd_nosp $exp/tri2_swbd $exp/tri2_swbd_ali_100k_nodup

  # From now, we start using all of the data (except some duplicates of common
  # utterances, which don't really contribute much).
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_swbd_asr_nodup $data/lang_swbd_nosp $exp/tri2_swbd $exp/tri2_swbd_ali_nodup

  # Do another iteration of LDA+MLLT training, on all the data.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    6000 140000 $data/train_swbd_asr_nodup $data/lang_swbd_nosp $exp/tri2_swbd_ali_nodup $exp/tri3_swbd
fi


if [ $stage -le 5 ]; then
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  steps/get_prons.sh --cmd "$train_cmd" $data/train_swbd_asr_nodup $data/lang_swbd_nosp $exp/tri3_swbd
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    $data/local_swbd/dict_nosp $exp/tri3_swbd/pron_counts_nowb.txt $exp/tri3_swbd/sil_counts_nowb.txt \
    $exp/tri3_swbd/pron_bigram_counts_nowb.txt $data/local_swbd/dict
  
  utils/prepare_lang.sh $data/local_swbd/dict "<unk>" $data/local_swbd/lang $data/lang_swbd
  LM=$data/local_swbd/lm/sw1.o3g.kn.gz
  srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    $data/lang_swbd $LM $data/local_swbd/dict/lexicon.txt $data/lang_swbd_sw1_tg
  LM=$data/local_swbd/lm/sw1_fsh.o4g.kn.gz
  has_fisher=true
  [ -f $LM ] || has_fisher=false
  if $has_fisher; then
    utils/build_const_arpa_lm.sh $LM $data/lang_swbd $data/lang_swbd_sw1_fsh_fg
  fi
fi
 
 
if [ $stage -le 6 ]; then
  # Train tri4, which is LDA+MLLT+SAT, on all the (nodup) data.
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_swbd_asr_nodup $data/lang_swbd $exp/tri3_swbd $exp/tri3_swbd_ali_nodup
#   steps/train_sat.sh  --cmd "$train_cmd" \
#     11500 200000 $data/train_swbd_asr_nodup $data/lang_swbd $exp/tri3_swbd_ali_nodup $exp/tri4_swbd
  # Do not use too many senones
  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 100000 $data/train_swbd_asr_nodup $data/lang_swbd $exp/tri3_swbd_ali_nodup $exp/tri4_swbd

  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_swbd_asr_nodup $data/lang_swbd $exp/tri4_swbd $exp/tri4_swbd_ali_nodup
fi

 
