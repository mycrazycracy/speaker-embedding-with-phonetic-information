#!/bin/bash

# The separate egs for am and xvector training are generated
# The combination and shuffle of the combined egs are accessed using nnet3-merge-am-xvec-egs

cmd=run.pl
nj=6
am_feat_dir=
am_ali_dir=
xvec_feat_dir=
samples_per_iter=400000
frames_per_eg=8
min_frames_per_chunk=200
max_frames_per_chunk=400
repeats_per_spk=5000
num_train_archives=150
num_heldout_utts=1000
stage=0


if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 1 ]; then
  echo "Usage: $0 --cmd run.pl --am-feat-dir <am_feat_dir> --am-ali-dir <ali_dir> --xvec-feat-dir <xvec_feat_dir> <nnet-dir>"
  echo "  "
  echo "  --cmd "
  echo "  --am-feat-dir"
  echo "  --am-ali-dir"
  echo "  --xvec-feat-dir"
  echo "  --samples-per-iter"
  echo "  --frames-per-eg"
  echo "  --min-frames-per-chunk"
  echo "  --max-frames-per-chunk"
  echo "  --repeats-per-spk"
  echo "  --num-train-archives"
  echo "  --num-heldout-utts"
fi

[ -z $am_feat_dir ] && echo "need am_feat_dir" && exit 1
[ -z $am_ali_dir ] && echo "need am_ali_dir" && exit 1
[ -z $xvec_feat_dir ] && echo "need xvec_feat_dir" && exit 1

nnetdir=$1

# The features should be processed beforehead (e.g. cmvn, delete nonspeech frames)

# keep the num of ark between 50 to 150 (as recommanded in xvector recipe)

# AM egs
am_left_context=$(grep 'model_left_context' $nnetdir/configs/vars_am | cut -d '=' -f 2)
am_right_context=$(grep 'model_right_context' $nnetdir/configs/vars_am | cut -d '=' -f 2)

if [ $stage -le 0 ]; then
sid/nnet3_cvector/cvector/get_egs_am.sh --cmd "$cmd" \
  --nj $nj \
  --left-context $am_left_context \
  --right-context $am_right_context \
  --stage 0 \
  --samples-per-iter $samples_per_iter \
  --frames-per-eg $frames_per_eg \
  $am_feat_dir $am_ali_dir $nnetdir/egs_am
fi

# Xvector egs
if [ $stage -le 1 ]; then
sid/nnet3_cvector/cvector/get_egs_xvec.sh --cmd "$cmd" \
  --nj $nj \
  --stage 0 \
  --min-frames-per-chunk $min_frames_per_chunk \
  --max-frames-per-chunk $max_frames_per_chunk \
  --num-train-archives $num_train_archives \
  --repeats-per-spk $repeats_per_spk \
  --num-heldout-utts $num_heldout_utts \
  $xvec_feat_dir $nnetdir/egs_xvec
  # --am-egs-dir $nnetdir/egs_am \
fi 

