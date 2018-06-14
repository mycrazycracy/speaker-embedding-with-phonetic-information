#!/bin/bash

# Copyright 2012-2016 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.
#
# This script dumps egs with several frames of labels, controlled by the
# frames_per_eg config variable (default: 8).  This takes many times less disk
# space because typically we have 4 to 7 frames of context on the left and
# right, and this ends up getting shared.  This is at the expense of slightly
# higher disk I/O while training.

set -o pipefail
trap "" PIPE

# Begin configuration section.
cmd=run.pl
frame_subsampling_factor=1
frames_per_eg=8   # number of frames of labels per example.  more->less disk space and
                  # less time preparing egs, but more I/O during training.
                  # Note: may in general be a comma-separated string of alternative
                  # durations (more useful when using large chunks, e.g. for BLSTMs);
                  # the first one (the principal num-frames) is preferred.
left_context=4    # amount of left-context per eg (i.e. extra frames of input features
                  # not present in the output supervision).
right_context=4   # amount of right-context per eg.
left_context_initial=-1    # if >=0, left-context for first chunk of an utterance
right_context_final=-1     # if >=0, right-context for last chunk of an utterance
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_utts_subset=300     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.
num_valid_frames_combine=0 # #valid frames for combination weights at the very end.
num_train_frames_combine=60000 # # train frames for the above.
num_frames_diagnostic=10000 # number of frames for "compute_prob" jobs
samples_per_iter=400000 # this is the target number of egs in each archive of egs
                        # (prior to merging egs).  We probably should have called
                        # it egs_per_iter. This is just a guideline; it will pick
                        # a number that divides the number of samples in the
                        # entire data.

stage=0
nj=6         # This should be set to the maximum number of jobs you are
             # comfortable to run in parallel; you can increase it if your disk
             # speed is greater and you have more machines.
srand=0     # rand seed for nnet3-copy-egs and nnet3-shuffle-egs

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <data> <ali-dir> <egs-dir>"
  echo " e.g.: $0 data/train exp/tri3_ali exp/tri4_nnet/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --samples-per-iter <#samples;400000>             # Target number of egs per archive (option is badly named)"
  echo "  --frames-per-eg <frames;8>                       # number of frames per eg on disk"
  echo "                                                   # May be either a single number or a comma-separated list"
  echo "                                                   # of alternatives (useful when training LSTMs, where the"
  echo "                                                   # frames-per-eg is the chunk size, to get variety of chunk"
  echo "                                                   # sizes).  The first in the list is preferred and is used"
  echo "                                                   # when working out the number of archives etc."
  echo "  --left-context <int;4>                           # Number of frames on left side to append for feature input"
  echo "  --right-context <int;4>                          # Number of frames on right side to append for feature input"
  echo "  --left-context-initial <int;-1>                  # If >= 0, left-context for first chunk of an utterance"
  echo "  --right-context-final <int;-1>                   # If >= 0, right-context for last chunk of an utterance"
  echo "  --num-frames-diagnostic <#frames;4000>           # Number of frames used in computing (train,valid) diagnostics"
  echo "  --num-valid-frames-combine <#frames;10000>       # Number of frames used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
alidir=$2
dir=$3

# Check some files.
for f in $data/feats.scp $alidir/final.mdl $alidir/tree ; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log $dir/info
cp $alidir/tree $dir


num_utts=$(cat $data/utt2spk | wc -l)
if ! [ $num_utts -gt $[$num_utts_subset*4] ]; then
  echo "$0: number of utterances $num_utts in your training data is too small versus --num-utts-subset=$num_utts_subset"
  echo "... you probably have so little data that it doesn't make sense to train a neural net."
  exit 1
fi

# Get list of validation utterances.
awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset \
    > $dir/valid_uttlist || exit 1;

if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
  echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
  echo "include all perturbed versions of the same 'real' utterances."
  mv $dir/valid_uttlist $dir/valid_uttlist.tmp
  utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $dir/uniq2utt
  cat $dir/valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
    sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
    awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid_uttlist
  rm $dir/uniq2utt $dir/valid_uttlist.tmp
fi

awk '{print $1}' $data/utt2spk | utils/filter_scp.pl --exclude $dir/valid_uttlist | \
   utils/shuffle_list.pl | head -$num_utts_subset > $dir/train_subset_uttlist || exit 1;


## Set up features.
echo "$0: feature type is raw"

feats="scp,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp |"
valid_feats="scp,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp |"
train_subset_feats="scp,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp |"

if [ $stage -le 1 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/info/num_frames
  echo "$0: working out feature dim"
  feats_one="$(echo $feats | sed s/JOB/1/g)"
  if feat_dim=$(feat-to-dim "$feats_one" - 2>/dev/null); then
    echo $feat_dim > $dir/info/feat_dim
  else # run without redirection to show the error.
    feat-to-dim "$feats_one" -; exit 1
  fi
else
  num_frames=$(cat $dir/info/num_frames) || exit 1;
  feat_dim=$(cat $dir/info/feat_dim) || exit 1;
fi


# the first field in frames_per_eg (which is a comma-separated list of numbers)
# is the 'principal' frames-per-eg, and for purposes of working out the number
# of archives we assume that this will be the average number of frames per eg.
frames_per_eg_principal=$(echo $frames_per_eg | cut -d, -f1)

# the + 1 is to round up, not down... we assume it doesn't divide exactly.
num_archives=$[$num_frames/($frames_per_eg_principal*$samples_per_iter)+1]
if [ $num_archives -eq 1 ]; then
  echo "*** $0: warning: the --frames-per-eg is too large to generate one archive with"
  echo "*** as many as --samples-per-iter egs in it.  Consider reducing --frames-per-eg."
  sleep 4
fi

# We may have to first create a smaller number of larger archives, with number
# $num_archives_intermediate, if $num_archives is more than the maximum number
# of open filehandles that the system allows per process (ulimit -n).
# This sometimes gives a misleading answer as GridEngine sometimes changes that
# somehow, so we limit it to 512.
max_open_filehandles=$(ulimit -n) || exit 1
[ $max_open_filehandles -gt 512 ] && max_open_filehandles=512
num_archives_intermediate=$num_archives
archives_multiple=1
while [ $[$num_archives_intermediate+4] -gt $max_open_filehandles ]; do
  archives_multiple=$[$archives_multiple+1]
  num_archives_intermediate=$[$num_archives/$archives_multiple+1];
done
# now make sure num_archives is an exact multiple of archives_multiple.
num_archives=$[$archives_multiple*$num_archives_intermediate]

# here we have the num of archives
echo $num_archives >$dir/info/num_archives
echo $frames_per_eg >$dir/info/frames_per_eg

# Work out the number of egs per archive
egs_per_archive=$[$num_frames/($frames_per_eg_principal*$num_archives)]
! [ $egs_per_archive -le $samples_per_iter ] && \
  echo "$0: script error: egs_per_archive=$egs_per_archive not <= samples_per_iter=$samples_per_iter" \
  && exit 1;

echo $egs_per_archive > $dir/info/egs_per_archive

echo "$0: creating $num_archives archives, each with $egs_per_archive egs, with"
echo "$0:   $frames_per_eg labels per example, and (left,right) context = ($left_context,$right_context)"

# we don't need left_context_initial and right_context_final now
if [ $left_context_initial -ge 0 ] || [ $right_context_final -ge 0 ]; then
  echo "$0:   ... and (left-context-initial,right-context-final) = ($left_context_initial,$right_context_final)"
  echo "No need for left_context_initial and right_context_final now. Strange, Emmm..."
  exit 1
fi



if [ -e $dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See
  # utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $dir/egs.$x.ark; done)
  for x in $(seq $num_archives_intermediate); do
    utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/egs_orig.$y.$x.ark; done)
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: copying data alignments"
  cp $alidir/ali.scp $dir/ali.scp
  # extract to the original dir, since scp point to there
  gunzip -c $alidir/ali.gz > $alidir/ali.ark

#  for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
#    copy-int-vector ark:- ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
fi

egs_opts="--left-context=$left_context --right-context=$right_context --compress=$compress --num-frames=$frames_per_eg"

# no need currently
[ $left_context_initial -ge 0 ] && egs_opts="$egs_opts --left-context-initial=$left_context_initial"
[ $right_context_final -ge 0 ] && egs_opts="$egs_opts --right-context-final=$right_context_final"


echo $left_context > $dir/info/left_context
echo $right_context > $dir/info/right_context
echo $left_context_initial > $dir/info/left_context_initial
echo $right_context_final > $dir/info/right_context_final

num_pdfs=$(tree-info --print-args=false $alidir/tree | grep num-pdfs | awk '{print $2}')
echo $num_pdfs > $dir/info/num_pdfs

if [ $stage -le 3 ]; then
  echo "$0: Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null
  echo "$0: ... extracting validation and training-subset alignments."

  # do the filtering just once, as ali.scp may be long.
  utils/filter_scp.pl <(cat $dir/valid_uttlist $dir/train_subset_uttlist) \
    <$dir/ali.scp >$dir/ali_special.scp

  $cmd $dir/log/create_valid_subset.log \
    utils/filter_scp.pl $dir/valid_uttlist $dir/ali_special.scp \| \
    ali-to-pdf $alidir/final.mdl scp:- ark:- \| \
    ali-to-post ark:- ark:- \| \
    nnet3-get-egs --num-pdfs=$num_pdfs --frame-subsampling-factor=$frame_subsampling_factor \
      $egs_opts "$valid_feats" \
      ark,s,cs:- "ark:$dir/valid_all.egs" || touch $dir/.error &

  $cmd $dir/log/create_train_subset.log \
    utils/filter_scp.pl $dir/train_subset_uttlist $dir/ali_special.scp \| \
    ali-to-pdf $alidir/final.mdl scp:- ark:- \| \
    ali-to-post ark:- ark:- \| \
    nnet3-get-egs --num-pdfs=$num_pdfs --frame-subsampling-factor=$frame_subsampling_factor \
      $egs_opts "$train_subset_feats" \
      ark,s,cs:- "ark:$dir/train_subset_all.egs" || touch $dir/.error &
  wait;
  [ -f $dir/.error ] && echo "Error detected while creating train/valid egs" && exit 1
  echo "... Getting subsets of validation examples for diagnostics and combination."
  valid_diagnostic_output="ark,scp:$dir/valid_diagnostic.egs,$dir/valid_diagnostic.scp"
  train_diagnostic_output="ark,scp:$dir/train_diagnostic.egs,$dir/train_diagnostic.scp"
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet3-subset-egs --n=$[$num_valid_frames_combine/$frames_per_eg_principal] ark:$dir/valid_all.egs \
      ark:$dir/valid_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet3-subset-egs --n=$[$num_frames_diagnostic/$frames_per_eg_principal] ark:$dir/valid_all.egs \
    $valid_diagnostic_output || touch $dir/.error &

  $cmd $dir/log/create_train_subset_combine.log \
    nnet3-subset-egs --n=$[$num_train_frames_combine/$frames_per_eg_principal] ark:$dir/train_subset_all.egs \
      ark:$dir/train_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet3-subset-egs --n=$[$num_frames_diagnostic/$frames_per_eg_principal] ark:$dir/train_subset_all.egs \
    $train_diagnostic_output || touch $dir/.error &
  wait
  sleep 5  # wait for file system to sync.

  cat $dir/valid_combine.egs $dir/train_combine.egs  | \
  nnet3-copy-egs ark:- ark,scp:$dir/combine.egs,$dir/combine.scp

  for f in $dir/{combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
  rm $dir/valid_all.egs $dir/train_subset_all.egs $dir/{train,valid}_combine.egs
fi

if [ $stage -le 4 ]; then
  # create egs_orig.*.*.ark; the first index goes to $nj,
  # the second to $num_archives_intermediate.

  egs_list=
  for n in $(seq $num_archives_intermediate); do
    egs_list="$egs_list ark:$dir/egs_orig.JOB.$n.ark"
  done
  echo "$0: Generating training examples on disk"
  # The examples will go round-robin to egs_list.
  $cmd JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet3-get-egs --num-pdfs=$num_pdfs --frame-subsampling-factor=$frame_subsampling_factor \
    $egs_opts "$feats" \
    "ark,s,cs:filter_scp.pl $sdata/JOB/utt2spk $dir/ali.scp | ali-to-pdf $alidir/final.mdl scp:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet3-copy-egs --random=true --srand=\$[JOB+$srand] ark:- $egs_list || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: recombining and shuffling order of archives on disk"
  # combine all the "egs_orig.*.JOB.scp" (over the $nj splits of the data) and
  # shuffle the order, writing to the egs.JOB.ark

  # the input is a concatenation over the input jobs.
  egs_list=
  for n in $(seq $nj); do
    egs_list="$egs_list $dir/egs_orig.$n.JOB.ark"
  done

  if [ $archives_multiple == 1 ]; then # normal case.
    output_archive="ark,scp:$dir/egs.JOB.ark,$dir/egs.JOB.scp"

    $cmd --max-jobs-run $nj JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=\$[JOB+$srand] "ark:cat $egs_list|" $output_archive  || exit 1;

    #concatenate egs.JOB.scp in single egs.scp
    rm $dir/egs.scp 2> /dev/null || true
    for j in $(seq $num_archives_intermediate); do
      cat $dir/egs.$j.scp || exit 1;
    done > $dir/egs.scp || exit 1;
  else
    # we need to shuffle the 'intermediate archives' and then split into the
    # final archives.  we create soft links to manage this splitting, because
    # otherwise managing the output names is quite difficult (and we don't want
    # to submit separate queue jobs for each intermediate archive, because then
    # the --max-jobs-run option is hard to enforce).
    output_archives="$(for y in $(seq $archives_multiple); do echo ark,scp:$dir/egs.JOB.$y.ark,$dir/egs.JOB.$y.scp; done)"
    for x in $(seq $num_archives_intermediate); do
      for y in $(seq $archives_multiple); do
        archive_index=$[($x-1)*$archives_multiple+$y]
        # egs.intermediate_archive.{1,2,...}.ark will point to egs.archive.ark
        ln -sf egs.$archive_index.ark $dir/egs.$x.$y.ark || exit 1
      done
    done
    $cmd --max-jobs-run $nj JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=\$[JOB+$srand] "ark:cat $egs_list|" ark:- \| \
      nnet3-copy-egs ark:- $output_archives || exit 1;

    #concatenate egs.JOB.scp in single egs.scp
    rm $dir/egs.scp 2> /dev/null || true
    for j in $(seq $num_archives_intermediate); do
      for y in $(seq $num_archives_intermediate); do
        cat $dir/egs.$j.$y.scp || exit 1;
      done
    done > $dir/egs.scp || exit 1;
  fi
fi

if [ $frame_subsampling_factor -ne 1 ]; then
  echo $frame_subsampling_factor > $dir/info/frame_subsampling_factor
fi

if [ $stage -le 6 ]; then
  echo "$0: removing temporary archives"
  for x in $(seq $nj); do
    for y in $(seq $num_archives_intermediate); do
      file=$dir/egs_orig.$x.$y.ark
      [ -L $file ] && rm $(utils/make_absolute.sh $file)
      rm $file
    done
  done
  if [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $dir/egs.*.*.ark; do rm $f; done
  fi
  echo "$0: removing temporary alignments and transforms"
  # Ignore errors below because trans.* might not exist.
  rm $dir/{ali,trans}.{ark,scp} 2>/dev/null
  rm $alidir/ali.ark 2>/dev/null
fi

echo "$0: Finished preparing training examples"

