#!/bin/bash

# Copyright      2017 Johns Hopkins University (Author: Daniel Povey)
#                2017 Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017 David Snyder
# Apache 2.0
#
# This script dumps training examples (egs) for multiclass xvector training.
# These egs consist of a data chunk and a zero-based speaker label.
# Each archive of egs has, in general, a different input chunk-size.
# We don't mix together different lengths in the same archive, because it
# would require us to repeatedly run the compilation process within the same
# training job.
#
# This script, which will generally be called from other neural net training
# scripts, extracts the training examples used to train the neural net (and
# also the validation examples used for diagnostics), and puts them in
# separate archives.


# Begin configuration section.
cmd=run.pl
stage=0
compress=true
phone_set=
phone=
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <alidir> <xvec_egs> <egs>"
  echo " e.g.: $0 data/train exp/ali exp/xvector_a/egs exp/xvector_b/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --phone-set                                      # how to bind the phone into classes (only takes effect when phone is given)"
  echo "  --phone                                          # phone level alignment if given. usually to be lang/phone.txt to map int to symbol"

  exit 1;
fi

data=$1
alidir=$2
xvec_dir=$3
dir=$4

for f in $data/feats.scp $data/vad.scp $xvec_dir/temp/ranges.1 $xvec_dir/temp/outputs.1; do
  [ ! -f $f ] && echo "$0: expected file $f" && exit 1;
done

nj=$(ls $xvec_dir/temp/ranges.* | wc -l) || exit 1
feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1
mkdir -p $dir/info 

if [ $stage -le 0 ]; then
  cp -r $xvec_dir/temp $dir
  cp $xvec_dir/info/left_context $dir/info/left_context
  cp $xvec_dir/info/right_context $dir/info/right_context
  echo '1' > $dir/info/frames_per_eg
  echo $feat_dim > $dir/info/feat_dim
  cp $xvec_dir/info/num_frames $dir/info/num_frames
  cp $xvec_dir/info/num_archives $dir/info/num_archives
  cp $xvec_dir/info/num_diagnostic_archives $dir/info/num_diagnostic_archives
  cp $xvec_dir/temp/archive_chunk_lengths $dir/temp/archive_chunk_lengths
  cp $xvec_dir/pdf2num $dir/pdf2num
fi

temp=$dir/temp

num_spks=$(awk '{print $2}' $temp/utt2int | sort | uniq -c | wc -l)

num_train_frames=$(awk '{n += $2} END{print n}' <$temp/utt2num_frames.train)
num_train_subset_frames=$(awk '{n += $2} END{print n}' <$temp/utt2num_frames.train_subset)
num_train_archives=$(cat $dir/info/num_archives)
num_diagnostic_archives=$(cat $xvec_dir/info/num_diagnostic_archives)

echo "$0: Producing $num_train_archives archives with alignment from xvector egs"

if [ $nj -gt $num_train_archives ]; then
  echo "$0: Reducing num-jobs $nj to number of training archives $num_train_archives"
  nj=$num_train_archives
fi

num_ali_jobs=$(cat $alidir/num_jobs)

if [ $stage -le 1 ]; then
  if [ -z $phone ]; then
    for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
      ali-to-pdf $alidir/final.mdl ark:- ark,scp:$dir/temp/pdf.ark,$dir/temp/pdf.scp || exit 1;
    utils/filter_scp.pl $data/utt2spk $dir/temp/pdf.scp > $dir/temp/pdf.scp.filter
    select-voiced-ali scp:$dir/temp/pdf.scp.filter scp,s,cs:$data/vad.scp ark:- | \
      ali-to-post ark:- ark:$dir/temp/post.ark.old
    # post-squeeze cannot use pipe to input, because there is two pass in the program. 
    post-squeeze ark:$dir/temp/post.ark.old $dir/pdf_id_map ark,scp:$dir/temp/post.ark,$dir/temp/post.scp
    rm -f $dir/temp/pdf.ark $dir/temp/pdf.scp $dir/temp/pdf.scp.filter $dir/temp/post.ark.old
  else
    for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
      ali-to-phones --per-frame $alidir/final.mdl ark:- ark,scp:$dir/temp/phones.ark,$dir/temp/phones.scp
    utils/filter_scp.pl $data/utt2spk $dir/temp/phones.scp > $dir/temp/phones.scp.filter
    select-voiced-ali scp:$dir/temp/phones.scp.filter scp,s,cs:$data/vad.scp ark,t:- | \
      int2sym.pl -f 2- $phone - > $dir/temp/phones.txt
    python utils/convert_phone_set.py $phone_set $dir/temp/phones.txt $dir/temp/phone_set.txt
    ali-to-post ark:$dir/temp/phone_set.txt ark:$dir/temp/post.ark.old
    post-squeeze ark:$dir/temp/post.ark.old $dir/pdf_id_map ark,scp:$dir/temp/post.ark,$dir/temp/post.scp
    rm -f $dir/temp/phones.ark $dir/temp/phones.scp $dir/temp/phones.scp.filter $dir/temp/phones.txt $dir/temp/phone_set.txt $dir/temp/post.ark.old
  fi
  nnet3-post-count scp:$dir/temp/post.scp > $dir/num_pdfs
fi

num_pdfs=$(cat $dir/num_pdfs)
echo "$0: There are $num_pdfs pdf-ids in total."

# The script assumes you've prepared the features ahead of time.
feats="scp,s,cs:utils/filter_scp.pl $temp/ranges.JOB $data/feats.scp |"
train_subset_feats="scp,s,cs:utils/filter_scp.pl $temp/train_subset_ranges.1 $data/feats.scp |"
valid_feats="scp,s,cs:utils/filter_scp.pl $temp/valid_ranges.1 $data/feats.scp |"

alifeats="scp,s,cs:utils/filter_scp.pl $temp/ranges.JOB $dir/temp/post.scp |"
train_subset_alifeats="scp,s,cs:utils/filter_scp.pl $temp/train_subset_ranges.1 $dir/temp/post.scp |" 
valid_alifeats="scp,s,cs:utils/filter_scp.pl $temp/valid_ranges.1 $dir/temp/post.scp |"

if [ $stage -le 2 ]; then
  # the path in outputs.* should be changed to the current path
  for g in $(seq $nj); do
    sed -i "s:$xvec_dir:$dir:g" $temp/outputs.$g
  done
  for f in `ls $temp/train_subset_outputs.*`; do
    sed -i "s:$xvec_dir:$dir:g" $f
  done
  for f in `ls $temp/valid_outputs.*`; do
    sed -i "s:$xvec_dir:$dir:g" $f
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: Generating training examples on disk"
  rm $dir/.error 2>/dev/null
  for g in $(seq $nj); do
    outputs=$(awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/outputs.$g)
    $cmd $dir/log/train_create_examples.$g.log \
      nnet3-xvector-get-egs-with-post --compress=$compress --num-spks=$num_spks --num-pdfs=$num_pdfs $temp/ranges.$g \
      "`echo $feats | sed s/JOB/$g/g`" "`echo $alifeats | sed s/JOB/$g/g`" $outputs || touch $dir/.error &
  done
  wait

  train_subset_outputs=$(awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/train_subset_outputs.1)
  echo "$0: Generating training subset examples on disk"
  $cmd $dir/log/train_subset_create_examples.1.log \
    nnet3-xvector-get-egs-with-post --compress=$compress --num-spks=$num_spks --num-pdfs=$num_pdfs $temp/train_subset_ranges.1 \
    "$train_subset_feats" "$train_subset_alifeats" $train_subset_outputs || touch $dir/.error &
  
  valid_outputs=$(awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/valid_outputs.1)
  echo "$0: Generating validation examples on disk"
  $cmd $dir/log/valid_create_examples.1.log \
    nnet3-xvector-get-egs-with-post --compress=$compress --num-spks=$num_spks --num-pdfs=$num_pdfs $temp/valid_ranges.1 \
    "$valid_feats" "$valid_alifeats" $valid_outputs || touch $dir/.error &
  wait
  if [ -f $dir/.error ]; then
    echo "$0: Problem detected while dumping examples"
    exit 1
  fi
fi

if [ $stage -le 4 ]; then
  echo "$0: Shuffling order of archives on disk"
  $cmd --max-jobs-run $nj JOB=1:$num_train_archives $dir/log/shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:$dir/egs_temp.JOB.ark \
    ark,scp:$dir/egs.JOB.ark,$dir/egs.JOB.scp || exit 1;

  $cmd --max-jobs-run $nj JOB=1:$num_diagnostic_archives $dir/log/train_subset_shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:$dir/train_subset_egs_temp.JOB.ark \
    ark,scp:$dir/train_diagnostic_egs.JOB.ark,$dir/train_diagnostic_egs.JOB.scp || exit 1;

  $cmd --max-jobs-run $nj JOB=1:$num_diagnostic_archives $dir/log/valid_shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:$dir/valid_egs_temp.JOB.ark \
    ark,scp:$dir/valid_egs.JOB.ark,$dir/valid_egs.JOB.scp || exit 1;
fi

if [ $stage -le 5 ]; then
  for file in $(for x in $(seq $num_diagnostic_archives); do echo $dir/train_subset_egs_temp.$x.ark; done) \
    $(for x in $(seq $num_diagnostic_archives); do echo $dir/valid_egs_temp.$x.ark; done) \
    $(for x in $(seq $num_train_archives); do echo $dir/egs_temp.$x.ark; done); do
    [ -L $file ] && rm $(readlink -f $file)
    rm $file
  done
  rm -rf $dir/valid_diagnostic.scp $dir/train_diagnostic.scp
  for x in $(seq $num_diagnostic_archives); do
    cat $dir/train_diagnostic_egs.$x.scp >> $dir/train_diagnostic.scp
    cat $dir/valid_egs.$x.scp >> $dir/valid_diagnostic.scp
  done
  # TODO: the combine egs?
  ln -sf train_diagnostic.scp $dir/combine.scp
fi

echo "$0: Finished preparing training examples"
