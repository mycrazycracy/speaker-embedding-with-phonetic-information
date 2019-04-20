#!/bin/bash

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
phone=
phone_set=
use_map=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: $0 <info-dir> <nnet-dir> <output-node> <data> <data-nosil> <ali-dir> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --phone-set                                      # how to bind the phone into classes (only takes effect when phone is given)"
  echo "  --phone                                          # phone level alignment if given. usually to be lang/phone.txt to map int to symbol"
  echo "  --use-map                                        # the map file used in the network training example generation"
fi

infodir=$1
srcdir=$2
output_node=$3
data=$4
data_nosil=$5
alidir=$6
dir=$7

for f in $data/utt2spk $data/vad.scp $data_nosil/feats.scp $alidir/num_jobs $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

[ -z $use_map ] && "--use-map must be specified to the same file used in the network training." && exit 1

mkdir -p $dir/temp

if [ $stage -le 0 ]; then
  echo "$0: extract ali"
  num_ali_jobs=$(cat $alidir/num_jobs)
  if [ -z $phone ]; then
    for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
      ali-to-pdf $alidir/final.mdl ark:- ark,scp:$dir/temp/pdf.ark,$dir/temp/pdf.scp || exit 1;
    utils/filter_scp.pl $data/utt2spk $dir/temp/pdf.scp > $dir/temp/pdf.scp.filter
    select-voiced-ali scp:$dir/temp/pdf.scp.filter scp,s,cs:$data/vad.scp ark:- | \
      ali-to-post ark:- ark:$dir/temp/post.ark.old
    # post-squeeze cannot use pipe to input, because there is two pass in the program. 
    post-squeeze --use-map=$use_map ark:$dir/temp/post.ark.old ark,scp:$dir/temp/post.ark,$dir/temp/post.scp
    rm -f $dir/temp/pdf.ark $dir/temp/pdf.scp $dir/temp/pdf.scp.filter $dir/temp/post.ark.old
  else
    for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
      ali-to-phones --per-frame $alidir/final.mdl ark:- ark,scp:$dir/temp/phones.ark,$dir/temp/phones.scp
    utils/filter_scp.pl $data/utt2spk $dir/temp/phones.scp > $dir/temp/phones.scp.filter
    select-voiced-ali scp:$dir/temp/phones.scp.filter scp,s,cs:$data/vad.scp ark,t:- | \
      int2sym.pl -f 2- $phone - > $dir/temp/phones.txt
    python utils/convert_phone_set.py $phone_set $dir/temp/phones.txt $dir/temp/phone_set.txt
    ali-to-post ark:$dir/temp/phone_set.txt ark:$dir/temp/post.ark.old
    post-squeeze --use-map=$use_map ark:$dir/temp/post.ark.old ark,scp:$dir/temp/post.ark,$dir/temp/post.scp
    rm -f $dir/temp/phones.ark $dir/temp/phones.scp $dir/temp/phones.scp.filter $dir/temp/phones.txt $dir/temp/phone_set.txt $dir/temp/post.ark.old
  fi
fi

echo "$0: extract output for node $output_node"

mkdir -p $dir/log

utils/split_data.sh $data_nosil $nj
echo "$0: extracting xvectors for $data_nosil"
sdata=$data_nosil/split$nj/JOB

name=`basename $data_nosil`

# disable GPUs
export CUDA_VISIBLE_DEVICES=""

if [ $stage -le 1 ]; then
  echo "$0: extracting xvectors from nnet"
  utils/filter_scps.pl JOB=1:$nj \
    $sdata/utt2spk $dir/temp/post.scp $sdata/post.scp || exit 1;
  $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
    python $srcdir/config/extract_embedding_with_text.py --info_dir $infodir ${sdata}/feats.scp ${sdata}/post.scp ${dir}/xvector_$name.JOB.ark $srcdir $output_node || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: combining xvectors across jobs"
  $cmd JOB=1:$nj ${dir}/log/generate_scp.JOB.log \
    copy-vector ark:${dir}/xvector_$name.JOB.ark ark,scp:${dir}/tf_xvector_$name.JOB.ark,${dir}/tf_xvector_$name.JOB.scp || exit 1;
  for j in $(seq $nj); do cat $dir/tf_xvector_$name.$j.scp; done >$dir/tf_xvector_$name.scp || exit 1;
fi

if [ $stage -le 3 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors.
  echo "$0: computing mean of xvectors for each speaker"
  $cmd $dir/log/speaker_mean.log \
    ivector-mean ark:$data/spk2utt scp:$dir/tf_xvector_$name.scp \
      ark,scp:$dir/tf_spk_xvector_$name.ark,$dir/tf_spk_xvector_$name.scp ark,t:$dir/num_utts.ark || exit 1;
  for j in $(seq $nj); do rm -f ${dir}/xvector_$name.$j.ark; done || exit 1
fi

