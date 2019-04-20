#!/bin/bash

# This script generate features with silence and alignment, 
# according to the filtered features without silence.
# 1. WCMVN to the features
# 2. generate the alignments

cmd="run.pl"

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <src-data-dir> <src-ali-dir> <data-dir-filter> <ali-dir-nosil>"
  echo ""
  exit 1
fi

# After alignment, some utterances will fail and are excluded from the scp.
# So we filter the data-dir and only the successful utterances remain

srcdir=$1
alidir=$2
filtdir=$3
dir=$4

for f in $srcdir/vad.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
feat_dir=$(utils/make_absolute.sh $dir)

num_ali_jobs=$(cat $alidir/num_jobs) || exit 1;
cp $alidir/final.mdl $dir/final.mdl
cp $alidir/tree $dir/tree

for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
  copy-int-vector ark:- ark,scp:$dir/ali.ark.bak,$dir/ali.scp.bak || exit 1;

# we get the phone alignment per frame
ali-to-phones --per-frame $alidir/final.mdl ark:$feat_dir/ali.ark.bak ark:- | \
  select-voiced-ali ark:- scp,s,cs:$srcdir/vad.scp ark,scp:$feat_dir/phones.ark,$feat_dir/phones.scp || exit 1;

# we get the transition-id alignment
select-voiced-ali scp:$feat_dir/ali.scp.bak scp,s,cs:$srcdir/vad.scp ark,scp:$feat_dir/ali.ark,$feat_dir/ali.scp
gzip -c $dir/ali.ark > $dir/ali.gz

rm -f $dir/ali.ark.bak $dir/ali.scp.bak $dir/ali.ark > /dev/null

rm -rf ${filtdir}_ali
cp -r $filtdir ${filtdir}_ali
mv ${filtdir}_ali/utt2spk ${filtdir}_ali/utt2spk.old
utils/filter_scp.pl $dir/ali.scp ${filtdir}_ali/utt2spk.old > ${filtdir}_ali/utt2spk
utils/fix_data_dir.sh ${filtdir}_ali

