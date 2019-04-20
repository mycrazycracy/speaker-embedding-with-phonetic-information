#!/bin/bash

# This script generate features with silence and alignment, 
# according to the filtered features without silence.
# 1. WCMVN to the features
# 2. generate the alignments

cmd="run.pl"
nj=16
stage=0
norm_vars=false
center=true
compress=true
cmn_window=300

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 <src-data-dir> <src-ali-dir> <src-data-dir-nosil> <target-data-dir> <target-ali-dir> <target-ali-dir-nosil>"
  echo ""
  echo "Options:"
  echo "  --cmd <run.pl>"
  echo "  --stage <0>"
  echo "  --nj <16>"
  echo "  --center <true>"
  echo "  --norm-vars <false>"
  echo "  --compress <true>"
  echo "  --cmn-window <300>"
  exit 1
fi

srcdir=$1
alidir=$2
srcdir_nosil=$3
data=$4
dir=$5
dir_nosil=$6

for f in $srcdir/vad.scp $srcdir/feats.scp $srcdir_nosil/utt2spk ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $dir_nosil/log
mkdir -p $data
feat_dir=$(utils/make_absolute.sh $dir)

name=`basename $srcdir`

utils/filter_scp.pl $srcdir_nosil/utt2spk $srcdir/vad.scp > $data/vad.scp
utils/filter_scp.pl $srcdir_nosil/utt2spk $srcdir/feats.scp > $data/feats.scp
utils/filter_scp.pl $srcdir_nosil/utt2spk $srcdir/utt2spk > $data/utt2spk
cp $srcdir/spk2utt $data/spk2utt
cp $srcdir/wav.scp $data/wav.scp
[ -f $srcdir/segments ] && cp $srcdir/segments $data/segments
utils/fix_data_dir.sh $data

sdata_in=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

write_num_frames_opt="--write-num-frames=ark,t:$feat_dir/log/utt2num_frames.JOB"

$cmd JOB=1:$nj $dir/log/create_wcmvn_feats_${name}.JOB.log \
  apply-cmvn-sliding --norm-vars=$norm_vars --center=$center --cmn-window=$cmn_window \
  scp:${sdata_in}/JOB/feats.scp ark:- \| \
  copy-feats --compress=$compress $write_num_frames_opt ark:- \
  ark,scp:$feat_dir/wcmvn_feats_${name}.JOB.ark,$feat_dir/wcmvn_feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $feat_dir/wcmvn_feats_${name}.$n.scp || exit 1;
done > ${data}/feats.scp || exit 1

for n in $(seq $nj); do
  cat $feat_dir/log/utt2num_frames.$n || exit 1;
done > ${data}/utt2num_frames || exit 1
rm $feat_dir/log/utt2num_frames.*


num_ali_jobs=$(cat $alidir/num_jobs) || exit 1;

# with silence, just filter using the utterances 
for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
  copy-int-vector ark:- ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;

mv $dir/ali.scp $dir/ali.scp.bak
utils/filter_scp.pl $data/utt2spk $dir/ali.scp.bak > $dir/ali.scp 
gzip -c $dir/ali.ark > $dir/ali.gz
rm -f $dir/ali.ark $dir/ali.scp.bak > /dev/null

cp $alidir/final.mdl $dir/final.mdl
cp $alidir/tree $dir/tree


num_ali_jobs=$(cat $alidir/num_jobs) || exit 1;

for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
  copy-int-vector ark:- ark,scp:$dir_nosil/ali.ark.bak,$dir_nosil/ali.scp.bak || exit 1;

abs_dir=$(utils/make_absolute.sh $dir_nosil)
utils/filter_scp.pl $srcdir_nosil/utt2spk $srcdir/vad.scp > $srcdir_nosil/vad.scp
utils/filter_scp.pl $srcdir_nosil/utt2spk $dir_nosil/ali.scp.bak > $dir_nosil/ali.scp.new
select-voiced-ali scp:$abs_dir/ali.scp.new scp,s,cs:$srcdir_nosil/vad.scp ark,scp:$abs_dir/ali.ark,$abs_dir/ali.scp
gzip -c $dir_nosil/ali.ark > $dir_nosil/ali.gz
rm -f $dir_nosil/ali.ark.bak $dir_nosil/ali.scp.bak $dir_nosil/ali.ark $dir_nosil/ali.scp.new > /dev/null
 
cp $alidir/final.mdl $dir_nosil/final.mdl
cp $alidir/tree $dir_nosil/tree

# Some utterances are removed due to decoding fail. So the num of utterances in ali.scp may be smaller
# We fix the data according to the alignments 
mv $data/utt2spk $data/utt2spk.old
utils/filter_scp.pl $dir/ali.scp $data/utt2spk.old > $data/utt2spk
utils/fix_data_dir.sh $data

rm -rf ${srcdir_nosil}_ali
cp -r $srcdir_nosil ${srcdir_nosil}_ali
mv ${srcdir_nosil}_ali/utt2spk ${srcdir_nosil}_ali/utt2spk.old
utils/filter_scp.pl $dir_nosil/ali.scp ${srcdir_nosil}_ali/utt2spk.old > ${srcdir_nosil}_ali/utt2spk
utils/fix_data_dir.sh ${srcdir_nosil}_ali

