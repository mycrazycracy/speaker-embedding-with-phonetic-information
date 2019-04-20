#!/bin/bash

# This script do:
# 1. WCMVN to the features
# 2. delete nonspeech frames
# 3. filter the features

cmd="run.pl"
nj=16
stage=0
norm_vars=false
center=true
compress=true
cmn_window=300
min_len=200
min_num_utts=8
vad_dir=
ali_dir=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <src-data-dir> <target-data-dir> <target-ali-dir> <target-data-dir-nosil> <target-ali-dir-nosil>"
  echo ""
  echo "Options:"
  echo "  --cmd <run.pl>"
  echo "  --vad-dir "
  echo "  --ali-dir "
  echo "  --stage <0>"
  echo "  --nj <16>"
  echo "  --stage <0>"
  echo "  --center <true>"
  echo "  --norm-vars <false>"
  echo "  --compress <true>"
  echo "  --cmn-window <300>"
  echo "  --min-len <200>"
  echo "  --min-num-utts <8>"
fi

srcdir=$1
data=$2
dir=$3
data_nosil=$4
dir_nosil=$5

[ -z $vad_dir ] && echo "vad-dir must be specified" && exit 1
[ $vad_dir = $srcdir ] && echo "vad-dir must be different with srcdir. You can copy them." && exit 1

for f in $srcdir/feats.scp $vad_dir/vad.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done


# Set various variables.
mkdir -p $dir/log
mkdir -p $dir_nosil/log
mkdir -p $data
mkdir -p $data_nosil
feat_dir=$(utils/make_absolute.sh $dir)
feat_nosil_dir=$(utils/make_absolute.sh $dir_nosil)

name=`basename $srcdir`

# copy the vad file, and check they have the same order
utils/filter_scp.pl $srcdir/utt2spk $vad_dir/vad.scp > $srcdir/vad.scp
cut -d ' ' -f 1 $srcdir/feats.scp > $srcdir/feats.scp.bak
cut -d ' ' -f 1 $srcdir/vad.scp > $srcdir/vad.scp.bak
[ `diff $srcdir/feats.scp.bak $srcdir/vad.scp.bak | wc -l` -gt 0 ] && echo "Not all features in $srcdir have VAD info" && exit 1
rm -f $srcdir/feats.scp.bak $srcdir/vad.scp.bak 


sdata_in=$srcdir/split$nj;
utils/split_data.sh $srcdir $nj || exit 1;

## First, do WCMVN
cp $srcdir/utt2spk $data/utt2spk
cp $srcdir/spk2utt $data/spk2utt
cp $srcdir/wav.scp $data/wav.scp
cp $srcdir/vad.scp $data/vad.scp

[ -f $srcdir/segments ] && cp $srcdir/segments $data/segments

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

# remove features that are too short 
echo "remove features too short"
cp -r $data ${data}.bak   # we need a backup here
mv $data/utt2num_frames $data/utt2num_frames.bak
awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/utt2num_frames.bak > $data/utt2num_frames
utils/filter_scp.pl $data/utt2num_frames $data/utt2spk > $data/utt2spk.new
mv $data/utt2spk.new $data/utt2spk
utils/fix_data_dir.sh $data

echo "remove speakers with not enough utterances"
awk '{print $1, NF-1}' $data/spk2utt > $data/spk2num
awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/spk2num | utils/filter_scp.pl - $data/spk2utt > $data/spk2utt.new
mv $data/spk2utt.new $data/spk2utt
utils/spk2utt_to_utt2spk.pl $data/spk2utt > $data/utt2spk
utils/filter_scp.pl $data/utt2spk $data/utt2num_frames > $data/utt2num_frames.new
mv $data/utt2num_frames.new $data/utt2num_frames
utils/fix_data_dir.sh $data


## Then, do WCMVN and remove nonspeech features
cp $srcdir/utt2spk $data_nosil/utt2spk
cp $srcdir/spk2utt $data_nosil/spk2utt
cp $srcdir/wav.scp $data_nosil/wav.scp
cp $srcdir/vad.scp $data_nosil/vad.scp

[ -f $srcdir/segments ] && cp $srcdir/segments $data_nosil/segments

write_num_frames_opt="--write-num-frames=ark,t:$feat_nosil_dir/log/utt2num_frames.JOB"

$cmd JOB=1:$nj $dir_nosil/log/create_wcmvn_nosil_feats_${name}.JOB.log \
  apply-cmvn-sliding --norm-vars=$norm_vars --center=$center --cmn-window=$cmn_window \
  scp:${sdata_in}/JOB/feats.scp ark:- \| \
  select-voiced-frames ark:- scp,s,cs:${sdata_in}/JOB/vad.scp ark:- \| \
  copy-feats --compress=$compress $write_num_frames_opt ark:- \
  ark,scp:$feat_nosil_dir/wcmvn_nosil_feats_${name}.JOB.ark,$feat_nosil_dir/wcmvn_nosil_feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $feat_nosil_dir/wcmvn_nosil_feats_${name}.$n.scp || exit 1;
done > ${data_nosil}/feats.scp || exit 1

for n in $(seq $nj); do
  cat $feat_nosil_dir/log/utt2num_frames.$n || exit 1;
done > ${data_nosil}/utt2num_frames || exit 1
rm $feat_nosil_dir/log/utt2num_frames.*


# remove features that are too short 
cp -r $data_nosil ${data_nosil}.bak   # we need a backup here
mv $data_nosil/utt2num_frames $data_nosil/utt2num_frames.bak
awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data_nosil/utt2num_frames.bak > $data_nosil/utt2num_frames
utils/filter_scp.pl $data_nosil/utt2num_frames $data_nosil/utt2spk > $data_nosil/utt2spk.new
mv $data_nosil/utt2spk.new $data_nosil/utt2spk
utils/fix_data_dir.sh $data_nosil

awk '{print $1, NF-1}' $data_nosil/spk2utt > $data_nosil/spk2num
awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data_nosil/spk2num | utils/filter_scp.pl - $data_nosil/spk2utt > $data_nosil/spk2utt.new
mv $data_nosil/spk2utt.new $data_nosil/spk2utt
utils/spk2utt_to_utt2spk.pl $data_nosil/spk2utt > $data_nosil/utt2spk
utils/filter_scp.pl $data_nosil/utt2spk $data_nosil/utt2num_frames > $data_nosil/utt2num_frames.new
mv $data_nosil/utt2num_frames.new $data_nosil/utt2num_frames
utils/fix_data_dir.sh $data_nosil

echo "$0: process features successfully"

# Process the alignments 
if [ ! -z $ali_dir ]; then
  echo "$0: process the alignments"
  num_ali_jobs=$(cat $ali_dir/num_jobs) || exit 1;
  
  # with silence, just filter using the utterances 
  for id in $(seq $num_ali_jobs); do gunzip -c $ali_dir/ali.$id.gz; done | \
    copy-int-vector ark:- ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
 
  mv $dir/ali.scp $dir/ali.scp.bak
  utils/filter_scp.pl $data/utt2spk $dir/ali.scp.bak > $dir/ali.scp 
  gzip -c $dir/ali.ark > $dir/ali.gz
  rm -f $dir/ali.ark $dir/ali.scp.bak > /dev/null

  cp $ali_dir/final.mdl $dir/final.mdl
  cp $ali_dir/tree $dir/tree
 
  # with no silence, filter the utterances and select the voice frames
  for id in $(seq $num_ali_jobs); do gunzip -c $ali_dir/ali.$id.gz; done | \
    copy-int-vector ark:- ark,scp:$dir_nosil/ali.ark.bak,$dir_nosil/ali.scp.bak || exit 1;

  abs_dir=$(utils/make_absolute.sh $dir_nosil)
  utils/filter_scp.pl $data_nosil/utt2spk $dir_nosil/ali.scp.bak > $dir_nosil/ali.scp.new
  select-voiced-ali scp:$abs_dir/ali.scp.new scp,s,cs:$data_nosil/vad.scp ark,scp:$abs_dir/ali.ark,$abs_dir/ali.scp
  gzip -c $dir_nosil/ali.ark > $dir_nosil/ali.gz
  rm -f $dir_nosil/ali.ark.bak $dir_nosil/ali.scp.bak $dir_nosil/ali.ark $dir_nosil/ali.scp.new > /dev/null

  cp $ali_dir/final.mdl $dir_nosil/final.mdl
  cp $ali_dir/tree $dir_nosil/tree
 
  # Some utterances are removed due to decoding fail. So the num of utterances in ali.scp may be smaller
  # We fix the data according to the alignments 
  mv $data/utt2spk $data/utt2spk.old
  utils/filter_scp.pl $dir/ali.scp $data/utt2spk.old > $data/utt2spk
  utils/fix_data_dir.sh $data

  mv $data_nosil/utt2spk $data_nosil/utt2spk.old
  utils/filter_scp.pl $dir_nosil/ali.scp $data_nosil/utt2spk.old > $data_nosil/utt2spk
  utils/fix_data_dir.sh $data_nosil

fi

