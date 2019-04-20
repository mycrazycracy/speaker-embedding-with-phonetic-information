#!/bin/bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  feats.scp
#  wav.scp
#  vad.scp
#  spk2utt
#  utt2spk
#  text
#
# It combines different segments to utterances, that share the same prefix.
# It is useful when training and extracting i-vectors in a dataset with lots of segments.
#

# begin configuration section
prefix=
compress=true
# end configuration section

. utils/parse_options.sh


if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <num-utt-fields> <srcdir> <destdir>"
  echo "e.g.:"
  echo " $0 3 data/segments data/utterances"
  echo "    fe_03_10035-07679-B-000313-000697 -> fe_03_10035-07679-B"
  echo "Options:"
  echo " --prefix: the prefix to be added ton the new feature archives."
  exit 1;
fi


export LC_ALL=C

num_field=$1
srcdir=$2
destdir=$3

if [ ! -f $srcdir/utt2spk ]; then
  echo "copy_data_dir.sh: no such file $srcdir/utt2spk"
  exit 1;
fi

if [ "$destdir" == "$srcdir" ]; then
  echo "$0: this script requires <srcdir> and <destdir> to be different."
  exit 1
fi

set -e;

mkdir -p $destdir

python -c "
import sys
num_field = int(sys.argv[1])
seg2spk = sys.argv[2]
utt2seg = {}
utt2spk = {}
with open(seg2spk, 'r') as f:
    for line in f.readlines():
        [seg, spk] = line.strip().split(' ')
        utt = '-'.join(seg.split('-')[:num_field])
        if utt not in utt2spk:
            utt2spk[utt] = spk
        else:
            assert utt2spk[utt] == spk
        if utt not in utt2seg:
            utt2seg[utt] = []
        utt2seg[utt].append(seg)
with open(sys.argv[3], 'w') as f:
    for utt in utt2seg:
        f.write('%s ' % utt)
        utt2seg[utt].sort()
        for seg in utt2seg[utt]:
            f.write('%s ' % seg)
        f.write('\n')
with open(sys.argv[4], 'w') as f:
    for utt in utt2spk:
        f.write('%s %s\n' % (utt, utt2spk[utt]))
" $num_field $srcdir/utt2spk $destdir/utt2seg $destdir/utt2spk 

utils/utt2spk_to_spk2utt.pl <$destdir/utt2spk > $destdir/spk2utt

python -c "
import sys
num_field = int(sys.argv[1])
wav2utt = {}
with open(sys.argv[3], 'r') as f:
    for line in f.readlines():
        [seg, wav, _, _] = line.strip().split(' ')
        utt = '-'.join(seg.split('-')[:num_field])
        if wav not in wav2utt:
            wav2utt[wav] = utt
        else:
            assert wav2utt[wav] == utt
with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        [wav, cmd] = line.strip().split(' ', 1)
        print('%s %s' % (wav2utt[wav], cmd))
" $num_field $srcdir/wav.scp $srcdir/segments > $destdir/wav.scp

utils/fix_data_dir.sh $destdir

function check_sorted {
  file=$1
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
}
check_sorted $destdir/utt2seg

name=`basename $destdir`
combine-feats --compress=$compress \
  ark:$destdir/utt2seg scp,s,cs:$srcdir/feats.scp ark,scp:$mfccdir/${prefix}${name}.ark,$destdir/feats.scp
if [ -f $srcdir/vad.scp ]; then 
  combine-vectors ark:$destdir/utt2seg scp,s,cs:$srcdir/vad.scp ark,scp:$vaddir/vad_${name}.ark,$destdir/vad.scp
fi



