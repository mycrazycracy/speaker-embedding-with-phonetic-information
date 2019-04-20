#!/bin/bash


if [ $# != 4 ]; then
  echo "Usage: remove_short_utts.sh min-phones <lexicon> <src-data-dir> <dest-data-dir>"
  echo "e.g.: remove_short_utts.sh 10 lexicon.txt data/train data/train_noshort"
  exit 1;
fi

minphones=$1
lexicon=$2
srcdir=$3
destdir=$4
mkdir -p $destdir

[ ! -f $srcdir/text ] && echo "$0: Invalid input directory $srcdir" && exit 1;

! mkdir -p $destdir && echo "$0: could not create directory $destdir" && exit 1;

! [ "$minphones" -gt 1 ] && echo "$0: invalid min-phones '$minphones'" && exit 1;

cp $srcdir/* $destdir

# convert the text to phones
python -c "
import sys
minphones = int(sys.argv[1])
lexicon = {}
with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split(' ')
        lexicon[tmp[0]] = len(tmp) - 1

with open(sys.argv[3], 'r') as f:
    for line in f.readlines():
        total_len = 0 
        for w in line.strip().split(' ')[1:]:
            total_len += lexicon[w]
        if total_len >= minphones:
            print(line.strip())
" $minphones $lexicon $srcdir/text > $destdir/text

echo "Reduced number of utterances from `cat $srcdir/text | wc -l` to `cat $destdir/text | wc -l`"

echo "Using fix_data_dir.sh to reconcile the other files."
utils/fix_data_dir.sh $destdir
rm -r $destdir/.backup

exit 0
