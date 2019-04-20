#!/bin/bash

if [ $# != 3 ]; then
  echo "Usage: remove_oov_utts.sh <lexicon> <src-data-dir> <dest-data-dir>"
  exit 1;
fi

lexicon=$1
srcdir=$2
destdir=$3
mkdir -p $destdir

[ ! -f $srcdir/text ] && echo "$0: Invalid input directory $srcdir" && exit 1;

! mkdir -p $destdir && echo "$0: could not create directory $destdir" && exit 1;

cp $srcdir/* $destdir

python -c "
import sys
lexicon = set()
with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        lexicon.add(line.split(' ')[0])
with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        valid = True
        for w in line.strip().split(' ')[1:]:
            if w not in lexicon:
                valid = False
                break
        if valid:
            print(line.strip())
" $lexicon $srcdir/text > $destdir/text

echo "Reduced number of utterances from `cat $srcdir/text | wc -l` to `cat $destdir/text | wc -l`"

echo "Using fix_data_dir.sh to reconcile the other files."
utils/fix_data_dir.sh $destdir
rm -r $destdir/.backup

exit 0

