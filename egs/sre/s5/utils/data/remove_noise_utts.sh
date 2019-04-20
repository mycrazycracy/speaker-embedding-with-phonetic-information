#!/bin/bash

if [ $# != 2 ]; then
  echo "Usage: remove_noise_utts.sh <src-data-dir> <dest-data-dir>"
  exit 1;
fi

srcdir=$1
destdir=$2
mkdir -p $destdir

[ ! -f $srcdir/text ] && echo "$0: Invalid input directory $srcdir" && exit 1;

! mkdir -p $destdir && echo "$0: could not create directory $destdir" && exit 1;

cp $srcdir/* $destdir

python -c "
import sys
with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        if ' [' in line:
            continue
        print(line.strip())
" $srcdir/text >$destdir/text

echo "Reduced number of utterances from `cat $srcdir/text | wc -l` to `cat $destdir/text | wc -l`"

echo "Using fix_data_dir.sh to reconcile the other files."
utils/fix_data_dir.sh $destdir
rm -r $destdir/.backup

exit 0
