There are 2 versions of c-vector:
1. v1 uses multi-task learning to train c-vector
2. v2 introduces phonetic vector to train c-vector

*** The code is based on Kaldi 5.2

Steps:
0. Prepare the data. I provide the data list in data.tar.gz. This is a dataset based on Fisher English which I stated in the paper.
1. Place the source code we use in src/ and make them.
2. Please read through run.sh in the egs and run it line-by-line.
3. When executing each stage in run.sh, copy the corresponding files to the correct directories. 

*** Check that your kaldi has been backed up since the code over-write some files.

List of files that we modified:


I didn't make these changes as a patch or PR, so you can combine them with your own version manually.
If the kaldi is just downloaded from github, then simply copying the files to the toolkit should work.

mailto: liu-yi15 at mails.tsinghua.edu.cn if you find any problems.

This code follows Apache 2.0 as Kaldi states.

Please cite the paper if you use the code.
Liu, Y., He, L., Liu, J. and Johnson, M.T.. Speaker Embedding Extraction with Phonetic Information. Interspeech 2018.
