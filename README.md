## This is the repository for Interspeech paper "Speaker Embedding Extraction with Phonetic Information" 

New architectures are included in addition to the original methods in the Interspeech paper.

If you are looking for the original repository containing the Fisher recipe and the data list, simply switch to branch "cvector_v1".

## Introduction

In the original Interspeech paper, we proposed two speaker embeddings, x-vector with multitask learning and phonetic adaptation. 
In our recent work, we found the c-vector (a phonetic information combined vector) generally performs better. Here, we also include the c-vector and the results are reported on SRE10 and SRE16. 

The speaker training data comes from NIST SRE04-08 and Switchboard while the phonetic training data is Fisher and Switchboard-1. Data augmentation is used in our recipe. MUSAN and RIRS are used as the noise datasets. Refer to the scripts for more details. 

Before running the scripts, you should first go through the Kaldi SRE16 v2 recipe since the scripts share the same idea.

We will also update a new simplified c-vector and scripts on VoxCeleb and Librispeech soon. 

## Files

```
|
|- egs - sre - s5: An ASR system that generate the alignments
|            |     run.sh  GMM/HMM used to align the text and the utterances
|            |
|            - v1: The i-vector framework, i.e., GMM/i-vector and DNN/i-vector. 
|            |     run_sre10.sh/run_sre16.sh  GMM/i-vector
|            |     run_sre10_nnet2.sh/run_sre16_nnet2.sh  DNN/i-vector
|            |    
|            |     For i-vector, we do not use data augmentation since it does improve a lot.
|            |     Data augmentation is used for x-vector and c-vectors. 
|            |
|            - v2: x-vector (Kaldi recipe)
|            |     run_sre10.sh/run_sre16.sh  x-vector
|            |
|            - v3: speaker embedding with multitask learning
|            |     run_sre10.sh/run_sre16.sh  c-vector (multitask learning)
|            |
|            - v4: speaker embedding with phonetic adaptation
|            |     run_sre10.sh/run_sre16.sh  c-vector (phonetic adaptation)
|            |     
|            - v5: c-vector (combining multitask learning and phonetic adaptation)
|                  run_sre10.sh/run_sre16.sh  the final c-vector
|
|- tools - det_score: SRE evaluation tools provided by NIST
```

Steps:

1. Place the egs/src/tools on your kaldi directory and make sure you have a backup on your own changes. 

2. Go to src, add the program to the corresponding makefile and then make (2 programs). 

3. Go to egs/sre and choose the system you like.



## Reference

```
@inproceedings{liu2018speaker,
  title={Speaker Embedding Extraction with Phonetic Information},
  author={Liu, Yi and He, Liang and Liu, Jia and Johnson, Michael T},
  booktitle={Proc. Interspeech 2018},
  pages={2247--2251},
  year={2018}
}
```

