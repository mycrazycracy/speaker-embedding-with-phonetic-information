
from __future__ import print_function
import sys
import os

if len(sys.argv) != 4:
    print("{0}: sre04_dir wav_dir output_dir".format(sys.argv[0]))
    quit()

input_dir = sys.argv[1]
wav_dir = sys.argv[2]
output_dir = sys.argv[3]


fp_wav = open(input_dir+"/wav.scp", "r")
fp_utt2spk = open(input_dir+"/utt2spk", "r")
fp_spk2gender = open(input_dir+"/spk2gender", "r")

fp_wav_new = open(output_dir+"/wav.scp", "w")
fp_utt2spk_new = open(output_dir+"/utt2spk", "w")
fp_spk2gender_new = open(output_dir+"/spk2gender", "w")

utt2spk = {}
spk2gender = {}
spks = set()
utts = set()

for line in fp_utt2spk.readlines():
    [utt, spk] = line.strip().split(" ")
    [sid, suffix] = utt.split('-', 1)
    utt = sid + "_sre-" + suffix.replace("-", "_")
    utt2spk[utt] = spk + "_sre"

for line in fp_spk2gender.readlines():
    [spk, gender] = line.strip().split(" ")
    spk = spk + "_sre"
    spk2gender[spk] = gender

for line in fp_wav.readlines():
    tmp = line.rstrip().split(" ")
    utt = tmp[0]
    [sid, suffix] = utt.split('-', 1)
    utt = sid + "_sre-" + suffix.replace("-", "_")
    wav = tmp[-2]
    wav = wav.rsplit(".",1)[0].split("/",6)[-1]
    wav = wav_dir + wav + ".wav"
    if not os.path.isfile(wav):
        print("Cannot find file {0}".format(wav))
        continue
    utts.add(utt)
    spks.add(utt2spk[utt])
    print("{0} sox {1} -t wav -e signed-integer -r 8000 -b 16 -|".format(utt, wav), file=fp_wav_new)

for utt in sorted(list(utts)):
    print("{0} {1}".format(utt, utt2spk[utt]), file=fp_utt2spk_new)

for spk in sorted(list(spks)):
    print("{0} {1}".format(spk, spk2gender[spk]), file=fp_spk2gender_new)

fp_wav.close()
fp_utt2spk.close()
fp_spk2gender.close()

fp_wav_new.close()
fp_utt2spk_new.close()
fp_spk2gender_new.close()
