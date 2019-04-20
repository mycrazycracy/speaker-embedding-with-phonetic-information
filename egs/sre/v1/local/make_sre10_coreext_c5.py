
from __future__ import print_function
import sys
import os
import commands

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("{0}: sre10_list_dir coreext_c5_output_dir".format(sys.argv[0]))
        quit()

    sre10_list_dir = sys.argv[1]
    output_dir = sys.argv[2]


    # female part
    enroll_wav_scp = output_dir + "/enroll/female/wav.scp"
    enroll_utt2spk = output_dir + "/enroll/female/utt2spk"
    enroll_spk2utt = output_dir + "/enroll/female/spk2utt"

    test_wav_scp = output_dir + "/test/female/wav.scp"
    test_utt2spk = output_dir + "/test/female/utt2spk"
    test_spk2utt = output_dir + "/test/female/spk2utt"
    test_trials = output_dir + "/test/female/trials"

    if not os.path.isdir(output_dir + "/enroll/female"):
        os.makedirs(output_dir + "/enroll/female")
    if not os.path.isdir(output_dir + "/test/female"):
        os.makedirs(output_dir + "/test/female")

    trial_wav_map = {}
    org_trials = sre10_list_dir + "/trials/coreext-coreext.ndx"
    with open(org_trials, "r") as f:
        for line in f.readlines():
            [spk, gender, wav] = line.strip().split(" ")
            if gender == "f":
                chan = wav.rsplit(":", 1)[1].lower()
                wav_name = wav.rsplit("/",1)[1].split(".")[0] + "_" + chan
                wav_path = "/mnt/lv10/database_r1/sre/nist/SRE10/eval/data/" + wav.split(".")[0] + "_" + chan + ".wav"
                if wav_name in trial_wav_map:
                    if trial_wav_map[wav_name] != wav_path:
                        print("error")
                        quit()
                trial_wav_map[wav_name] = wav_path

    f_test_wav_scp = open(test_wav_scp, "w")
    f_test_utt2spk = open(test_utt2spk, "w")
    f_test_trials = open(test_trials, "w")
    
    spk_list = set()
    trial_wav_list = set()

    trial_key = sre10_list_dir + "/keys/coreext-coreext.trialkey.csv"
    with open(trial_key, "r") as f:
        f.readline()
        for line in f.readlines():
            if "Y,N,N,N,N,Y,N,N,N,N" in line:
                [spk, wav, chan, type, dummy] = line.strip().split(",", 4)
                wav_name = wav + "_" + chan
                if wav_name in trial_wav_map:
                    print("{0}_sre10 {1}_{2}_sre10 {3}".format(spk, wav, chan, type), file=f_test_trials)
                    trial_wav_list.add(wav_name)
                    spk_list.add(spk)

    for wav in sorted(list(trial_wav_list)):
        if not os.path.isfile(trial_wav_map[wav]):
            print("Cannot find file {0}".format((trial_wav_map[wav])))
        print("{0}_sre10 {0}_sre10".format(wav, wav), file=f_test_utt2spk)
        print("{0}_sre10 sox {1} -t wav -e signed-integer -r 8000 -b 16 -|".format(wav, trial_wav_map[wav]), file=f_test_wav_scp)

    f_test_wav_scp.close()
    f_test_utt2spk.close()
    f_test_trials.close()

    f_enroll_wav_scp = open(enroll_wav_scp, "w")
    f_enroll_utt2spk = open(enroll_utt2spk, "w")
    
    coreext_train = sre10_list_dir + "/train/coreext.trn"
    with open(coreext_train, "r") as f:
        for line in f.readlines():
            [spk, gender, wav] = line.strip().split(" ")
            if gender == "f":
                chan = wav.rsplit(":", 1)[1].lower()
                wav_name = wav.rsplit("/",1)[1].split(".")[0] + "_" + chan
                wav_path = "/mnt/lv10/database_r1/sre/nist/SRE10/eval/data/" + wav.split(".")[0] + "_" + chan + ".wav"
                if spk in spk_list:
                    if not os.path.isfile(wav_path):
                        print("Cannot find file {0}".format((wav_path)))
                    print("{1}_sre10-{0}_sre10 {1}_sre10".format(wav_name, spk), file=f_enroll_utt2spk)
                    print("{0}_sre10-{1}_sre10 sox {2} -t wav -e signed-integer -r 8000 -b 16 -|".format(spk, wav_name, wav_path), file=f_enroll_wav_scp)
            
    f_enroll_wav_scp.close()
    f_enroll_utt2spk.close()

    print(commands.getoutput("utils/utt2spk_to_spk2utt.pl {0} > {1}".format(output_dir+"enroll/female/utt2spk", output_dir+"enroll/female/spk2utt")))
    print(commands.getoutput("utils/utt2spk_to_spk2utt.pl {0} > {1}".format(output_dir+"test/female/utt2spk", output_dir+"test/female/spk2utt")))
    print(commands.getoutput("utils/fix_data_dir.sh {0}".format(output_dir+"enroll/female")))
    print(commands.getoutput("utils/fix_data_dir.sh {0}".format(output_dir+"test/female")))


    # male part
    enroll_wav_scp = output_dir + "/enroll/male/wav.scp"
    enroll_utt2spk = output_dir + "/enroll/male/utt2spk"
    enroll_spk2utt = output_dir + "/enroll/male/spk2utt"

    test_wav_scp = output_dir + "/test/male/wav.scp"
    test_utt2spk = output_dir + "/test/male/utt2spk"
    test_spk2utt = output_dir + "/test/male/spk2utt"
    test_trials = output_dir + "/test/male/trials"

    if not os.path.isdir(output_dir + "/enroll/male"):
        os.makedirs(output_dir + "/enroll/male")
    if not os.path.isdir(output_dir + "/test/male"):
        os.makedirs(output_dir + "/test/male")

    trial_wav_map = {}
    org_trials = sre10_list_dir + "/trials/coreext-coreext.ndx"
    with open(org_trials, "r") as f:
        for line in f.readlines():
            [spk, gender, wav] = line.strip().split(" ")
            if gender == "m":
                chan = wav.rsplit(":", 1)[1].lower()
                wav_name = wav.rsplit("/",1)[1].split(".")[0] + "_" + chan
                wav_path = "/mnt/lv10/database_r1/sre/nist/SRE10/eval/data/" + wav.split(".")[0] + "_" + chan + ".wav"
                if wav_name in trial_wav_map:
                    if trial_wav_map[wav_name] != wav_path:
                        print("error")
                        quit()
                trial_wav_map[wav_name] = wav_path

    f_test_wav_scp = open(test_wav_scp, "w")
    f_test_utt2spk = open(test_utt2spk, "w")
    f_test_trials = open(test_trials, "w")
    
    spk_list = set()
    trial_wav_list = set()

    trial_key = sre10_list_dir + "/keys/coreext-coreext.trialkey.csv"
    with open(trial_key, "r") as f:
        f.readline()
        for line in f.readlines():
            if "Y,N,N,N,N,Y,N,N,N,N" in line:
                [spk, wav, chan, type, dummy] = line.strip().split(",", 4)
                wav_name = wav + "_" + chan
                if wav_name in trial_wav_map:
                    print("{0}_sre10 {1}_{2}_sre10 {3}".format(spk, wav, chan, type), file=f_test_trials)
                    trial_wav_list.add(wav_name)
                    spk_list.add(spk)

    for wav in sorted(list(trial_wav_list)):
        if not os.path.isfile(trial_wav_map[wav]):
            print("Cannot find file {0}".format((trial_wav_map[wav])))
        print("{0}_sre10 {0}_sre10".format(wav, wav), file=f_test_utt2spk)
        print("{0}_sre10 sox {1} -t wav -e signed-integer -r 8000 -b 16 -|".format(wav, trial_wav_map[wav]), file=f_test_wav_scp)

    f_test_wav_scp.close()
    f_test_utt2spk.close()
    f_test_trials.close()

    f_enroll_wav_scp = open(enroll_wav_scp, "w")
    f_enroll_utt2spk = open(enroll_utt2spk, "w")
    
    coreext_train = sre10_list_dir + "/train/coreext.trn"
    with open(coreext_train, "r") as f:
        for line in f.readlines():
            [spk, gender, wav] = line.strip().split(" ")
            if gender == "m":
                chan = wav.rsplit(":", 1)[1].lower()
                wav_name = wav.rsplit("/",1)[1].split(".")[0] + "_" + chan
                wav_path = "/mnt/lv10/database_r1/sre/nist/SRE10/eval/data/" + wav.split(".")[0] + "_" + chan + ".wav"
                if spk in spk_list:
                    if not os.path.isfile(wav_path):
                        print("Cannot find file {0}".format((wav_path)))
                    print("{1}_sre10-{0}_sre10 {1}_sre10".format(wav_name, spk), file=f_enroll_utt2spk)
                    print("{0}_sre10-{1}_sre10 sox {2} -t wav -e signed-integer -r 8000 -b 16 -|".format(spk, wav_name, wav_path), file=f_enroll_wav_scp)
            
    f_enroll_wav_scp.close()
    f_enroll_utt2spk.close()

    print(commands.getoutput("utils/utt2spk_to_spk2utt.pl {0} > {1}".format(output_dir+"enroll/male/utt2spk", output_dir+"enroll/male/spk2utt")))
    print(commands.getoutput("utils/utt2spk_to_spk2utt.pl {0} > {1}".format(output_dir+"test/male/utt2spk", output_dir+"test/male/spk2utt")))

    print(commands.getoutput("utils/fix_data_dir.sh {0}".format(output_dir+"enroll/male")))
    print(commands.getoutput("utils/fix_data_dir.sh {0}".format(output_dir+"test/male")))


    # the pool part
    enroll_wav_scp = output_dir + "/enroll/pool/wav.scp"
    enroll_utt2spk = output_dir + "/enroll/pool/utt2spk"
    enroll_spk2utt = output_dir + "/enroll/pool/spk2utt"

    test_wav_scp = output_dir + "/test/pool/wav.scp"
    test_utt2spk = output_dir + "/test/pool/utt2spk"
    test_spk2utt = output_dir + "/test/pool/spk2utt"
    test_trials = output_dir + "/test/pool/trials"

    if not os.path.isdir(output_dir + "/enroll/pool"):
        os.makedirs(output_dir + "/enroll/pool")
    if not os.path.isdir(output_dir + "/test/pool"):
        os.makedirs(output_dir + "/test/pool")

    trial_wav_map = {}
    org_trials = sre10_list_dir + "/trials/coreext-coreext.ndx"
    with open(org_trials, "r") as f:
        for line in f.readlines():
            [spk, gender, wav] = line.strip().split(" ")
            chan = wav.rsplit(":", 1)[1].lower()
            wav_name = wav.rsplit("/",1)[1].split(".")[0] + "_" + chan
            wav_path = "/mnt/lv10/database_r1/sre/nist/SRE10/eval/data/" + wav.split(".")[0] + "_" + chan + ".wav"
            if wav_name in trial_wav_map:
                if trial_wav_map[wav_name] != wav_path:
                    print("error")
                    quit()
            trial_wav_map[wav_name] = wav_path

    f_test_wav_scp = open(test_wav_scp, "w")
    f_test_utt2spk = open(test_utt2spk, "w")
    f_test_trials = open(test_trials, "w")
    
    spk_list = set()
    trial_wav_list = set()

    trial_key = sre10_list_dir + "/keys/coreext-coreext.trialkey.csv"
    with open(trial_key, "r") as f:
        f.readline()
        for line in f.readlines():
            if "Y,N,N,N,N,Y,N,N,N,N" in line:
                [spk, wav, chan, type, dummy] = line.strip().split(",", 4)
                wav_name = wav + "_" + chan
                if wav_name in trial_wav_map:
                    print("{0}_sre10 {1}_{2}_sre10 {3}".format(spk, wav, chan, type), file=f_test_trials)
                    trial_wav_list.add(wav_name)
                    spk_list.add(spk)

    for wav in sorted(list(trial_wav_list)):
        if not os.path.isfile(trial_wav_map[wav]):
            print("Cannot find file {0}".format((trial_wav_map[wav])))
        print("{0}_sre10 {0}_sre10".format(wav, wav), file=f_test_utt2spk)
        print("{0}_sre10 sox {1} -t wav -e signed-integer -r 8000 -b 16 -|".format(wav, trial_wav_map[wav]), file=f_test_wav_scp)

    f_test_wav_scp.close()
    f_test_utt2spk.close()
    f_test_trials.close()

    f_enroll_wav_scp = open(enroll_wav_scp, "w")
    f_enroll_utt2spk = open(enroll_utt2spk, "w")
    
    coreext_train = sre10_list_dir + "/train/coreext.trn"
    with open(coreext_train, "r") as f:
        for line in f.readlines():
            [spk, gender, wav] = line.strip().split(" ")
            chan = wav.rsplit(":", 1)[1].lower()
            wav_name = wav.rsplit("/",1)[1].split(".")[0] + "_" + chan
            wav_path = "/mnt/lv10/database_r1/sre/nist/SRE10/eval/data/" + wav.split(".")[0] + "_" + chan + ".wav"
            if spk in spk_list:
                if not os.path.isfile(wav_path):
                    print("Cannot find file {0}".format((wav_path)))
                print("{1}_sre10-{0}_sre10 {1}_sre10".format(wav_name, spk), file=f_enroll_utt2spk)
                print("{0}_sre10-{1}_sre10 sox {2} -t wav -e signed-integer -r 8000 -b 16 -|".format(spk, wav_name, wav_path), file=f_enroll_wav_scp)
            
    f_enroll_wav_scp.close()
    f_enroll_utt2spk.close()

    print(commands.getoutput("utils/utt2spk_to_spk2utt.pl {0} > {1}".format(output_dir+"enroll/pool/utt2spk", output_dir+"enroll/pool/spk2utt")))
    print(commands.getoutput("utils/utt2spk_to_spk2utt.pl {0} > {1}".format(output_dir+"test/pool/utt2spk", output_dir+"test/pool/spk2utt")))

    print(commands.getoutput("utils/fix_data_dir.sh {0}".format(output_dir+"enroll/pool")))
    print(commands.getoutput("utils/fix_data_dir.sh {0}".format(output_dir+"test/pool")))

