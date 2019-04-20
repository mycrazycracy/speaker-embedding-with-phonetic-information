import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("%s: trials score" % sys.argv[0])
        quit()
    trials = sys.argv[1]
    score = sys.argv[2]

    fp_trials = open(trials, "r")
    fp_score = open(score, "r")

    line = fp_score.readline().strip()
    while line:
        [spk, utt, s] = line.split(" ")
        [spk2, utt2, target] = fp_trials.readline().strip().split(" ")
        while spk != spk2 or utt != utt2:
            print("{0} {1} -1000.0".format(spk2, utt2))
            [spk2, utt2, target] = fp_trials.readline().strip().split(" ")
        print("{0} {1} {2}".format(spk, utt, s))
        line = fp_score.readline().strip()

    fp_trials.close()
    fp_score.close()
