import sys
import random

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('usage: %s num_heldout_spk num_heldout_utts_per_spk input_spk2utt' % sys.argv[0])
        quit()

    num_spks = int(sys.argv[1])
    num_utts_per_spk = int(sys.argv[2])

    satisfy_spks = []
    not_satisfy_spks = []
    with open(sys.argv[3], 'r') as f:
        for line in f.readlines():
            spk, utts = line.strip().split(' ', 1)
            utts = utts.split(' ')
            if len(utts) >= num_utts_per_spk + 2:
                satisfy_spks.append([spk, utts])
            else:
                not_satisfy_spks.append([spk, utts])

    if len(satisfy_spks) < num_spks:
        satisfy_spks += random.sample(not_satisfy_spks, num_spks - len(satisfy_spks))

    sampled_spks = random.sample(satisfy_spks, num_spks)
    for spk in sampled_spks:
        sys.stdout.write('%s' % spk[0])

        # We should ensure at lease one utterance of each speaker is left in the training set.    
        if len(spk[1]) > num_utts_per_spk:
            spk[1] = random.sample(spk[1], num_utts_per_spk)
        else:
            spk[1] = random.sample(spk[1], len(spk[1]) - 1)

        for utt in spk[1]:
            sys.stdout.write(' %s' % utt)
        sys.stdout.write('\n')

