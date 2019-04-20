#!/usr/bin/env python

# Generate the speaker alignments for utterances. The spk_list contains spk-id|sent-id|utt.
# But the utt may contains file path. So a split is necessary. In addition, the utterance name 
# in utt2num_frames may be segments (rather than the whole utterance), and is appended with
# x-1/-2/... suffix. A split is also necessary if we want to find the corresponding speaker-id
# in spk_list. 

import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 4: 
        print('Usage: %s utt2num_frames spk_list ali_filename' % sys.argv[0])
        quit()

    f_utt_len = sys.argv[1]
    f_spk_list = sys.argv[2]
    f_ali = sys.argv[3]

    if not os.path.isfile(f_utt_len) or not os.path.isfile(f_spk_list):
        print('[ERROR] Expecting utt2num_frames and spk_list existing')
        quit()

    utt_spk = {}

    with open(f_spk_list, 'r') as f:
        for line in f.readlines():
            info = line.strip().split('|')
            spk = info[0]
            # In case utt contains full path
            utt = info[-1].rsplit('/', 1)[1]
            utt_spk[utt] = spk

    mode = 0              # to indicate utterance or segment is in utt2num_frames
    spk_index = {}
    fid_ali = open(f_ali, 'w')
    
    with open(f_utt_len, 'r') as f:
        for line in f.readlines():
            [utt, num] = line.strip().split(' ')
            num = int(num)
            utt_tmp = utt.rsplit('-', 1)[0]
            if mode == 0:
                if utt in utt_spk and utt_tmp in utt_spk:
                    print('[ERROR] Cannot figure utterance or segment is in the utt2num_frames')
                    quit()
                if utt in utt_spk:
                    mode = 1          # utterance is used
                    print('Utterance name detected')
                elif utt_tmp in utt_spk:
                    mode = 2          # segment is used  
                    print('Segment name detected')
                else:
                    print('[ERROR] Cannot figure utterance or segment is in the utt2num_frames')
                    quit()

            if mode == 1:
                utt_tmp = utt
            elif mode == 2:
                pass

            spk = utt_spk[utt_tmp]
            if spk not in spk_index:
                spk_id = len(spk_index.keys())
                spk_index[spk] = spk_id
            else:
                spk_id = spk_index[spk]
        
            fid_ali.write('%s ' % utt)
            fid_ali.write('%s\n' % ((str(spk_id)+' ')*num))

    fid_ali.close()

