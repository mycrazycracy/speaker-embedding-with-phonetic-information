import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: %s vad phone_align combined_output' % sys.argv[0])
        quit()
    fvad = open(sys.argv[1], 'r')
    fphone = open(sys.argv[2], 'r')
    fcomb = open(sys.argv[3], 'w')

#     import pdb
#     pdb.set_trace()
    vad_filename = None
    for phone_line in fphone.readlines():
        filename, align = phone_line.strip().split(' ', 1)
        align = align.split(' ')
        while filename != vad_filename:
            vad_line = fvad.readline()
            vad_filename, vad = vad_line.strip().split('[')
            vad_filename = vad_filename.strip()
            vad = [int(v) for v in vad[:-1].strip().split(' ')]
        assert len(align) == len(vad)
        fcomb.write('%s\n' % filename)
        align_str = ''
        vad_str = ''
        for index, a in enumerate(align):
            align_len = len(align[index])
            align_str += '%s ' % align[index]
            vad_str += ('%'+str(align_len)+'d ') % vad[index]
        fcomb.write(align_str + '\n')
        fcomb.write(vad_str + '\n')

    fphone.close()
    fvad.close()
    fcomb.close()


