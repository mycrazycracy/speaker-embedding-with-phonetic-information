import sys

# This script convert the phone-level alignment (in text format) 
# to phone-class-level alignment (in integer)

def convert_phones(phone2class, phone_ali_filename, phone_set_ali_filename):
    phone_ali = open(phone_ali_filename, 'r')
    phone_set_ali = open(phone_set_ali_filename, 'w')

    for line in phone_ali.readlines():
        tmp = line.strip().split(' ')
        index = []
        for phone in tmp[1:]:
            index.append(str(phone2class[phone]))
        ali = ' '.join(index)
        phone_set_ali.write('%s %s\n' % (tmp[0], ali))
    
    phone_ali.close()
    phone_set_ali.close()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: %s phone_set phone_ali phone_set_ali')
        quit()

    phone2class = {}
    index = 0
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            for phone in line.strip().split(' '):
                phone2class[phone] = index
            index += 1

    convert_phones(phone2class, sys.argv[2], sys.argv[3])


