import random

def shuffle_multi(*ls):

    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

def pack_multi(*ls):
    #print(*ls)
    return list(zip(*ls))

def unpack_multi(l):
    return zip(*l)
