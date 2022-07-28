from nltk.corpus import wordnet as wn


def get_nick_name(s):
    n = s.name()
    result = n[: -5]
    return result.replace('_', ' ')

def noffset_to_synset(noffset):
    noffset = noffset.strip()
    return wn.synset_from_pos_and_offset(noffset[0], int(noffset[1:]))

