import os

cur_path = os.path.abspath(__file__)[:-len(os.path.basename(__file__))]
wd = os.path.abspath(cur_path + '../')


def indices_to_one_hot_encodings(index, vector_length):
    return [[1, 0] if i == index else [0, 1] for i in xrange(vector_length)]
