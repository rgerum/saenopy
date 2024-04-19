import numpy as np
import copy


def update_keys(d1, d2):
    # recounts keys of d2 by starting with last key of d1. keys must all be integers
    d3 = copy.deepcopy(d2)
    max_key = np.max(list(d1.keys()))
    for key, value in d2.items():
        max_key += 1
        d3[max_key] = value
    return d3


def join_dictionary(d1, d2, do_update_keys=False):
    if do_update_keys:
        d3 = update_keys(d1, d2)
    else:
        d3 = d2
    return {**d1, **d3}
    # note:z = {**x, **y} and "update" are nice tricks here


def make_random_discrete_color_range(size):
    colors = []
    for i in range(size):
        colors.append("#%06X" % np.random.randint(0, 0xFFFFFF))
    return colors
