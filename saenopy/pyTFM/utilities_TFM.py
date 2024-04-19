import numpy as np


def join_dictionary(d1, d2, update_keys=False):
    if update_keys:
        d3 = update_keys(d1, d2)
    else:
        d3 = d2
    return {**d1, **d3}
    # note:z = {**x, **y} and "update" are nices tricks here


def make_random_discrete_color_range(size):
    colors = []
    for i in range(size):
        colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
    return colors
