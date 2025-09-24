# ------------------------------------------------------------------
"""
Sweep H space-filling curve

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------


def encode(x, y, width):
    return (y*width+x).tolist()

def decode(idx, width):
    return [(i % width, i // width) for i in idx]

def trans_encode(x, y, width):
    return (y * width + width - x - 1).tolist()

def trans_decode(idx, width):
    return [(width - i % width - 1, i // width) for i in idx]


if __name__ == "__main__":

    width = 45  # 1000  45
    height = 90   # 600  90

    import numpy as np
    idx = [i for i in range(width*height)]
    loc = trans_decode(idx, width)
    loc = np.array(loc)
    xs = loc[:, 0]
    ys = loc[:, 1]

    from itertools import product
    indices = []
    x_all, y_all = [], []
    for x, y in product(np.arange(width), np.arange(height)):
        idx = encode(x, y, width)
        indices.append(idx)

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('TkAgg')
    #plt.figure(figsize=(12, 12))
    plt.plot(xs, ys, '.-')
    #plt.plot(xs, yst, '.-', color='red', alpha=0.5)
    plt.show()

