# ------------------------------------------------------------------
"""
Zigzag V space-filling curve

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------


def encode(x, y, height):
    return (x*height + height - y - 1 + (x % 2 == 0) * (y + 1 - height + y)).tolist()

def decode(idx, height):
    return [(i // height, height - i % height - 1 if ((i // height) % 2) else i % height) for i in idx]

def trans_encode(x, y, height):
    return (x * height + y + (x % 2 == 0) * (height - y - y - 1)).tolist()

def trans_decode(idx, height):
    return [(i // height, i % height if ((i // height) % 2) else (height - 1 - i % height)) for i in idx]


if __name__ == "__main__":

    width = 45  # 1000
    height = 90   # 600

    import numpy as np

    idx = [i for i in range(width*height)]

    loc = decode(idx, height)
    loc = np.array(loc)
    xs = loc[:, 0]
    ys = loc[:, 1]

    idx = encode(xs, ys, height)

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

