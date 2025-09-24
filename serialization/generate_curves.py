# ------------------------------------------------------------------
"""
Script to generate serialization based on space-filling curves

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import numpy as np
import os
import argparse

from gilbert import gilbert_xy2d as gilbert_encode

from sweep_h import encode as sweep_h_encode
from sweep_h import trans_encode as sweep_h_trans_encode

from sweep_v import encode as sweep_v_encode
from sweep_v import trans_encode as sweep_v_trans_encode

from zigzag_h import encode as zigzag_h_encode
from zigzag_h import trans_encode as zigzag_h_trans_encode

from zigzag_v import encode as zigzag_v_encode
from zigzag_v import trans_encode as zigzag_v_trans_encode

from itertools import product

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--height', default=3000, type=int, help='height of the domain [default: 45]')
    parser.add_argument('--width', default=7200, type=int, help='width of the domain [default: 90]')
    parser.add_argument('--method', default='sweep_h', type=str,
                        help='curve type, '
                             '[gilbert, gilbert_trans, zigzag_h, zigzag_h_trans, zigzag_v, zigzag_v_trans, '
                             'sweep_h, sweep_h_trans, sweep_v, sweep_v_trans] '
                             '[default: gilbert]')
    parser.add_argument('--output_folder',
                        default='./curves',
                        type=str,
                        help='directory to save the serialization [default: ./curves]')

    args = parser.parse_args()
    return args

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def encode(args):

    method = args.method
    height, width = args.height, args.width

    assert method in {"gilbert", "gilbert_trans",
                      "zigzag_h", "zigzag_h_trans", "zigzag_v", "zigzag_v_trans",
                      "sweep_h", "sweep_h_trans", "sweep_v", "sweep_v_trans"}

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    indices = []

    if method == "gilbert":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = gilbert_encode(x, y, width, height)
            indices.append(idx)
    elif method == "gilbert_trans":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = gilbert_encode(x, -y+height-1, width, height)
            indices.append(idx)

    elif method == "sweep_h":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = sweep_h_encode(x, y, width)
            indices.append(idx)
    elif method == "sweep_h_trans":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = sweep_h_trans_encode(x, y, width)
            indices.append(idx)

    elif method == "sweep_v":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = sweep_v_encode(x, y, height)
            indices.append(idx)
    elif method == "sweep_v_trans":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = sweep_v_trans_encode(x, y, height)
            indices.append(idx)

    elif method == "zigzag_h":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = zigzag_h_encode(x, y, width)
            indices.append(idx)
    elif method == "zigzag_h_trans":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = zigzag_h_trans_encode(x, y, width)
            indices.append(idx)

    elif method == "zigzag_v":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = zigzag_v_encode(x, y, height)
            indices.append(idx)
    elif method == "zigzag_v_trans":
        for x, y in product(np.arange(width), np.arange(height)):
            idx = zigzag_v_trans_encode(x, y, height)
            indices.append(idx)

    else:
        raise NotImplementedError(f"Method {method} not implemented")

    np.save(os.path.join(output_folder, method + '.npy'),
            np.array(indices).reshape(width, height).T.astype(np.uint32))

    """
    # test the curve
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    indices_loaded = np.load(os.path.join(output_folder, method + '.npy'))
    indices_loaded = indices_loaded.flatten()
    indices_loaded = indices_loaded.argsort()

    xx = xx.flatten()[indices_loaded]
    yy = yy.flatten()[indices_loaded]

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('TkAgg')
    # plt.figure(figsize=(12, 6))
    plt.plot(xx, yy, '.-')
    # plt.plot(xs, yst, '.-', color='red', alpha=0.5)
    plt.show()

    # compute inverse indexing
    full_series = np.arange(height*width)
    dictionary = dict(zip(indices_loaded, full_series))
    indices_loaded_inv = list(map(dictionary.get, full_series))
    """


if __name__ == "__main__":

    args = parse_args()
    encode(args)

