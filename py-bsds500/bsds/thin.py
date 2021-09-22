import numpy as np


# Thinning morphological operation applied using lookup tables.
# We convert the 3x3 neighbourhood surrounding a pixel to an index
# used to lookup the output in a lookup table.

# Bit masks for each neighbour
#   1   2   4
#   8  16  32
#  64 128 256
NEIGH_MASK_EAST = 32
NEIGH_MASK_NORTH_EAST = 4
NEIGH_MASK_NORTH = 2
NEIGH_MASK_NORTH_WEST = 1
NEIGH_MASK_WEST = 8
NEIGH_MASK_SOUTH_WEST = 64
NEIGH_MASK_SOUTH = 128
NEIGH_MASK_SOUTH_EAST = 256
NEIGH_MASK_CENTRE = 16

# Masks in a list
# MASKS[0] = centre
# MASKS[1..8] = start from east, counter-clockwise
MASKS = [NEIGH_MASK_CENTRE,
         NEIGH_MASK_EAST, NEIGH_MASK_NORTH_EAST, NEIGH_MASK_NORTH, NEIGH_MASK_NORTH_WEST,
         NEIGH_MASK_WEST, NEIGH_MASK_SOUTH_WEST, NEIGH_MASK_SOUTH, NEIGH_MASK_SOUTH_EAST,
         ]

# Constant listing all indices
_LUT_INDS = np.arange(512)


def binary_image_to_lut_indices(x):
    """
    Convert a binary image to an index image that can be used with a lookup table
    to perform morphological operations. Non-zero elements in the image are interpreted
    as 1, zero elements as 0

    :param x: a 2D NumPy array.
    :return: a 2D NumPy array, same shape as x
    """
    if x.ndim != 2:
        raise ValueError('x should have 2 dimensions, not {}'.format(x.ndim))

    # If the dtype of x is not bool, convert
    if x.dtype != np.bool:
        x = x != 0

    # Add
    x = np.pad(x, [(1, 1), (1, 1)], mode='constant')

    # Convert to LUT indices
    lut_indices = x[:-2,   :-2] * NEIGH_MASK_NORTH_WEST + \
                  x[:-2,  1:-1] * NEIGH_MASK_NORTH + \
                  x[:-2,    2:] * NEIGH_MASK_NORTH_EAST + \
                  x[1:-1,  :-2] * NEIGH_MASK_WEST + \
                  x[1:-1, 1:-1] * NEIGH_MASK_CENTRE + \
                  x[1:-1,   2:] * NEIGH_MASK_EAST + \
                  x[2:,    :-2] * NEIGH_MASK_SOUTH_WEST + \
                  x[2:,   1:-1] * NEIGH_MASK_SOUTH + \
                  x[2:,     2:] * NEIGH_MASK_SOUTH_EAST

    return lut_indices.astype(np.int32)


def apply_lut(x, lut):
    """
    Perform a morphological operation on the binary image x using the supplied lookup table
    :param x:
    :param lut:
    :return:
    """
    if lut.ndim != 1:
        raise ValueError('lut should have 1 dimension, not {}'.format(lut.ndim))

    if lut.shape[0] != 512:
        raise ValueError('lut should have 512 entries, not {}'.format(lut.shape[0]))

    lut_indices = binary_image_to_lut_indices(x)

    return lut[lut_indices]


def identity_lut():
    """
    Create identity lookup tablef
    :return:
    """
    lut = np.zeros((512,), dtype=bool)
    inds = np.arange(512)

    lut[(inds & NEIGH_MASK_CENTRE)!=0] = True

    return lut


def _lut_mutate_mask(lut):
    """
    Get a mask that shows which neighbourhood shapes result in changes to the image
    :param lut: lookup table
    :return: mask indicating which lookup indices result in changes
    """
    return lut != identity_lut()



def lut_masks_zero(neigh):
    """
    Create a LUT index mask for which the specified neighbour is 0
    :param neigh: neighbour index; counter-clockwise from 1 staring at the eastern neighbour
    :return: a LUT index mask
    """
    if neigh > 8:
        neigh -= 8
    return (_LUT_INDS & MASKS[neigh]) == 0

def lut_masks_one(neigh):
    """
    Create a LUT index mask for which the specified neighbour is 1
    :param neigh: neighbour index; counter-clockwise from 1 staring at the eastern neighbour
    :return: a LUT index mask
    """
    if neigh > 8:
        neigh -= 8
    return (_LUT_INDS & MASKS[neigh]) != 0

def _thin_cond_g1():
    """
    Thinning morphological operation; condition G1
    :return: a LUT index mask
    """
    b = np.zeros(512, dtype=int)
    for i in range(1, 5):
        b += lut_masks_zero(2*i-1) & (lut_masks_one(2*i) | lut_masks_one(2*i+1))
    return b == 1

def _thin_cond_g2():
    """
    Thinning morphological operation; condition G2
    :return: a LUT index mask
    """
    n1 = np.zeros(512, dtype=int)
    n2 = np.zeros(512, dtype=int)
    for k in range(1, 5):
        n1 += (lut_masks_one(2*k-1) | lut_masks_one(2*k))
        n2 += (lut_masks_one(2*k) | lut_masks_one(2*k+1))
    m = np.minimum(n1, n2)
    return (m >= 2) & (m <= 3)

def _thin_cond_g3():
    """
    Thinning morphological operation; condition G3
    :return: a LUT index mask
    """
    return ((lut_masks_one(2) | lut_masks_one(3) | lut_masks_zero(8)) & lut_masks_one(1)) == 0

def _thin_cond_g3_prime():
    """
    Thinning morphological operation; condition G3'
    :return: a LUT index mask
    """
    return ((lut_masks_one(6) | lut_masks_one(7) | lut_masks_zero(4)) & lut_masks_one(5)) == 0

def _thin_iter_1_lut():
    """
    Thinning morphological operation; lookup table for iteration 1
    :return: lookup table
    """
    lut = identity_lut()
    cond = _thin_cond_g1() & _thin_cond_g2() & _thin_cond_g3()
    lut[cond] = False
    return lut

def _thin_iter_2_lut():
    """
    Thinning morphological operation; lookup table for iteration 2
    :return: lookup table
    """
    lut = identity_lut()
    cond = _thin_cond_g1() & _thin_cond_g2() & _thin_cond_g3_prime()
    lut[cond] = False
    return lut

def binary_thin(x, max_iter=None):
    """
    Binary thinning morphological operation

    :param x: a binary image, or an image that is to be converted to a binary image
    :param max_iter: maximum number of iterations; default is `None` that results in an infinite
    number of iterations (note that `binary_thin` will automatically terminate when no more changes occur)
    :return:
    """
    thin1 = _thin_iter_1_lut()
    thin2 = _thin_iter_2_lut()
    thin1_mut = _lut_mutate_mask(thin1)
    thin2_mut = _lut_mutate_mask(thin2)

    iter_count = 0
    while max_iter is None or iter_count < max_iter:
        # Iter 1
        lut_indices = binary_image_to_lut_indices(x)
        x_mut = thin1_mut[lut_indices]
        if x_mut.sum() == 0:
            break

        x = thin1[lut_indices]

        # Iter 2
        lut_indices = binary_image_to_lut_indices(x)
        x_mut = thin2_mut[lut_indices]
        if x_mut.sum() == 0:
            break

        x = thin2[lut_indices]

        iter_count += 1

    return x