"""
Utilities
"""

import math
import numpy as np
import torch


def np_invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def randomize_shape_order(nb_shapes, shape_list, is_bordering, is_containing,
                          relabel_shapeness=True):
    # Check which shapes are real and which are placeholders. Placeholders
    # are denoted with -1 in all columns, which is most clearly an invalid
    # value in the shapeness column.
    shape_is_present = shape_list[:, :, 2] > -0.5  # mask

    # We need to know how many vignettes and shapes there are
    n_vignette, n_shapes_max, n_symbolic_outputs = shape_list.size()
    n_shapes_actual = shape_is_present.sum(1)

    # Shuffle order of shapes in the vignette
    uu = []  # List of each vignette of reordered shape_list contents
    vv = []  # List of each vignette of reordered is_bordering
    ww = []  # List of each vignette of reordered is_containing
    # Loop over each vignette
    for (n, u, v, w) in zip(n_shapes_actual, shape_list, is_bordering, is_containing):
        # Create a random permutation for this vignette
        p = torch.cat([torch.randperm(n),
                       torch.arange(n, n_shapes_max).type(torch.LongTensor)])
        # Reorder each of the matrices
        uu.append(torch.index_select(u, 0, p))
        vv.append(torch.index_select(torch.index_select(v, 0, p), 1, p))
        ww.append(torch.index_select(torch.index_select(w, 0, p), 1, p))

    # Now relabel shapeness, so it is labelled in the order it appears
    if relabel_shapeness:
        for i in range(n_vignette):
            shapeness = uu[i][:n, 2]
            # Note that the unsorted list is in the *reverse* order to when
            # unique values first occur.
            #unique_shapeness, idx = torch.unique(
            #    shapeness, sorted=False, return_inverse=True)
            #remapped_shapeness = len(unique_shapeness) - idx - 1

            unique_val, unique_indices, unique_inverse = np.unique(
                shapeness.numpy(), return_index=True, return_inverse=True)
            shapeness_map = np_invert_permutation(np.argsort(unique_indices))
            remapped_shapeness = shapeness_map[unique_inverse]

            uu[i][:n, 2] = torch.Tensor(remapped_shapeness)

    shape_list = torch.stack(uu)
    is_bordering = torch.stack(vv)
    is_containing = torch.stack(ww)

    return shape_list, is_bordering, is_containing


def randomize_shape_rotations(shape_list, nb_shapes):
    # For each set of shapes with the same shapeness, randomly rotate
    # them all by the same amount

    # Check which shapes are real and which are placeholders. Placeholders
    # are denoted with -1 in all columns, which is most clearly an invalid
    # value in the shapeness column.
    shape_is_present = shape_list[:, :, 2] > -0.5  # mask

    # We need to know how many vignettes and shapes there are, so we can
    # generate the correct number of random rotation offsets
    n_vignette, n_shapes_max, n_symbolic_outputs = shape_list.size()
    # We generate a random rotation for every possible shapeness
    rot_offsets = 2 * math.pi * torch.rand((n_vignette, n_shapes_max))

    # Check which shapeness each shape has. We will use this to index into
    # the randomly generated rotation states, so all identical shapes
    # are changed in the same manner.
    idx = shape_list[:, :, 2].type(torch.LongTensor)

    # Select the offsets based on shapeness index
    offsets = torch.gather(rot_offsets,
                           1,
                           idx * shape_is_present.type(torch.LongTensor))
    shape_list[:, :, 3] += offsets

    # Wrap to the interval [0, 2PI)
    shape_list[:, :, 3] %= 2 * math.pi

    return shape_list


def randomize_shape_reflections(shape_list, nb_shapes):
    # For each set of shapes with the same shapeness, randomly flip their
    # mirroredness

    # Check which shapes are real and which are placeholders. Placeholders
    # are denoted with -1 in all columns, which is most clearly an invalid
    # value in the shapeness column.
    shape_is_present = shape_list[:, :, 2] > -0.5  # mask

    # We need to know how many vignettes and shapes there are, so we can
    # generate the correct number of random rotation offsets
    n_vignette, n_shapes_max, n_symbolic_outputs = shape_list.size()
    # We generate a random reflection for every possible shapeness
    ref_offsets = torch.rand((n_vignette, n_shapes_max))

    # Check which shapeness each shape has. We will use this to index into
    # the randomly generated reflection states, so all identical shapes
    # are changed in the same manner.
    idx = shape_list[:, :, 2].type(torch.LongTensor)

    # Select the offsets based on shapeness index
    offsets = torch.gather(ref_offsets,
                           1,
                           idx * shape_is_present.type(torch.LongTensor))
    shape_list[:, :, 5] += offsets

    # Wrap to the interval [0, 1)
    shape_list[:, :, 5] = torch.round(shape_list[:, :, 5])
    shape_list[:, :, 5] %= 2

    return shape_list


def obfuscate_shape_construction(nb_shapes, shape_list, is_bordering, is_containing):
    shape_list, is_bordering, is_containing = randomize_shape_order(
        nb_shapes, shape_list, is_bordering, is_containing)
    shape_list = randomize_shape_rotations(shape_list, nb_shapes)
    shape_list = randomize_shape_reflections(shape_list, nb_shapes)
    return nb_shapes, shape_list, is_bordering, is_containing
