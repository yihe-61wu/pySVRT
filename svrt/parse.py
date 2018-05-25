"""
Parsing symbolic output
"""

import math
import torch


def parse_vignette_to_string_classic(
        nb_shapes, shape_list, is_bordering, is_containing):
    """
    Parse symoblic form of vignette into textual output.
    Matching the style of sasquatch (Ellis et al., 2015).

    Parameter
    ---------
    nb_shapes : integer
        Number of shapes in the vignette.
    shape_list : Tensor [max_num_shapes, 6]
        Symbolic representation of each shape, containing:
        x, y, shape_id, rotation, scale, is_mirrored
    is_bordering : Tensor [max_num_shapes, max_num_shapes ]
        Whether each pair of shapes are touching .
    is_containing : Tensor [max_num_shapes, max_num_shapes ]
        For each pair of shapes, whether one shape entirely surrounds another.

    Returns
    -------
    string
        One multiline string encoding
    """

    # Normalise sizes
    shape_list[:, 4] /= torch.max(shape_list[:, 4])

    # First, parse each shape
    shape_reps = []
    for i_shape in range(nb_shapes):
        this_shape_str = "Shape({:.0f},{:.0f},{:.0f},{:f})".format(
            shape_list[i_shape, 0],
            shape_list[i_shape, 1],
            shape_list[i_shape, 2] + 1,
            shape_list[i_shape, 4])
        shape_reps.append(this_shape_str)
    out = ','.join(shape_reps)
    out += "\n"

    # Next, parse which shapes contain each other
    for i in range(nb_shapes):
        for j in range(nb_shapes):
            if i == j:
                continue
            if is_containing[i,j] > 0.5:
                out += "contains(" + str(i) + ", " + str(j) + ")\n"

    # Next, parse which shapes are bordering each other
    for i in range(nb_shapes):
        for j in range(i, nb_shapes):
            if i == j:
                continue
            if is_bordering[i,j] > 0.75:
                out += "borders(" + str(i) + ", " + str(j) + ")\n"

    return out


def parse_vignette_to_string(nb_shapes, shape_list, is_bordering,
                             is_containing):
    """
    Parse symoblic form of vignette into textual output.

    Parameter
    ---------
    nb_shapes : integer
        Number of shapes in the vignette.
    shape_list : Tensor [max_num_shapes, 6]
        Symbolic representation of each shape, containing:
        x, y, shape_id, rotation, scale, is_mirrored
    is_bordering : Tensor [max_num_shapes, max_num_shapes ]
        Whether each pair of shapes are touching .
    is_containing : Tensor [max_num_shapes, max_num_shapes ]
        For each pair of shapes, whether one shape entirely surrounds another.

    Returns
    -------
    string
        One multiline string encoding
    """

    # Normalise sizes
    shape_list[:, 4] /= torch.max(shape_list[:, 4])

    # First, parse each shape
    shape_reps = []
    for i_shape in range(nb_shapes):
        shape_list[i_shape][3] %= 2 * math.pi
        shape_list[i_shape][3] *= 180 / math.pi  # Convert to degrees
        this_shape_str = "Shape({:.0f},{:.0f},{:.0f},{:f},{:f},{:.0f})".format(
            shape_list[i_shape, 0],
            shape_list[i_shape, 1],
            shape_list[i_shape, 2] + 1,
            shape_list[i_shape, 4],
            shape_list[i_shape, 3],
            shape_list[i_shape, 5])
        shape_reps.append(this_shape_str)
    out = ','.join(shape_reps)
    out += "\n"

    # Next, parse which shapes contain each other
    for i in range(nb_shapes):
        for j in range(nb_shapes):
            if i == j:
                continue
            if is_containing[i,j] > 0.5:
                out += "contains(" + str(i) + ", " + str(j) + ")\n"

    # Next, parse which shapes are bordering each other
    for i in range(nb_shapes):
        for j in range(i, nb_shapes):
            if i == j:
                continue
            if is_bordering[i,j] > 0.75:
                out += "borders(" + str(i) + ", " + str(j) + ")\n"

    return out


def parse_vignettes_to_strings(nb_shapes, shape_list, is_bordering,
                               is_containing):
    """
    Parse symoblic form of vignette into textual output.

    Parameter
    ---------
    nb_shapes : Tensor [num_vignettes, ]
        Number of shapes in each vignette.
    shape_list : Tensor [num_vignettes, max_num_shapes, 6]
        Symbolic representation of each shape, containing:
        x, y, shape_id, rotation, scale, is_mirrored
    is_bordering : Tensor [num_vignettes, max_num_shapes, max_num_shapes ]
        Whether each pair of shapes are touching .
    is_containing : Tensor [num_vignettes, max_num_shapes, max_num_shapes ]
        For each pair of shapes, whether one shape entirely surrounds another.

    Returns
    -------
    list of strings
        One multiline string for each vignette.
    """

    out = []
    for i in range(len(nb_shapes)):
        out.append(
            parse_vignette_to_string(
                nb_shapes[i],
                shape_list[i],
                is_bordering[i],
                is_containing[i]
                )
            )
    return out
