"""
I/O Utilities
"""

import os.path

import h5py
import torch

def load_from_h5(dirname, problem, label=None):
    '''
    Loads a symbolic representation of a dataset from HDF5.

    Arguments
    ---------
    dirname : str
        Path to directory where HDF5 file can be found.
    problem : int
        ID of the problem to load.
    label : [0, 1, None]
        Which class label to load. If None, both 0 and 1 are loaded. Default
        is None.

    Yields
    ------
    labels : torch.LongTensor
        Labels for each sample vignette.
    nb_shapes : torch.ByteTensor
        Number of shapes in each sample vignette.
    shape_list : torch.FloatTensor
        Details (dim 2) about each shape (dim 1) in each vignette (dim 0).
    intershape_distance : torch.FloatTensor
        For each pair of shapes, are they bordering each other.
    is_containing : torch.FloatTensor
        For each pair of shapes, is one inside the other.
    '''
    if label is None:
        # We start with class 1, because they are generated first
        labels_to_load = [1, 0]
    else:
        labels_to_load = [label]

    fname = os.path.join(dirname, 'problem_{:02d}.h5'.format(problem))

    with h5py.File(fname, 'r') as f:
        # Need to know what records are in the HDF5
        l = labels_to_load[0]
        record_names = f['class_{}'.format(l)]['nb_shapes'].keys()

        for record_name in record_names:
            # We'll combine both classes for this record and yield them
            # together
            labels = []
            nb_shapes = []
            shape_list = []
            intershape_distance = []
            is_containing = []

            for l in labels_to_load:
                # Load each of the variables from the HDF5 file
                nb_shapes.append(
                    torch.ByteTensor(
                        f['class_{}'.format(l)]['nb_shapes'][record_name]
                        )
                    )
                shape_list.append(
                    torch.FloatTensor(
                        f['class_{}'.format(l)]['shape_list'][record_name]
                        )
                    )
                intershape_distance.append(
                    torch.FloatTensor(
                        f['class_{}'.format(l)]['intershape_distance'][record_name]
                        )
                    )
                is_containing.append(
                    torch.FloatTensor(
                        f['class_{}'.format(l)]['is_containing'][record_name]
                        )
                    )
                # Generate a vector of the correct length showing the labels
                labels.append(
                    torch.LongTensor(nb_shapes[-1].size(0)).fill_(l)
                    )

            labels = torch.cat(labels)
            nb_shapes = torch.cat(nb_shapes)
            shape_list = torch.cat(shape_list)
            intershape_distance = torch.cat(intershape_distance)
            is_containing = torch.cat(is_containing)

            yield labels, nb_shapes, shape_list, intershape_distance, is_containing
