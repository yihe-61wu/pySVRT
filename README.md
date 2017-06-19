# Introduction #

This is a port of the Synthetic Visual Reasoning Test problems to the
pytorch framework, with an implementation of two convolutional
networks to solve them.

# Installation and test #

Executing

```
make -j -k
./test-svrt.py
```

should generate an image
[`example.png`](https://fleuret.org/git-extract/pysvrt/example.png) in
the current directory.

Note that the image generation does not take advantage of GPUs or
multi-core, and can be as fast as 10,000 vignettes per second and as
slow as 40 on a 4GHz i7-6700K.

# Vignette generation and compression #

## Vignette sets ##

The file [`svrtset.py`](https://fleuret.org/git-extract/pysvrt/svrtset.py) implements the classes `VignetteSet` and
`CompressedVignetteSet` with the following constructor

```
__init__(problem_number, nb_samples, batch_size, cuda = False, logger = None)
```

and the following method to return one batch

```
(torch.FloatTensor, torch.LongTensor) get_batch(b)
```

as a pair composed of a 4d 'input' Tensor (i.e. single channel 128x128
images), and a 1d 'target' Tensor (i.e. Boolean labels).

## Low-level functions ##

The main function for genering vignettes is

```
torch.ByteTensor svrt.generate_vignettes(int problem_number, torch.LongTensor labels)
```

where

 * `problem_number` indicates which of the 23 problem to use
 * `labels` indicates the boolean labels of the vignettes to generate

The returned ByteTensor has three dimensions:

 * Vignette index
 * Pixel row
 * Pixel col

The two additional functions

```
torch.ByteStorage svrt.compress(torch.ByteStorage x)
```

and

```
torch.ByteStorage svrt.uncompress(torch.ByteStorage x)
```

provide a lossless compression scheme adapted to the ByteStorage of
the vignette ByteTensor (i.e. expecting a lot of 255s, a few 0s, and
no other value).

This compression reduces the memory footprint by a factor ~50, and may
be usefull to deal with very large data-sets and avoid re-generating
images at every batch. It induces a little overhead for decompression,
and moving from CPU to GPU memory.

See vignette_set.py for a class CompressedVignetteSet using it.

# Testing convolution networks #

The file
[`cnn-svrt.py`](https://fleuret.org/git-extract/pysvrt/cnn-svrt.py)
provides the implementation of two deep networks designed by Afroze
Baqapuri during an internship at Idiap, and allows to train them with
several millions vignettes on a PC with 16Gb and a GPU with 8Gb.
