# Introduction #

This is the port of the Synthetic Visual Reasoning Test to the pytorch
framework.

The main function is

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

# Installation and test #

Executing

```
make -j -k
./test-svrt.py
```

should generate an image example.png in the current directory.

Note that the image generation does not take advantage of GPUs or
multi-core, and can be as fast as 10,000 vignettes per second and as
slow as 40 on a 4GHz i7-6700K.

# Vignette compression #

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

```
cnn-svrt.py
```

provides the implementation of two deep networks, and use the
compressed vignette code to allow the training with several millions
vignettes on a PC with 16Gb and a GPU with 8Gb.

The networks were designed by Afroze Baqapuri during an internship at
Idiap.
