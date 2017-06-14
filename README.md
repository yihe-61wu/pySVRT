# Introduction #

This is the port of the Synthetic Visual Reasoning Test to the pytorch
framework.

The main function is

```
torch.ByteTensor generate_vignettes(int problem_number, torch.LongTensor labels)
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
multi-core, and can be as slow as 40 vignettes per second s on a 4GHz
i7-6700K.
