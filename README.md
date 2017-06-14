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

```
make -j -k
./built.py
./test-svrt.py
```
