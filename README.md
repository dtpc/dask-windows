# Dask Windows

Equivalent of `skimage.util.view_as_windows` for dask arrays.

Example:

```
>>> import numpy as np
>>> import dask.array
>>> from dask_windows import view_as_windows

>>> x = dask.array.random.randint(0, 255, (100_000, 100_000), chunks=(10_000, 10_000), dtype=np.uint8)

>>> w = view_as_windows(x, window_shape=50, step=3)

>>> w
dask.array<_block_windows, shape=(9951, 9951, 50, 50), dtype=uint8, chunksize=(1000, 1000, 50, 50), chunktype=numpy.ndarray>

>>> print(f"{w.nbytes // (1024 ** 3)} GB")
2584 GB

>>> w[5000, 5000, ...].compute()
array([[181, 155,  63, ..., 152, 154,  52],
       [ 45,  92, 200, ...,  63, 241, 253],
       [215, 246,  95, ..., 243,   9, 101],
       ...,
       [224,  97, 134, ...,  92, 247, 189],
       [112,  59,  44, ...,  45,  96, 237],
       [232,  13, 242, ..., 153,  35, 106]], dtype=uint8)

```

This is superceded by `dask.array.overlap.sliding_window_view` and slicing:

```
>>> from dask.array.overlap import sliding_window_view

>>> w2 = sliding_window_view(x, window_shape=[50, 50])[::3, ::3, ...]

>>> w2
dask.array<getitem, shape=(33317, 33317, 50, 50), dtype=uint8, chunksize=(3334, 3334, 50, 50), chunktype=numpy.ndarray>

>>> w2[5000, 5000, ...].compute()
array([[181, 155,  63, ..., 152, 154,  52],
       [ 45,  92, 200, ...,  63, 241, 253],
       [215, 246,  95, ..., 243,   9, 101],
       ...,
       [224,  97, 134, ...,  92, 247, 189],
       [112,  59,  44, ...,  45,  96, 237],
       [232,  13, 242, ..., 153,  35, 106]], dtype=uint8)

```