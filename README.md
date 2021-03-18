# Dask Windows

Equivalent of `skimage.util.view_as_windows` for dask arrays.

Example:

```
>>> import numpy as np
>>> import dask.array
>>> from dask_windows import view_as_windows

>>> x = dask.array.random.randint(0, 255, (10000, 10000), chunks=(1000, 1000), dtype=np.uint8)

>>> w = view_as_windows(x, window_shape=50, step=1)

>>> w
dask.array<_block_windows, shape=(9951, 9951, 50, 50), dtype=uint8, chunksize=(1000, 1000, 50, 50), chunktype=numpy.ndarray>

>>> print(f"{w.nbytes // (1024 ** 3)} GB")
230 GB

>>> w[5000, 5000, ...].compute()
array([[214, 233, 155, ..., 236,  53, 111],
       [231,  46, 100, ...,  32, 179, 220],
       [190, 212,  55, ..., 140, 243, 206],
       ...,
       [204,  50, 104, ..., 251, 176, 128],
       [254,  19,  99, ..., 119,   1, 219],
       [ 31,  80,  43, ...,  91, 180, 200]], dtype=uint8)
```
