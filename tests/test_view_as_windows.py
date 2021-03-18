import pytest
import dask.array
import numpy as np
from skimage.util import view_as_windows as skimage_view_as_windows

from dask_windows import view_as_windows as dask_view_as_windows

params = [
    ((10,), (10,), 10, 1),
    ((100,), (10,), 33, 1),
    ((100,), (1, 5, 14, 60, 1, 19), 10, 3),
    ((1000,), (88,), 27, 1),
    ((100, 100), (1, 100), 5, 6),
    ((100, 100, 100), (50, 3, 33), (5, 18, 10), (6, 3, 4)),
]


def assert_windows_equal_to_skimage(arr, chunks, window_shape, step):
    dask_arr = dask.array.from_array(arr, chunks=chunks)
    sk_win = skimage_view_as_windows(arr, window_shape, step)
    dask_win = dask_view_as_windows(dask_arr, window_shape, step)
    np.testing.assert_equal(sk_win, dask_win.compute())


@pytest.mark.parametrize("shape,chunks,window_shape,step", params)
def test_view_as_windows(shape, chunks, window_shape, step):
    arr = np.random.randint(0, 99, size=shape, dtype=np.uint8)
    assert_windows_equal_to_skimage(arr, chunks, window_shape, step)


def _gen_chunks(ndims, nchunks):
    if ndims == 0:
        return []
    elif ndims == 1:
        return [int(nchunks)]
    c = int(nchunks ** (1 / float(ndims)))
    return [c] + _gen_chunks(ndims - 1, nchunks / c)


def _gen_rand_chunk_shape(shape, nchunks):
    print(shape)
    ndims = len(shape)
    cs = _gen_chunks(ndims, nchunks)
    np.random.shuffle(cs)
    print(cs)

    def _rand_chunking(c, n):
        print(c, n)
        if c == 1:
            return n
        elif c == n:
            return 1
        else:
            low = int(np.ceil(n / c))
            high = int(np.ceil(n / (c - 1)))
            if low == high:
                return low
            else:
                return np.random.randint(low, high)

    chunks = tuple([_rand_chunking(c, n) for c, n in zip(cs, shape)])
    return chunks


@pytest.mark.parametrize("ndim", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_view_as_windows_ndims(ndim):
    n = 1_000_000
    nchunks = 100
    shape = _gen_chunks(ndim, n)
    chunks = _gen_rand_chunk_shape(shape, nchunks)
    step = [np.random.randint(1, s + 1) for s in shape]
    arr = np.random.randint(0, 99, size=shape, dtype=np.uint8)
    window_shape = 3
    assert_windows_equal_to_skimage(arr, chunks, window_shape, step)
