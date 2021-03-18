from itertools import groupby
from typing import Dict, Sequence, Tuple, Union

import dask.array
import numpy as np
from skimage.util import view_as_windows as skimage_view_as_windows


def _window_indices_and_chunks(
    input_chunks: Tuple[int, ...], window_len: int, step: int = 1
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Calculate window indices into overlaped blocks and resulting chunks."""
    overlap = window_len // 2
    n = sum(input_chunks)

    window_ixs_e = np.arange(window_len, n + 1, step=step)
    window_ixs_s = window_ixs_e - window_len

    chunk_cs = np.cumsum(input_chunks)
    chunk_ixs_s = np.insert(chunk_cs - overlap, 0, 0)[:-1]
    chunk_ixs_e = np.insert(chunk_cs[:-1] + overlap, len(chunk_cs) - 1, chunk_cs[-1])

    window_chunk_ixs = np.searchsorted(chunk_ixs_e, window_ixs_e, side="left")

    window_chunks = [0] * len(input_chunks)
    chunk_local_ixs = [-1] * len(input_chunks)
    win_ctr = 0
    for c, g in groupby(window_chunk_ixs):
        num_win = len(list(g))
        window_chunks[c] = num_win
        chunk_local_ixs[c] = window_ixs_s[win_ctr] - chunk_ixs_s[c]
        win_ctr += num_win

    return np.array(chunk_local_ixs), tuple(window_chunks)


def _block_windows(
    arr: np.ndarray,
    wix_arr: np.ndarray,
    window_shape: Union[int, Tuple[int, ...]],
    step: Union[int, Tuple[int, ...]],
    block_info: Dict,
) -> np.ndarray:
    """Return numpy view of windows when mapped over dask array blocks."""
    chunk_shape = block_info[None]["chunk-shape"]
    if 0 in chunk_shape:
        windows = np.empty(chunk_shape, dtype=block_info[None]["dtype"])
    else:
        ix = tuple(slice(i, None) for i in wix_arr.item())
        windows = skimage_view_as_windows(arr[ix], window_shape, step)
    return windows


def _merge_small_chunks(chunks: Tuple[int, ...], min_size: int) -> Tuple[int, ...]:
    """Merge together small chunks with neighbouring chunks."""
    newchunks = [chunks[0]]
    for c in chunks[1:]:
        if newchunks[-1] < min_size:
            newchunks[-1] += c
        else:
            newchunks.append(c)

    if newchunks[-1] < min_size:
        newchunks = newchunks[:-2] + [sum(newchunks[-2:])]
    return tuple(newchunks)


def _validate_window_shape_step(
    arr_shape: Tuple[int, ...],
    window_shape: Union[int, Sequence[int]],
    step: Union[int, Sequence[int]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Validate window_shape and step parameters."""
    ndim = len(arr_shape)

    if isinstance(window_shape, int):
        window_shape = (window_shape,) * ndim

    if isinstance(step, int):
        step = (step,) * ndim

    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr.shape`")

    if len(window_shape) != ndim:
        raise ValueError("`window_shape` is incompatible with `arr.shape`")

    if any(w > a for w, a in zip(window_shape, arr_shape)):
        raise ValueError("`window_shape` is too large")

    if any(w < 1 for w in window_shape):
        raise ValueError("`window_shape` is too small")

    return tuple(window_shape), tuple(step)


def view_as_windows(
    arr: dask.array.Array,
    window_shape: Union[int, Sequence[int]],
    step: Union[int, Sequence[int]] = 1,
) -> dask.array.Array:
    """Rolling window view of the input n-dimensional dask array.

    Dask array equivalent of skimage.util.view_as_windows. Each resulting chunk is a numpy
    view consisting of windows.

    :param arr: N-d input dask array
    :param window_shape: Shape of the rolling window (integer shape applies to all dims)
    :param step: Rolling window step size (integer step applies to all dims)
    :return: Dask array (rolling) window view of the input array
    """
    window_shape, step = _validate_window_shape_step(arr.shape, window_shape, step)

    if arr.ndim == 0:
        return arr

    depth = tuple(w // 2 for w in window_shape)
    fix_chunks = [_merge_small_chunks(cs, d) for cs, d in zip(arr.chunks, depth)]
    arr = arr.rechunk(fix_chunks)

    window_ixs, window_chunks = zip(
        *[
            _window_indices_and_chunks(c, w, s)
            for c, w, s in zip(arr.chunks, window_shape, step)
        ]
    )

    # create array of slices per block for calculating windows
    # this array blocksize matches arr
    wix = np.array(np.meshgrid(*window_ixs, indexing="ij"))
    wss_dtype = [(f"f{i}", "i") for i in range(arr.ndim)]
    wss = np.apply_along_axis(lambda a: np.array(tuple(a), dtype=wss_dtype), 0, wix)
    wss_dask = dask.array.from_array(wss, chunks=(1,) * arr.ndim)

    arr_o = dask.array.overlap.overlap(arr, depth=depth, boundary="none")
    output_chunks = tuple(w for w in window_chunks if w) + tuple(
        (w,) for w in window_shape
    )

    windows = dask.array.map_blocks(
        _block_windows,
        arr_o,
        wss_dask,
        dtype=arr_o.dtype,
        chunks=output_chunks,
        new_axis=tuple(range(arr.ndim, arr.ndim * 2)),
        window_shape=window_shape,
        step=step,
        meta=np.array((), dtype=arr_o.dtype),
    )

    # remove zero length chunks
    cc = tuple(tuple(c for c in cs if c > 0) for cs in windows.chunks)

    if cc != windows.chunks:
        windows = windows.rechunk(cc)

    return windows
