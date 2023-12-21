'''
image operations
'''
from __future__ import annotations
import functools

from typing import Any, Tuple, Union, Callable
import numpy as np
import xarray as xr
from xarray.core import dtypes
from skimage.morphology import disk, binary_dilation, binary_erosion
from aicsimageio.dimensions import DimensionNames

# from .ximagetool import apply_ufunc_each_plane

    
def rescale_intensity(
    data: xr.DataArray,
    *,
    method: str = "zscore",
    **kwargs,
) -> xr.DataArray:
    """
    Rescale for 2D [+ T] [+ C] image, plane by plane.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
    method : str
        Normalizing method. Currently only 'zscore' and 'percentile'
        are supported. If using 'percentile', can also specify the
        clipping cutoff as `p` (default to 2, meaning 2% and 98%).
    """

    rescaled = xr.apply_ufunc(
        Rescale(method),
            data,
        input_core_dims=[['Y','X']],
        output_core_dims=[['Y','X']],
        kwargs={**kwargs},
        vectorize=True,
        keep_attrs='override',
    )
    return rescaled

def rescale_intensity_by(
    data: xr.DataArray, 
    by: xr.DataArray, 
    *,
    method: str = "zscore", 
    **kwargs,
) -> xr.DataArray:
    """
    Rescale for 2D [+ T] [+ C] image `data` by another image `by`.

    Similar to `Image.normalize` but uses a separate image to 
    calculate how to rescale.

    Parameters
    ----------
    data, by : xr.DataArray or xr.Dataset
        Normalize data based on by. Both are of the same type and 
        same dimentions.
    method : str
        Normalizing method. Choose from 'zscore' and 'pclip'
    **kwargs: dict
        passed as Normalize(method, **kwargs)
    """
    rescaled = xr.apply_ufunc(
        lambda x, y: Rescale(method).get_func(y, **kwargs)(x),
            data,
            by,
        input_core_dims=[['Y','X'], ['Y','X']],
        output_core_dims=[['Y','X']],
        vectorize=True,
        keep_attrs='override',
    )
    return rescaled


def ensure_signed(dataarray: xr.DataArray) -> xr.DataArray:
    """if image data is unsigned, convert to signed integer type
    """
    if np.issubdtype(dataarray.data.dtype, np.unsignedinteger):
        dataarray.data = _convert_to_signed_min_dtype(dataarray.data)
    return dataarray


def _convert_to_signed_min_dtype(arr):
    # Convert to the largest signed integer type first
    arr_signed = arr.astype(np.int64)

    # Check the maximum absolute value
    max_val = np.max(np.abs(arr_signed))

    # Choose the smallest signed integer type that can accommodate the data
    if max_val < np.iinfo(np.int8).max:
        return arr_signed.astype(np.int8)
    elif max_val < np.iinfo(np.int16).max:
        return arr_signed.astype(np.int16)
    elif max_val < np.iinfo(np.int32).max:
        return arr_signed.astype(np.int32)
    else:
        return arr_signed


def remove_background(dataarray: xr.DataArray, background) -> xr.DataArray:
    """Subtract background from image

    If background is a list of scalar, assume its order is same as
    the channel order of the image. If a scalar is provided, it is
    used as background for every channel in the image. If background
    is a dict, use the values corresponding to the keys that match
    the channel order in the image.
    """
    # ensure we have channel dimension to work with. currently 
    # dataarray might not have channel dimension even when it has 
    # channel axis label. (mostly after a .squeeze() call)
    if DimensionNames.Channel not in dataarray.dims:
        dataarray = dataarray.expand_dims(DimensionNames.Channel)
        need_to_squeeze_channel = True
    else:
        need_to_squeeze_channel = False

    # construct a list of background values for each channel
    # following the order of dataarray's channel axis label
    if np.isscalar(background):
        background = [background] * dataarray.sizes[DimensionNames.Channel]
    elif isinstance(background, dict):
        background = [
            background[oc] if oc in background else 0
            for oc in dataarray.coords[DimensionNames.Channel].values
        ]
    elif isinstance(background, list):
        if not (len(background) 
                == dataarray.sizes[DimensionNames.Channel]):
            raise ValueError("Invalid background given")
    # all other situations are not allowed for now
    else:
        raise ValueError("Invalid background given")

    background = xr.DataArray(
        data=background, 
        coords=[
            (DimensionNames.Channel, 
                dataarray.coords[DimensionNames.Channel].values)
        ]
    )
    dataarray = dataarray - background
    
    if need_to_squeeze_channel:
        dataarray = dataarray.squeeze(dim=DimensionNames.Channel, drop=False)
    
    return dataarray


def mask(
    data: xr.DataArray, 
    mask: str | xr.DataArray,
    *,
    complement: bool = False,
    other: Any = dtypes.NA,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Mask data by setting values outside the mask to 'other' (default: NaN).

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data to be masked.
    mask : xr.DataArray
        Mask to be applied. mask's dimensions must all be found in 
        data, and must not contain 'C' (channel) dimension.
    """
    if isinstance(mask, str):
        mask = data.coords[mask]
    elif not isinstance(mask, xr.DataArray):
        raise ValueError("mask needs to be coords var or DataArray")

    if complement:
        mask = ~ mask

    # This is a workaround for possibly having input mask with 
    # dimensions without coordinates. Note, `.sel()` will fall back
    # to using index if coordinate is not found.
    # 
    # This also assumes data and mask's common coords vars are each
    # in the same units. 
    # 
    # Above two are non-issues if mask is specified as a coords var of 
    # data.
    masked = (data.where(mask.sel({d: data.coords[d] for d in mask.dims}), 
                         other=other))
    
    return masked


# def erode_by_disk(mask: Union[xr.DataArray, xr.Dataset], 
#                     radius: int
# ) -> Union[xr.DataArray, xr.Dataset]:
#     return apply_ufunc_each_plane(binary_erosion, mask, 
#                                   footprint=disk(radius))


# def dilate_by_disk(mask: Union[xr.DataArray, xr.Dataset], 
#                     radius: int
# ) -> Union[xr.DataArray, xr.Dataset]:
#     return apply_ufunc_each_plane(binary_dilation, mask, 
#                                   footprint=disk(radius))


class Rescale:
    """
    The Rescale class provides methods to rescale image data using 
    z-score or percentile methods.
    
    The rescaling transformation is in the form of 
    rescaled = (data - A) / B.

    Attributes
    ----------
    method : str
        The normalization method. Supported methods are 'zscore' and 
        'percentile'. When 'percentile' is used, you can also specify 
        the clipping cutoff as `p` (default to 2, meaning 2% and 98%).

    Methods
    -------
    __init__(method: str)
        Initialize the Rescale object with the specified method.

    __call__(data: Union[xr.Dataset, xr.DataArray], **kwargs
        ) -> Union[xr.Dataset, xr.DataArray]
        Applies the rescaling to the provided data.

    get_params(data, **kwargs) -> Tuple[float, float]
        Computes and returns parameters A and B for the transformation.
        
    get_func(data, **kwargs) -> Callable
        Generates and returns the transformation function based on the 
        provided data.
    """

    def __init__(self, method: str) -> None:
        if method not in ['zscore', 'percentile']:
            raise ValueError(
                f"Normalizing method `{method}` not supported.")
        elif method == 'percentile':
            self._get_params = self._params_percentile
        elif method == 'zscore':
            self._get_params = self._params_zscore

    def __call__(self, data: Union[xr.Dataset, xr.DataArray], **kwargs
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Return rescaled data using generated transformation function
        """
        return self.get_func(data, **kwargs)(data)
    
    def get_params(self, data: Union[xr.Dataset, xr.DataArray], **kwargs
    ) -> Tuple[float, float]:
        """
        Return params A and B for transformation y = (data - A) / B.
        """
        return self._get_params(data, **kwargs)
    
    def get_func(self, data: Union[xr.Dataset, xr.DataArray],
                           **kwargs) -> Callable:
        """
        Return transformation function y = (data - A) / B
        """
        A, B = self.get_params(data, **kwargs)
        return lambda x: (x - A) / B
    
    @staticmethod
    def _params_zscore(data):
        A = np.nanmean(data)
        B = np.nanstd(data)
        return A, B
    
    @staticmethod
    def _params_percentile(data, p: float = 2.0):
        _p = np.nanpercentile(data, p)
        _q = np.nanpercentile(data, 100-p)
        A = _p
        B = _q - _p
        return A, B
