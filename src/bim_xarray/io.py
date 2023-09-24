from typing import Any, Callable, Optional, Union, Dict, List

from pathlib import Path

from xarray import DataArray
from aicsimageio import AICSImage
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.dimensions import DimensionNames

from . import metadata
from . import process


def imread(
    fpath: Union[Path, str],
    *,
    channel_names: Optional[Union[
        str,
        List[str],
        Dict[str, Optional[str]],
    ]] = None,
    physcial_pixel_sizes: Optional[Union[
        PhysicalPixelSizes, 
        Dict[str, Optional[float]]
    ]] = None,
    preserve_dtype: bool = False,
    kind: Optional[str] = None,
    preprocess: Optional[Callable] = None,
    **kwargs: Any,
) -> DataArray:
    """Read image from file into 5D DataArray.

    Wraps around aicsimageio.AICSImage with sensible defaults. Primary 
    use is to return xarray.DataArray with ome scene-metadata always 
    exposed.

    Parameters
    ----------
    fpath : Union[Path, str]
        path or uri to image
    channel_names : Optional[Union[ str, List[str], Dict[str, Optional[str]], ]], optional
        list of str for each channel (must match exact number of channels),
        or dict mapping optical config to channel names (can skip channels),
        by default None
    physcial_pixel_sizes : Optional[Union[ PhysicalPixelSizes, Dict[str, Optional[float]] ]], optional
        spatial dimension pixel sizes with unit in micron, keys other
        than 'X', 'Y', and 'Z' will ignored. by default None
    preserve_dtype : bool, optional
        strictly preserve original dtype and prevent casting unsigned 
        to signed integers, by default False
    kind : Optional[str], optional
        'intensity' ('i') or 'object' ('o'), by default None
    preprocess : Optional[Callable], optional
        not supported yet, by default None
    kwargs : 
        keyword arguments to pass to aicsimageio.AICSImage. 

    Returns
    -------
    DataArray
        5D data with coordinates if can be parsed from metadata
    """

    # checking arguments
    #

    # resolve shorthands    
    if kind == 'i':
        kind = 'intensity'
    elif kind == 'o':
        kind = 'object'


    # read data as DataArray 
    #

    # start with 5D DataArray
    image = AICSImage(fpath, **kwargs).xarray_data


    # altering DataArray shape
    #

    # altering DataArray coordinates
    #
    if channel_names is not None:
        image = metadata.label_channel_axis(image, channel_names)

    # remove channel label for object data (binary/label)
    # if it has only a single channel label
    if kind == 'object':
        if image.sizes[DimensionNames.Channel] == 1:
            image = image.drop_vars(DimensionNames.Channel)
        else:
            Warning("Data specified as object, but channel axis "
                    "is not singleton. Consider converting to "
                    "a Dataset.")
    
    
    # altering DataArray variables & their data
    #

    # if not strictly preserving dtype, convert to signed dtype to
    # make it safe for downstream ariethmetic operations 
    if not preserve_dtype:
        image = process.ensure_signed(image)

    if preprocess is not None:
        image.data = preprocess(image.data)


    # altering DataArray attrs
    #

    if physcial_pixel_sizes is not None:
        image = metadata.attach_physical_pixel_sizes(
            image, physcial_pixel_sizes)


    return image


def imsave():
    pass

imwrite = imsave