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

    # remove singleton dim but original coord label will remain
    image = image.squeeze(drop=False)


    # altering DataArray coordinates
    #
    if (channel_names is not None 
        and DimensionNames.Channel in image.coords
    ):
        image = metadata.label_channel_axis(image, channel_names)

    # remove channel label for object data (binary/label)
    # if it has only a single channel label
    if kind == 'object':
        if DimensionNames.Channel not in image.dims:
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