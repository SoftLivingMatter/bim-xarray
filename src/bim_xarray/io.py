from typing import Union, Optional, List, Dict, Any, Callable
from pathlib import Path

from xarray.core.dataarray import DataArray
from aicsimageio import AICSImage

from . import metadata, process
from .metadata import DimensionNames, PhysicalPixelSizes


def imread(
    fpath: Union[Path, str],
    *,
    scene_id: Optional[Union[str, int]] = None,
    channel_names: Optional[Union[
        str,
        List[str],
        Dict[str, Optional[str]],
    ]] = None,
    physical_pixel_sizes: Optional[Union[
        PhysicalPixelSizes, 
        Dict[str, Optional[float]]
    ]] = None,
    preserve_dtype: bool = False,
    kind: Optional[str] = None,
    preprocess: Optional[Callable] = None,
    **kwargs: Any,
) -> DataArray:
    """
    Read image from file into 5D DataArray.

    This function wraps around `aicsimageio.AICSImage` with sensible 
    defaults. Its primary use is to return an `xarray.DataArray` with
    OME scene-metadata always exposed.

    Parameters
    ----------
    fpath : Union[Path, str]
        Path or URI to image.
    scene_id : Optional[Union[str, int]], optional
        ID of the scene to read, by default None.
    channel_names : Optional[Union[ str, List[str], Dict[str, Optional[str]], ]], optional
        List of strings for each channel (must match exact number of 
        channels), or dict mapping optical config to channel names 
        (can skip channels), by default None.
    physical_pixel_sizes : Optional[Union[ PhysicalPixelSizes, Dict[str, Optional[float]] ]], optional
        Spatial dimension pixel sizes with unit in micron, keys other
        than 'X', 'Y', and 'Z' will be ignored, by default None.
    preserve_dtype : bool, optional
        Strictly preserve original dtype and prevent casting unsigned 
        to signed integers, by default False.
    kind : Optional[str], optional
        'intensity' ('i') or 'object' ('o'), by default None.
    preprocess : Optional[Callable], optional
        Not supported yet, by default None.
    kwargs : 
        Keyword arguments to pass to `aicsimageio.AICSImage`. 

    Returns
    -------
    DataArray
        5D data with coordinates if can be parsed from metadata, with:
        .ndim <= 5 (always squeezed)
        .attrs:
            'ome_metadata' guaranteed, scene-specific (OME.Image | None])
            'ome_metadata_full' guaranteed (OME | None)
            'unprocessed' & 'processed' may not exist (reader-dependent)
            'physical_pixel_sizes' guaranteed (may be dict[str, None])

    Raises
    ------
    NotImplementedError
        If `preprocess` is not None.

    Warning
    -------
    If `ome_metadata` cannot be found.
    If `kind` is 'object' but channel axis is not singleton.
    If `physical_pixel_sizes` cannot be parsed.

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

    image_container = AICSImage(fpath, **kwargs)
    if scene_id is not None:
        image_container.set_scene(scene_id)
    # start with 5D array
    image = image_container.xarray_data
    # ome_metadata can be found reliably via reader's dedicated method
    # and not always under xarray_data.attrs['processed']
    # also distinguishes full vs. scene-specific (Image) metadata
    try:
        ome_metadata = image_container.ome_metadata
    except NotImplementedError:
        ome_metadata = None
    image.attrs['ome_metadata_full'] = ome_metadata
    if ome_metadata is not None:
        try:
            image.attrs['ome_metadata'] = (
                ome_metadata.images[image_container.current_scene_index])
        except AttributeError:
            image.attrs['ome_metadata'] = None
            Warning("Cannot find scene-specific ome metadata.")


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
    # make it safe for downstream arithmetic operations 
    if not preserve_dtype:
        image = process.ensure_signed(image)

    if preprocess is not None:
        raise NotImplementedError("Preprocessing not supported yet.")


    # altering DataArray attrs
    #

    # physical pixel sizes
    # user-specified > ome_metadata > reader > dict with Nones
    if physical_pixel_sizes is not None:
        pps = physical_pixel_sizes
    elif image.attrs['ome_metadata'] is not None:
        p = image.attrs['ome_metadata'].pixels
        pps = {
            'X': p.physical_size_x, 
            'Y': p.physical_size_y, 
            'Z': p.physical_size_z
        }
    else:
        try:
            pps = image_container.physical_pixel_sizes
        except ValueError:
            Warning("Cannot parse physical_pixel_sizes. "
                    "Setting all to None.")
            pps = {'X': None, 'Y': None, 'Z': None}
    # guaranteed to have a dict or PhysicalPixelSizes object
    image = metadata.attach_physical_pixel_sizes(image, pps)


    return image


def imsave():
    pass

imwrite = imsave