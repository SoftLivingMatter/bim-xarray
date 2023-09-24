from typing import Union, Optional, List, Dict, Any, Callable
from pathlib import Path

from xarray.core.dataarray import DataArray
from aicsimageio import AICSImage
from aicsimageio.readers.reader import Reader
from aicsimageio.types import MetaArrayLike

from . import metadata, process, constants
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
    timestamps: Optional[Union[
        float, 
        List[float], 
        MetaArrayLike
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

    # OME metadata
    # Reader-independent way of getting OME metadata is via reader's
    # .ome_metadata property, but may not be implemented for all readers
    # (while .xarray_data.attrs['processed'] might be available, it is
    # not guaranteed to be OME metadata)
    # Also, we can try to get scene-specific metadata and provide it
    # as a separate attribute for convenience
    try:
        ome_metadata = image_container.ome_metadata
        try:
            scene_meta = (
                ome_metadata.images[image_container.current_scene_index]
            )
        except AttributeError:
            scene_meta = None
            Warning("Cannot find scene-specific ome metadata.")
    except NotImplementedError:
        ome_metadata = scene_meta = None
    image.attrs[constants.METADATA_OME] = ome_metadata
    image.attrs[constants.METADATA_OME_SCENE] = scene_meta


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
    elif scene_meta is not None:
        p = scene_meta.pixels
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


    # time spacing
    # user-specified > ome_metadata
    # don't rely on ome_metadata here because it may not be available
    coords = {}
    time_per_frame = None
    if timestamps is not None:
        if isinstance(timestamps, (float, int)):
            time_per_frame = float(timestamps)
            coords[DimensionNames.Time] = Reader._generate_coord_array(
                0, image.sizes[DimensionNames.Time], time_per_frame
            )
        elif (isinstance(timestamps, (list, MetaArrayLike)) 
            and len(timestamps) == image.sizes[DimensionNames.Time]
        ):
            coords[DimensionNames.Time] = timestamps
        else:
            raise ValueError("Invalid timestamps provided.")
    # this branch is adpated from aicsimageio.metadata.utils.get_coords_from_ome
    # here we rely on ome_metadata
    elif scene_meta is not None:
        p = scene_meta.pixels
        if p.time_increment is not None:
            time_per_frame = float(p.time_increment)
            coords[DimensionNames.Time] = Reader._generate_coord_array(
                0, p.size_t, time_per_frame
            )
        elif (scene_meta.pixels.size_t > 1 
            and len(scene_meta.pixels.planes) > 0
        ):
            t_index_to_delta_map = {
                p.the_t: p.delta_t for p in scene_meta.pixels.planes
            }
            coords[DimensionNames.Time] = list(t_index_to_delta_map.values())
    # only if we can get actual time spacing, we assign it to the image
    # if only frame number is available, we drop the time dimension coord
    # while the dimension itself is kept (note this is different from 
    # aicsimageio's behavior whose fallback is integer frame number)
    if coords:
        image = image.assign_coords(coords)
    else:
        image = image.drop(DimensionNames.Time)
    if time_per_frame is not None:
        image.attrs['time_per_frame'] = time_per_frame

    return image


def imsave():
    pass

imwrite = imsave