from typing import Union, Optional, List, Dict, Any, Callable, Tuple
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
    timestamps : Optional[Union[float, List[float], MetaArrayLike]], optional
        List of timestamps for each timepoint (must match exact number 
        of timepoints), by default None.
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
        .ndim = 5 (intensity image never squeezed) or 4 (single object)
        .attrs:
            'ome_metadata' guaranteed, scene-specific (OME.Image | None])
            'ome_metadata_full' guaranteed (OME | None)
            'unprocessed' & 'processed' may not exist (reader-dependent)
            'physical_pixel_sizes' guaranteed (may be dict[str, None])
            'time_per_frame' may not exist

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
    # Resolve shorthands    
    if kind == constants.IMAGE_KIND_INTENSITY_SHORT:
        kind = constants.IMAGE_KIND_INTENSITY
    elif kind == constants.IMAGE_KIND_BINARY_OR_LABEL_SHORT:
        kind = constants.IMAGE_KIND_BINARY_OR_LABEL


    # Read data as DataArray, from a specific scene 
    image_container = AICSImage(fpath, **kwargs)
    if scene_id is not None:
        image_container.set_scene(scene_id)
    image = image_container.xarray_data


    # Add OME metadata to attributes
    ome_metadata, scene_meta = _get_ome_metadata(image_container)
    image.attrs[constants.METADATA_OME] = ome_metadata
    image.attrs[constants.METADATA_OME_SCENE] = scene_meta


    # Enhanced coordinates handling
    # Channel
    image = _update_channel_coords(image, channel_names)
    image = _drop_channel_dim_for_single_object(image, kind)
    
    # Time
    # If only "frame" info is available, time coords will be dropped
    # (this differs from aicsimageio's default behavior)
    coords, time_per_frame = _get_time_spacing(
        image, timestamps, scene_meta
    )
    if coords:
        image = image.assign_coords(coords)
    else:
        image = image.drop(DimensionNames.Time)
    if time_per_frame is not None:
        image.attrs[constants.COORDS_SIZE_T] = time_per_frame

    # Spatial
    # Nothing enhanced yet, only attaching physical pixel sizes as attrs
    pps = _get_physical_pixel_sizes_dict(
        physical_pixel_sizes, scene_meta, image_container
    )
    image = metadata.attach_physical_pixel_sizes(image, pps)


    # Array data processing
    image = _ensure_signed_dtype(image, preserve_dtype)
    if preprocess is not None:
        raise NotImplementedError("Preprocessing not supported yet.")

    return image


def _get_ome_metadata(image_container: AICSImage) -> Tuple[Optional[Any], Optional[Any]]:
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
    return ome_metadata, scene_meta


def _drop_channel_dim_for_single_object(image: DataArray, kind: Optional[str]) -> DataArray:
    if kind == constants.IMAGE_KIND_BINARY_OR_LABEL:
        if image.sizes[DimensionNames.Channel] == 1:
            image = image.drop_vars(DimensionNames.Channel)
        else:
            Warning(f"Data specified as {constants.IMAGE_KIND_BINARY_OR_LABEL}, "
                    "but channel axis is not scalar. "
                    "Consider converting it to a Dataset.")
    return image.squeeze()


def _update_channel_coords(image: DataArray, channel_names: Optional[Union[
        str,
        List[str],
        Dict[str, Optional[str]],
    ]] = None) -> DataArray:
    if channel_names is not None:
        image = metadata.label_channel_axis(image, channel_names)
    return image


def _ensure_signed_dtype(image: DataArray, preserve_dtype: bool) -> DataArray:
    if not preserve_dtype:
        image = process.ensure_signed(image)
    return image


def _get_physical_pixel_sizes_dict(
    physical_pixel_sizes, scene_meta, image_container,
) -> Union[PhysicalPixelSizes, Dict[str, Optional[float]]]:
    # order: user-specified > ome_metadata > reader > dict with Nones
    
    # input physical_pixel_sizes can be either
    #    1. a dict with none or some keys including 'X', 'Y', 'Z' and
    #       values as floats or None
    #    2. an aioimageio PhysicalPixelSizes object
    if physical_pixel_sizes is not None:
        pps = physical_pixel_sizes
    # or we make a dict from scene-specific ome metadata
    elif scene_meta is not None:
        p = scene_meta.pixels
        pps = {
            'X': p.physical_size_x, 
            'Y': p.physical_size_y, 
            'Z': p.physical_size_z
        }
    # or we get it from the reader's attribute, and if that fails,
    # we make a dict with all Nones
    else:
        try:
            pps = image_container.physical_pixel_sizes
        except ValueError:
            Warning("Cannot parse physical_pixel_sizes. "
                    "Setting all to None.")
            pps = {'X': None, 'Y': None, 'Z': None}
    return pps


def _get_time_spacing(
    image: DataArray,
    timestamps: Optional[Union[
        float, 
        List[float], 
        MetaArrayLike
    ]] = None,
    scene_meta: Optional[Any] = None
) -> Tuple[Dict[str, Any], Optional[float]]:
    """
    Returns coordinates and time per frame. If only "frame" info is
    available, coordinates returned is an empty dict. If not uniform
    time spacing, time_per_frame is None.
    """
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
    return coords, time_per_frame


def imsave():
    pass

imwrite = imsave