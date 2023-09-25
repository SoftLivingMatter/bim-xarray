from typing import Union, Optional, List, Dict, Any, Callable, Tuple
from pathlib import Path

import numpy as np
from xarray import DataArray
from aicsimageio import AICSImage
from aicsimageio.readers.reader import Reader
from aicsimageio.types import MetaArrayLike
from aicsimageio.transforms import reshape_data
from aicsimageio.writers import OmeTiffWriter

from . import metadata, process, constants, units
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
    elif DimensionNames.Time in image.coords:
        image = image.drop_vars(DimensionNames.Time)
    if time_per_frame is not None:
        image.attrs[constants.COORDS_SIZE_T] = time_per_frame

    # Spatial
    # If only "pixel" info is available, spatial coords will be dropped
    # (this differs from aicsimageio's default behavior)
    image = _drop_z_coords_if_2d(image)
    pps = _get_physical_pixel_sizes_dict(
        physical_pixel_sizes, scene_meta, image_container
    )
    image = metadata.attach_physical_pixel_sizes(image, pps)
    image = _ensure_spatial_coords_in_default_units(image, scene_meta, pps)


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
    return image


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


def _drop_z_coords_if_2d(image: DataArray) -> DataArray:
    if (image.sizes[DimensionNames.SpatialZ] == 1
        and DimensionNames.SpatialZ in image.coords
    ):
        image = image.drop_vars(DimensionNames.SpatialZ)
    return image


def _get_physical_pixel_sizes_dict(
    physical_pixel_sizes, scene_meta, image_container,
) -> Union[PhysicalPixelSizes, Dict[str, Optional[float]]]:
    # order: user-specified > ome_metadata > dict with Nones
    # no longer consider getting from reader's attribute directly
    # because we want to ensure able to get units from ome
    pps = {'X': None, 'Y': None, 'Z': None}

    # input physical_pixel_sizes can be either
    #    1. a dict with none or some keys including 'X', 'Y', 'Z' and
    #       values as floats or None
    #    2. an aioimageio PhysicalPixelSizes object
    if physical_pixel_sizes is not None:
        pps = physical_pixel_sizes
    # or we make a dict from scene-specific ome metadata
    elif scene_meta is not None:
        f = units.get_pixel_size_conversion_factors(scene_meta)
        if any(f.values()):
            p = scene_meta.pixels
            pps = {
                'X': p.physical_size_x * f['X'] if f['X'] else None,
                'Y': p.physical_size_y * f['Y'] if f['Y'] else None,
                'Z': p.physical_size_z * f['Z'] if f['Z'] else None,
            }

    return pps


def _ensure_spatial_coords_in_default_units(
    image: DataArray,
    scene_meta,
    pps: Union[PhysicalPixelSizes, Dict[str, Optional[float]]],
) -> DataArray:
    # modified from aicsimageio.metadata.utils.get_coords_from_ome
    coords = {}
    if pps['Z'] is not None and scene_meta.pixels.size_z > 1:
        coords[DimensionNames.SpatialZ] = Reader._generate_coord_array(
            0, scene_meta.pixels.size_z, pps['Z']
        )
    if pps['Y'] is not None and scene_meta.pixels.size_y > 1:
        coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
            0, scene_meta.pixels.size_y, pps['Y']
        )
    if pps['X'] is not None and scene_meta.pixels.size_x > 1:
        coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
            0, scene_meta.pixels.size_x, pps['X']
        )

    # only coords in default units will be added
    possible_spatial_coords_names = [
        DimensionNames.SpatialX, DimensionNames.SpatialY, DimensionNames.SpatialZ
    ]
    existing_spatial_coords_names = [
        d for d in possible_spatial_coords_names if d in image.coords
    ]
    image = image.drop_vars(existing_spatial_coords_names)
    image = image.assign_coords(coords)
    return image


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
            f = units.get_time_increment_conversion_factor(scene_meta)
            if f is not None:
                time_per_frame = float(p.time_increment) * f
                coords[DimensionNames.Time] = Reader._generate_coord_array(
                    0, p.size_t, time_per_frame
                )
        elif (scene_meta.pixels.size_t > 1 
            and len(scene_meta.pixels.planes) > 0
        ):
            f = units.get_delta_t_conversion_factor(scene_meta)
            if f is not None:
                t_index_to_delta_map = {
                    p.the_t: p.delta_t * f for p in scene_meta.pixels.planes
                }
                coords[DimensionNames.Time] = list(t_index_to_delta_map.values())
    return coords, time_per_frame


def imsave(
    image: DataArray, 
    fpath: str, 
    dim_order: Optional[Union[str, List[str]]] = None, 
    channel_names: Optional[List[str]] = None, 
    physical_pixel_sizes: Optional[Dict] = None,
) -> None:
    
    data = image.data

    # trasnpose to specified dim_order if different from current dims
    current_dims = "".join(image.dims)
    if dim_order is None:
        dim_order = current_dims
    elif dim_order != current_dims:
        data = reshape_data(image.data, current_dims, dim_order)

    if (channel_names is None 
        and DimensionNames.Channel in image.coords
    ):
        channel_names = np.atleast_1d(
            image.coords[DimensionNames.Channel].data)

    if physical_pixel_sizes is not None:
        if constants.COORDS_SIZE_SPATIAL in image.attrs.keys():
            Warning('Overwriting existing physical_pixel_sizes while saving')
        image = metadata.attach_physical_pixel_sizes(
            image, physical_pixel_sizes, forced=True
        )
    if constants.COORDS_SIZE_SPATIAL in image.attrs.keys():
        physical_pixel_sizes = metadata._to_aicsimageio_PhysicalPixelSizes(
            image.attrs[constants.COORDS_SIZE_SPATIAL]
        )

    OmeTiffWriter.save(
        data,
        fpath, 
        dim_order=dim_order, 
        channel_names=channel_names, 
        physical_pixel_sizes=physical_pixel_sizes,
    )

imwrite = imsave