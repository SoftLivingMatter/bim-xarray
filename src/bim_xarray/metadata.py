from typing import Dict, NamedTuple, Optional, Union, List

from xarray import DataArray
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.dimensions import DimensionNames

from . import constants


# Image Utility Types
# Mirrors the pattern of aicsimageio.types.PhysicalPixelSizes
class TimeIncrement(NamedTuple):
    T: Optional[float]


def label_channel_axis(dataarray: DataArray, channel_names) -> DataArray:
    """Label channel axis with channel names if specified

    If channel_names is a list of strings, label the channel axis
    with the strings. If channel_names is a dict, label the channel
    axis with the values of the dict corresponding to the keys that
    match the channel order in the image.
    """

    # ensure we have channel dimension to work with. currently 
    # dataarray might not have channel dimension even when it has 
    # channel axis label. (mostly after a .squeeze() call)
    if DimensionNames.Channel not in dataarray.dims:
        dataarray = dataarray.expand_dims(DimensionNames.Channel)
        need_to_squeeze_channel = True
    else:
        need_to_squeeze_channel = False

    # ensure we have a list of channel_names of matching length to 
    # the channel axis label length

    # if a dict is given, worst case we still reuse original unreplaced
    # channel axis label(s)
    if isinstance(channel_names, Dict):
        channel_names = [
            channel_names[oc] if oc in channel_names else oc
            for oc in dataarray.coords[DimensionNames.Channel].values
        ]  
    # if a list is given, it needs to have a matching length of 
    # the original label
    elif (isinstance(channel_names, List) 
          and len(channel_names) == dataarray.coords[DimensionNames.Channel].size
    ):
        channel_names = channel_names
    # if a string is given, there should be only one original label
    elif (isinstance(channel_names, str)
          and dataarray.coords[DimensionNames.Channel].size == 1
    ):
        channel_names = [channel_names]
    else:
        raise ValueError("Invalid channel_names provided")

    dataarray = dataarray.assign_coords({DimensionNames.Channel: channel_names})

    if need_to_squeeze_channel:
        dataarray = dataarray.squeeze(dim=DimensionNames.Channel, drop=False)

    return dataarray


def _to_aicsimageio_PhysicalPixelSizes(
    pps: Optional[Dict[str, float]] = None,
    *,
    X: Optional[float] = None,
    Y: Optional[float] = None,
    Z: Optional[float] = None,
) -> PhysicalPixelSizes:
    """
    Generate PhysicalPixelSizes used internally in aicsimageio.

    If no arguments passed (all None's), will still return an 
    object PhysicalPixelSizes with all three fields None.

    Parameters
    ----------
    pps: dict 
        keys including none or some of 'X', 'Y', and 'Z'. Specifying
        `pps` is not compat with kwargs (`X`, `Y`, `Z`).
    X: float 
    Y: float
    Z: float

    Raises
    ------
    ValueError: If X, Y, or Z AND pps are both provided
    """
    # first check if only one method is used for speification
    if pps and any([X, Y, Z]):
        raise ValueError("Expect either a dict or kwargs, "
                         "but both are specified.")

    # non-existing fields of positional dict will be patched
    if pps is not None:
        for k in ['X', 'Y', 'Z']:
            if k not in pps.keys():
                pps[k] = None
    else:
        pps = dict(X=X, Y=Y, Z=Z)

    return PhysicalPixelSizes(**pps)
    

def _from_aicsimageio_PhysicalPixelSizes(
    pps: PhysicalPixelSizes,
) -> Dict[str, Optional[float]]:
    """
    Convert aicsimageio's PhysicalPixelSizes to a dict.

    It is guaranteed the resulting dict will always have keys
    'X', 'Y', and 'Z', but their values can be `None`.
    """
    if not isinstance(pps, PhysicalPixelSizes):
        raise ValueError(
            f"pps should be of type {type(PhysicalPixelSizes)}, "
            f"but {type(pps)} was given."
        )
    
    return {
        'X': pps.X if pps.X is not None else None,
        'Y': pps.Y if pps.Y is not None else None,
        'Z': pps.Z if pps.Z is not None else None,
    }


def attach_physical_pixel_sizes(
    dataarray: DataArray, 
    pps: Union[PhysicalPixelSizes, 
               Dict[str, Optional[float]]],
    overwrite: bool = False,
) -> DataArray:
    if (constants.COORDS_SIZE_SPATIAL in dataarray.coords.keys()
        and not overwrite
    ):
        raise ValueError(
            f"DataArray already has {constants.COORDS_SIZE_SPATIAL} "
            f"attached. Use `overwrite=True` to overwrite."
        )

    # make sure we always have a dict with all three fields
    if isinstance(pps, dict):
        pps = _to_aicsimageio_PhysicalPixelSizes(pps)
    pps = _from_aicsimageio_PhysicalPixelSizes(pps)

    return dataarray.assign_attrs({constants.COORDS_SIZE_SPATIAL: pps})


def attach_time_increment():
    raise NotImplementedError


def attach_channel_colors():
    raise NotImplementedError


