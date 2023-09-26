from typing import Optional, Any, Tuple, Union, Dict, List

from numpy import atleast_1d
from xarray import DataArray
import matplotlib.pyplot as plt
from matplotlib import colors

from aicsimageio.dimensions import DimensionNames

from . import constants

def plot_by_channels(
    dataarray: DataArray, 
    *,
    cmap: Optional[Union[
        str, 
        Tuple[float, float, float, float],
        List[Optional[Union[str, Tuple[float, float, float, float]]]],
        Dict[str, Optional[Union[str, Tuple[float, float, float, float]]]],
    ]] = "Greys_r",
    axsize: Optional[Tuple[float, float]] = (5, 4),
    **kwargs: Any,
) -> Optional[plt.Figure]:
    
    scalar_dims = [d for d in dataarray.dims if dataarray.sizes[d] == 1]
    dataarray = dataarray.squeeze(scalar_dims, drop=False)
    # now, dim: non-scalar dims

    _check_2d_plus_channel(dataarray)
    # now, dim: at most 2D [+ C]
    
    # we might start without channel dimension or lost it after squeeze
    # so we need to make sure we have channel dimension for plotting
    if DimensionNames.Channel not in dataarray.dims:
        dataarray = dataarray.expand_dims(DimensionNames.Channel)
    # now dim: C + at most 2D
    
    
    # colormaps
    cmaps = _generate_cmaps(cmap, dataarray)

    # plotting. populate individual channel images to use automatic
    # scaling for better default contrast
    nch = dataarray.sizes[DimensionNames.Channel]
    fig, axs = plt.subplots(1, nch, 
                            figsize=(nch * axsize[0], axsize[1]))       
    
    for channel, ax, cmap in zip(
        atleast_1d(dataarray.coords[DimensionNames.Channel].values), 
        atleast_1d(axs),
        atleast_1d(cmaps),
    ):
        if cmap is None:
            cmap = "Greys_r"
        
        # select one channel and keep its label so xarray will include
        # nice channel time in subplot title
        single_channel = dataarray.sel(
            {DimensionNames.Channel: channel}, drop=False,
        )
        imshow_kwargs = dict(origin="upper", ax=ax, cmap=cmap)
        imshow_kwargs.update(kwargs)
        single_channel.plot.imshow(**imshow_kwargs)
    
    fig.set_tight_layout(True)
    return fig


def _check_2d_plus_channel(dataarray: DataArray):
    real_dims_not_channel = [
        d for d in dataarray.dims 
        if dataarray.sizes[d] > 1 and d != DimensionNames.Channel
    ]

    if len(real_dims_not_channel) > 2:
        raise ValueError(
            f"Selected dataarray must be 2D[+Channel]. \n"
            f"Consider subsetting the image first, from "
            f"non-scalar dimensions: \n{real_dims_not_channel}?"
        )
    

def _generate_cmaps(
    cmap: Optional[Union[
        str, 
        Tuple[float, float, float, float],
        List[Optional[Union[str, Tuple[float, float, float, float]]]],
        Dict[str, Optional[Union[str, Tuple[float, float, float, float]]]],
    ]],
    dataarray: DataArray,
) -> List[Optional[Union[str, colors.Colormap]]]:
    
    nch = dataarray.sizes[DimensionNames.Channel]

    # first make sure we have a list of cmap for each channel
    # then we will check and convert tuple to Colormap object
    if isinstance(cmap, str) and cmap == "auto":
        try:
            dct = dataarray.attrs[constants.CHANNEL_COLORS]
            cmaps = [
                dct[ch] 
                for ch in atleast_1d(dataarray.coords[DimensionNames.Channel])
            ]
        except KeyError:
            Warning("Cannot detect channel colors. Using default colormap.")
            cmaps = ["Greys_r"] * nch
    elif isinstance(cmap, (str, tuple)):
        cmaps = [cmap] * nch
    elif isinstance(cmap, list):
        if len(cmap) != nch:
            raise ValueError(
                f"Length of cmap list {len(cmap)} must match number "
                f"of channels: {nch}"
            )
        cmaps = cmap
    elif isinstance(cmap, dict):
        cmaps = [
            cmap[ch] if (ch in cmap) else None
            for ch in atleast_1d(dataarray.coords[DimensionNames.Channel]) 
        ]
    else:
        raise ValueError()

    cmaps = [
        None if c is None else (_rgb_to_cmap(c) if isinstance(c, tuple) else c)
        for c in cmaps
    ]
    
    return cmaps


def _rgb_to_cmap(
    rgb: Union[
        Tuple[float, float, float], 
        Tuple[float, float, float, float]
    ]
) -> colors.Colormap:
    """
    Convert a RGB(A) tuple to a matplotlib colormap; ignore A if provided.
    """
    # we will discard alpha channel

    # Define a color dictionary
    color_dict = {
        "red":   [(0, 0, 0), (1, rgb[0], rgb[0])],
        "green": [(0, 0, 0), (1, rgb[1], rgb[1])],
        "blue":  [(0, 0, 0), (1, rgb[2], rgb[2])],
        "alpha": [(0, 1, 1), (1, 1, 1)],
    }
    cmap_name = f"r{rgb[0]}_g{rgb[1]}_b{rgb[2]}"
    return colors.LinearSegmentedColormap(cmap_name, color_dict)