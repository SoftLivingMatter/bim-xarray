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
    
    # list of channels to plot
    channel_list = atleast_1d(dataarray.coords[DimensionNames.Channel].values)
    
    # Generate the colormaps to use for each channel
    cmaps = _generate_cmaps(cmap, dataarray)

    # Process any additional keyword arguments to align them with the 
    # channel list
    processed_kwargs_list = [{} for _ in channel_list]
    for key, value in kwargs.items():
        processed_values = _align_input_to_channel_list(value, channel_list)
        for i, processed_value in enumerate(processed_values):
            processed_kwargs_list[i][key] = processed_value


    # plotting. populate individual channel images to use each own's
    # colormap and kwargs.
    nch = dataarray.sizes[DimensionNames.Channel]
    fig, axs = plt.subplots(nrows=1, ncols=nch, 
                            figsize=(nch * axsize[0], axsize[1]))       
    
    for channel, ax, cmap, processed_kwargs in zip(*map(atleast_1d, 
        [channel_list, axs, cmaps, processed_kwargs_list])
    ):
        # select one channel and keep its label so xarray will include
        # nice channel time in subplot title
        single_channel = dataarray.sel(
            {DimensionNames.Channel: channel}, drop=False,
        )
        cmap = "Greys_r" if cmap is None else cmap
        imshow_kwargs = dict(ax=ax, origin="upper", cmap=cmap)
        imshow_kwargs.update(processed_kwargs)
        
        # specify aspect ratio first so xarray.plot.imshow will create
        # a colorbar of height always matching to the image height
        # NOTE: xarray.plot.imshow doesn't support `aspect` kwarg
        ax.set_aspect("equal")
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
    else:
        cmaps = _align_input_to_channel_list(
            cmap, atleast_1d(dataarray.coords[DimensionNames.Channel])
        )

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


def _align_input_to_channel_list(
    input_value: Optional[Union[Any, List[Optional[Any]], Dict[str, Optional[Any]]]], 
    channel_list: List[str]
) -> Optional[List[Optional[Any]]]:
    """
    Aligns the input value to the channel_list. Broadcasting and
    reordering are done when necessary. If the input value is None,
    returns [None].
    
    Parameters
    ----------
    input_value: Any
        The input value, can be None, any type, a list of any type or 
        a dictionary of str to any type.
    channel_list: List
         A list of strings representing the channels.
    
    Returns:
    - A list of length equal to the length of channel_list or None.
    
    """
    
    n = len(channel_list)
    
    # If input_value is of any type except list or dict, create a list 
    # of length n with repeated values.
    if not isinstance(input_value, (list, dict)):
        return [input_value] * n
    
    # If input_value is a list, check its length and if it's not equal 
    # to n, raise a ValueError.
    elif isinstance(input_value, list):
        if len(input_value) != n:
            raise ValueError(
                f"Length of input list {len(input_value)} must match "
                f"the number of channels: {n}"
            )
        return input_value
    
    # If input_value is a dictionary, create a list of length n with 
    # values from the dictionary or None if the key is not present in 
    # the dictionary.
    elif isinstance(input_value, dict):
        return [input_value.get(ch, None) for ch in channel_list]
