from typing import Optional, Any, Tuple

from numpy import atleast_1d
from xarray import DataArray
import matplotlib.pyplot as plt

from aicsimageio.dimensions import DimensionNames

def plot_by_channels(
    dataarray: DataArray, 
    *,
    cmap: str = "Greys_r",
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
    
    
    # plotting. populate individual channel images to use automatic
    # scaling for better default contrast
    nch = dataarray.sizes[DimensionNames.Channel]
    fig, axs = plt.subplots(1, nch, 
                            figsize=(nch * axsize[0], axsize[1]))       
    
    for channel, ax in zip(
        atleast_1d(dataarray.coords[DimensionNames.Channel].values), 
        atleast_1d(axs)
    ):
        # select one channel and keep its label so xarray will include
        # nice channel time in subplot title
        single_channel = dataarray.sel(
            {DimensionNames.Channel: channel}, drop=False,
        )
        single_channel.plot.imshow(
            origin="upper", ax=ax, cmap=cmap, **kwargs
        )
    
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