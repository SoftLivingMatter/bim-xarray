from xarray import DataArray
import napari
from aicsimageio.dimensions import DimensionNames

def napari_view(dataarray: DataArray, *, viewer: napari.Viewer = None):
    if DimensionNames.Channel in dataarray.dims:
        channel_axis = dataarray.dims.index(DimensionNames.Channel)
    else:
        channel_axis = None
    
    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(dataarray, channel_axis=channel_axis)
    
    return viewer