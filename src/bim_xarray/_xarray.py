import functools

from xarray import register_dataarray_accessor

from .process import (mask,
                      rescale_intensity,
                      rescale_intensity_by,
                      ensure_signed,
                      remove_background)
from .metadata import (label_channel_axis,
                       attach_physical_pixel_sizes,
                       attach_channel_colors,)
from .plot import plot_by_channels
from .napari import napari_view



def _wrap_pipeable(func):
    """
    Wrap functions while preserving signature and docstrings.

    This is specific for pipepable functions that takes its first
    argument a xarray.DataArray, and also returns one. 

    Make sure the attribute name for underlying DataArray is set up
    the same in the DataArray Accessor class.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self._dataarray, *args, **kwargs)
    return wrapper


class MetadataMixin:
    label_channel_axis = _wrap_pipeable(label_channel_axis)
    attach_physical_pixel_sizes = _wrap_pipeable(attach_physical_pixel_sizes)
    attach_channel_colors = _wrap_pipeable(attach_channel_colors)


class ProcessMixin:
    mask = _wrap_pipeable(mask)
    rescale_intensity = _wrap_pipeable(rescale_intensity)
    rescale_intensity_by = _wrap_pipeable(rescale_intensity_by)
    ensure_signed = _wrap_pipeable(ensure_signed)
    remove_background = _wrap_pipeable(remove_background)


class PlotMixin:
    plot_by_channels = _wrap_pipeable(plot_by_channels)    


class NapariMixin:
    napari_view = _wrap_pipeable(napari_view)


@register_dataarray_accessor('bim')
class BioimageDataArrayAccessor(
    ProcessMixin, 
    MetadataMixin, 
    PlotMixin, 
    NapariMixin,
):

    def __init__(self, dataarray):  # noqa: D107
        self._dataarray = dataarray

    @property
    def unit(self):
        pass