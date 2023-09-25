from typing import Dict, Optional
import pint

ureg = pint.UnitRegistry()

DEFAULT_UNIT_SPATIAL = 'micrometer'
DEFAULT_UNIT_TIME = 'second'

def get_pixel_size_units(scene_ome_meta) -> Dict[str, Optional[str]]:
    physical_size_units = {}

    pixels = scene_ome_meta.pixels
    try:
        physical_size_units = {
            'X': pixels.physical_size_x_unit.value,
            'Y': pixels.physical_size_y_unit.value,
            'Z': pixels.physical_size_z_unit.value,
        }
    except AttributeError:
        pass
    
    if not physical_size_units and len(pixels.planes) > 0:
        plane = pixels.planes[0]
        try:
            physical_size_units = {
                'X': plane.physical_size_x_unit.value,
                'Y': plane.physical_size_y_unit.value,
                'Z': plane.physical_size_z_unit.value,
            }
        except AttributeError:
            pass
    
    return physical_size_units


def get_time_increment_unit(scene_ome_meta) -> Optional[float]:
    time_increment_unit = None

    pixels = scene_ome_meta.pixels
    try:
        time_increment_unit = pixels.time_increment_unit.value
    except AttributeError:
        pass
    
    return time_increment_unit


def get_delta_t_unit(scene_ome_meta) -> Optional[float]:
    delta_t_unit = None

    planes = scene_ome_meta.pixels.planes
    if len(planes) > 0:
        try:
            delta_t_unit = planes[0].delta_t_unit.value
        except AttributeError:
            pass
    
    return delta_t_unit


def _factor_to_default_unit_spatial(unit) -> Optional[float]:
    factor = None
    if unit is not None:
        factor = float(ureg(unit).to(DEFAULT_UNIT_SPATIAL).magnitude)
    return factor
    

def get_pixel_size_conversion_factors(scene_meta) -> Dict[str, Optional[float]]:
    units = get_pixel_size_units(scene_meta)
    return {
        k: _factor_to_default_unit_spatial(v) 
        for k, v in units.items()
    }


def _factor_to_default_unit_time(unit) -> Optional[float]:
    factor = None
    if unit is not None:
        factor = float(ureg(unit).to(DEFAULT_UNIT_TIME).magnitude)
    return factor


def get_time_increment_conversion_factor(scene_meta) -> Optional[float]:
    unit = get_time_increment_unit(scene_meta)
    return _factor_to_default_unit_time(unit)


def get_delta_t_conversion_factor(scene_meta) -> Optional[float]:
    unit = get_delta_t_unit(scene_meta)
    return _factor_to_default_unit_time(unit)