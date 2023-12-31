# bim-xarray
`bim-xarray` is a python package providing a user-friendly and 
coherent interface for day-to-day ***b***iological ***im***age analysis, with 
its focus on microscopy data.

This package mostly wraps around awesome libraries including 
`aicsimageio`, `xarray`, and `napari`.

## Installation
Dependencies are specified in [pyproject.toml](pyproject.toml).

If you use mamba (or conda), simply issue the following:
```bash
mamba create -n bim-xarray
mamba env update -n bim-xarray -f environment.yaml
```
(Replace `bim-xarray` with whatever name desired.)

Note: If you cannot load your image files, check if certain readers
are installed with [aicsimageio](https://github.com/AllenCellModeling/aicsimageio).