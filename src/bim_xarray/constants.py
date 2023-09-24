import aicsimageio.constants

METADATA_UNPROCESSED = aicsimageio.constants.METADATA_UNPROCESSED
METADATA_PROCESSED = aicsimageio.constants.METADATA_PROCESSED
METADATA_OME = "ome_metadata_full"
METADATA_OME_SCENE = "ome_metadata"

COORDS_SIZE_SPATIAL = "physical_pixel_sizes"
COORDS_SIZE_T = "time_per_frame"

IMAGE_KIND_INTENSITY = "intensity"
IMAGE_KIND_BINARY_OR_LABEL = "object"
IMAGE_KIND_INTENSITY_SHORT = "i"
IMAGE_KIND_BINARY_OR_LABEL_SHORT = "o"