import re
import xarray as xr
from pyproj import Proj, transform
from datacube_utilities.createAOI import create_lat_lon


def create_base_query(aoi, res, output_projection, aoi_crs, dask_chunks):
    lat_extents, lon_extents = create_lat_lon(aoi)
    inProj = Proj("+init=EPSG:4326")
    outProj = Proj("+init=EPSG:3460")

    min_lat, max_lat = lat_extents
    min_lon, max_lon = lon_extents

    x_A, y_A = transform(inProj, outProj, min_lon, min_lat)
    x_B, y_B = transform(inProj, outProj, max_lon, max_lat)

    lat_range = (y_A, y_B)
    lon_range = (x_A, x_B)

    resolution = (-res, res)

    query = {
        "y": lat_range,
        "x": lon_range,
        "output_crs": output_projection,
        "resolution": resolution,
        "dask_chunks": dask_chunks,
        "crs": aoi_crs,
    }
    return query


def create_product_measurement(platform, all_measurements):

    if platform in ["SENTINEL_2"]:
        product = "s2_esa_sr_granule"
        measurements = all_measurements + ["coastal_aerosol", "scene_classification"]
        # Change with S2 WOFS ready
        water_product = "SENTINEL_2_PRODUCT DEFS"
    else:
        product_match = re.search("LANDSAT_(\d)", platform)
        if product_match:
            product = f"ls{product_match.group(1)}_usgs_sr_scene"
            measurements = all_measurements + ["pixel_qa"]
            water_product = f"ls{product_match.group(1)}_water_classification"
        else:
            raise Exception(f"invalid platform_name {platform}")

    return product, measurements, water_product


def is_dataset_empty(ds: xr.Dataset) -> bool:
    checks_for_empty = [
        lambda x: len(x.dims) == 0,  # Dataset has no dimensions
        lambda x: len(x.data_vars) == 0,  # Dataset no variables
    ]
    for f in checks_for_empty:
        if f(ds):
            return True
    return False
