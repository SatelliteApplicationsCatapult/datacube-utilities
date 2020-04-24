import odc.algo
from datacube_utilities.masking import mask_good_quality


def geomedian(ds, product, all_measurements):
    good_quality = mask_good_quality(ds, product)

    xx_data = ds[all_measurements]
    xx_clean = odc.algo.keep_good_only(xx_data, where=good_quality)

    scale, offset = (
        1 / 10_000,
        0,
    )  # differs per product, aim for 0-1 values in float32

    xx_clean = odc.algo.to_f32(xx_clean, scale=scale, offset=offset)
    yy = odc.algo.xr_geomedian(
        xx_clean,
        num_threads=1,  # disable internal threading, dask will run several concurrently
        eps=0.2 * scale,  # 1/5 pixel value resolution
        nocheck=True,  # disable some checks inside geomedian library that use too much ram
    )

    yy = odc.algo.from_float(
        yy, dtype="int16", nodata=-9999, scale=1 / scale, offset=-offset / scale
    )
    return yy
