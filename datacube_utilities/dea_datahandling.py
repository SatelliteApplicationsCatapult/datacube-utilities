"""
Description: This file contains a set of python functions for handling Digital Earth Australia data.

License: The code in this notebook is licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0). Digital Earth Australia data is licensed under the
Creative Commons by Attribution 4.0 license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data Cube Slack channel
(http://slack.opendatacube.org/) or on the GIS Stack Exchange
(https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag
(you can view previously asked questions here:
https://gis.stackexchange.com/questions/tagged/open-data-cube).

If you would like to report an issue with this script, you can file one on Github
(https://github.com/GeoscienceAustralia/dea-notebooks/issues/new).

Last modified: February 2020
"""

import numpy as np
import xarray as xr
from copy import deepcopy
from datacube.storage import masking

import warnings


def load_ard(
    dc,
    products=None,
    min_gooddata=0.0,
    fmask_gooddata=[1, 4, 5],
    mask_pixel_quality=True,
    mask_invalid_data=True,
    mask_contiguity="nbart_contiguity",
    mask_dtype=np.float32,
    ls7_slc_off=True,
    product_metadata=False,
    **dcload_kwargs,
):
    """
    Loads Landsat Collection 3 or Sentinel 2 Definitive and Near Real
    Time data for multiple sensors (i.e. ls5t, ls7e and ls8c for
    Landsat; s2a and s2b for Sentinel 2), and returns a single masked
    xarray dataset containing only observations that contain greater
    than a given proportion of good quality pixels. This can be used
    to extract clean time series of observations that are not affected
    by cloud, for example as an input to the `animated_timeseries`
    function from `dea_plotting`.

    The proportion of good quality pixels is calculated by summing the
    pixels flagged as good quality in `fmask`. By default non-cloudy or
    shadowed land, snow and water pixels are treated as good quality,
    but this can be customised using the `fmask_gooddata` parameter.

    Last modified: February 2020

    Parameters
    ----------
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.
    products : list
        A list of product names to load data from. Valid options are
        ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'] for Landsat,
        ['s2a_ard_granule', 's2b_ard_granule'] for Sentinel 2 Definitive,
        and ['s2a_nrt_granule', 's2b_nrt_granule'] for Sentinel 2 Near
        Real Time (on the DEA Sandbox only).
    min_gooddata : float, optional
        An optional float giving the minimum percentage of good quality
        pixels required for a satellite observation to be loaded.
        Defaults to 0.0 which will return all observations regardless of
        pixel quality (set to e.g. 0.99 to return only observations with
        more than 99% good quality pixels).
    fmask_gooddata : list, optional
        An optional list of fmask values to treat as good quality
        observations in the above `min_gooddata` calculation. The
        default is `[1, 4, 5]` which will return non-cloudy or shadowed
        land, snow and water pixels. Choose from:
        `{'0': 'nodata', '1': 'valid', '2': 'cloud',
          '3': 'shadow', '4': 'snow', '5': 'water'}`.
    mask_pixel_quality : bool, optional
        An optional boolean indicating whether to apply the good data
        mask to all observations that were not filtered out for having
        less good quality pixels than `min_gooddata`. E.g. if
        `min_gooddata=0.99`, the filtered observations may still contain
        up to 1% poor quality pixels. The default of False simply
        returns the resulting observations without masking out these
        pixels; True masks them and sets them to NaN using the good data
        mask. This will convert numeric values to floating point values
        which can cause memory issues, set to False to prevent this.
    mask_invalid_data : bool, optional
        An optional boolean indicating whether invalid -999 nodata
        values should be replaced with NaN. These invalid values can be
        caused by missing data along the edges of scenes, or terrain
        effects (for NBART). Be aware that masking out invalid values
        will convert all numeric values to floating point values when
        -999 values are replaced with NaN, which can cause memory issues.
    mask_contiguity : str or bool, optional
        An optional string or boolean indicating whether to mask out
        pixels missing data in any band (i.e. "non-contiguous" values).
        Although most missing data issues are resolved by
        `mask_invalid_data`, this step is important for generating
        clean and concistent composite datasets. The default
        is `mask_contiguity='nbart_contiguity'` which will set any
        pixels with non-contiguous values to NaN based on NBART data.
        If you are loading NBAR data instead, you should specify
        `mask_contiguity='nbar_contiguity'` instead. To ignore non-
        contiguous values completely, set `mask_contiguity=False`.
        Be aware that masking out non-contiguous values will convert
        all numeric values to floating point values when -999 values
        are replaced with NaN, which can cause memory issues.
    mask_dtype : numpy dtype, optional
        An optional parameter that controls the data type/dtype that
        layers are coerced to when when `mask_pixel_quality=True` or
        `mask_contiguity=True`. Defaults to `np.float32`, which uses
        approximately 1/2 the memory of `np.float64`.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to
        True, which keeps all Landsat 7 observations > May 31 2003.
    product_metadata : bool, optional
        An optional boolean indicating whether to return the dataset
        with a `product` variable that gives the name of the product
        that each observation in the time series came from (e.g.
        'ga_ls5t_ard_3'). Defaults to False.
    **dcload_kwargs :
        A set of keyword arguments to `dc.load` that define the
        spatiotemporal query used to extract data. This typically
        includes `measurements`, `x`, `y`, `time`, `resolution`,
        `resampling`, `group_by` and `crs`. Keyword arguments can
        either be listed directly in the `load_ard` call like any
        other parameter (e.g. `measurements=['nbart_red']`), or by
        passing in a query kwarg dictionary (e.g. `**query`). For a
        list of possible options, see the `dc.load` documentation:
        https://datacube-core.readthedocs.io/en/latest/dev/api/generate/datacube.Datacube.load.html

    Returns
    -------
    combined_ds : xarray Dataset
        An xarray dataset containing only satellite observations that
        contains greater than `min_gooddata` proportion of good quality
        pixels.

    """

    # Due to possible bug in xarray 0.13.0, define temporary function
    # which converts dtypes in a way that preserves attributes
    def astype_attrs(da, dtype=np.float32):
        """
        Loop through all data variables in the dataset, record
        attributes, convert to a custom dtype, then reassign attributes.
        If the data variable cannot be converted to the custom dtype
        (e.g. trying to convert non-numeric dtype like strings to
        floats), skip and return the variable unchanged.

        This can be combined with `.where()` to save memory. By casting
        to e.g. np.float32, we prevent `.where()` from automatically
        casting to np.float64, using 2x the memory. np.float16 could be
        used to save even more memory (although this may not be
        compatible with all downstream applications).

        This custom function is required instead of using xarray's
        built-in `.astype()`, due to a bug in xarray 0.13.0 that drops
        attributes: https://github.com/pydata/xarray/issues/3348
        """

        try:
            da_attr = da.attrs
            da = da.astype(dtype)
            da = da.assign_attrs(**da_attr)
            return da

        except ValueError:
            return da

    dcload_kwargs = deepcopy(dcload_kwargs)

    # Determine if lazy loading is required
    lazy_load = "dask_chunks" in dcload_kwargs

    # Warn user if they combine lazy load with min_gooddata
    if (min_gooddata > 0.0) & lazy_load:
        warnings.warn(
            "Setting 'min_gooddata' percentage to > 0.0 "
            "will cause dask arrays to compute when "
            "loading pixel-quality data to calculate "
            "'good pixel' percentage. This can "
            "significantly slow the return of your dataset."
        )

    # Verify that products were provided, and that only Sentinel-2 or
    # only Landsat products are being loaded at the same time
    if not products:
        raise ValueError(
            "Please provide a list of product names "
            "to load data from. Valid options are: \n"
            "['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'] "
            "for Landsat, ['s2a_ard_granule', "
            "'s2b_ard_granule'] \nfor Sentinel 2 Definitive, or "
            "['s2a_nrt_granule', 's2b_nrt_granule'] for "
            "Sentinel 2 Near Real Time"
        )
    elif all(["ls" in product for product in products]):
        pass
    elif all(["s2" in product for product in products]):
        pass
    else:
        raise ValueError(
            "Loading both Sentinel-2 and Landsat data "
            "at the same time is currently not supported"
        )

    # Create a list to hold data for each product
    product_data = []

    # Iterate through each requested product
    for product in products:

        try:

            # Load data including fmask band
            print(f"Loading {product} data")
            try:

                # If dask_chunks is specified, load data using query
                if lazy_load:
                    ds = dc.load(product=f"{product}", **dcload_kwargs)

                # If no dask chunks specified, add this param so that
                # we can lazy load data before filtering by good data
                else:
                    ds = dc.load(product=f"{product}", dask_chunks={}, **dcload_kwargs)

            except KeyError as e:
                raise ValueError(
                    f"Band {e} does not exist in this product. "
                    f"Verify all requested `measurements` exist "
                    f"in {products}"
                )

            # Keep a record of the original number of observations
            total_obs = len(ds.time)
            print(total_obs)

            # Identify all pixels not affected by cloud/shadow/invalid
            good_quality = ds

            # If any data was returned
            if len(ds.time) > 0:

                # Optionally apply pixel quality mask to observations
                # remaining after the filtering step above to mask out
                # all remaining bad quality pixels
                if mask_pixel_quality:
                    print("    Applying pixel quality/cloud mask")

                    # Change dtype to custom float before masking to
                    # save memory. See `astype_attrs` func docstring
                    # above for details
                    ds = ds.apply(astype_attrs, dtype=mask_dtype, keep_attrs=True)
                    ds = ds.where(good_quality)

                # Optionally filter to replace no data values with nans
                if mask_invalid_data:
                    print("    Applying invalid data mask")

                    # Change dtype to custom float before masking to
                    # save memory. See `astype_attrs` func docstring
                    # above for details
                    ds = ds.apply(astype_attrs, dtype=mask_dtype, keep_attrs=True)
                    ds = masking.mask_invalid_data(ds)

                # If any data was returned, add result to list
                product_data.append(ds)

            # If no data is returned, print status
            else:
                print(f"    No data for {product}")

            # If  AttributeError due to there being no variables in
            # the dataset, skip this product and move on to the next
        except AttributeError:
            print(f"    No data for {product}")
            # If any data was returned above, combine into one xarray
    if len(product_data) > 0:
        # Concatenate results and sort by time
        print(f"Combining and sorting data")
        combined_ds = xr.concat(product_data, dim="time").sortby("time")

        # If `lazy_load` is True, return data as a dask array without
        # actually loading it in
        if lazy_load:
            print(
                f"    Returning {len(combined_ds.time)} observations" " as a dask array"
            )
            return combined_ds
        else:
            print(f"    Returning {len(combined_ds.time)} observations ")
            return combined_ds.compute()

            # If no data was returned:
    else:
        print("No data returned for query")
        return None
