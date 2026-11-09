#!/usr/bin/env python3
# Copyright (C) 2019-2026 Space Science and Engineering Center (SSEC),
#   University of Wisconsin-Madison.
#
# This file is part of the polar2grid software package.
"""Reader for VIIRS NetCDF Flood products (Standard Granules & Regional Composites)."""

from __future__ import annotations

import logging
from datetime import datetime
import numpy as np
import xarray as xr
from pyproj import CRS
from pyresample.geometry import SwathDefinition
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class VIIRSEDRFlood(BaseFileHandler):
    """File handler for VIIRS Level-3 NetCDF Flood products."""

    def __init__(self, filename, filename_info, filetype_info):
        super().__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename, decode_times=False)

    def _parse_datetime(self, key):
        val = self.filename_info.get(key)
        if isinstance(val, str):
            return datetime.strptime(val[:14], "%Y%m%d%H%M%S")
        return val

    @property
    def start_time(self):
        return self._parse_datetime("start_time")

    @property
    def end_time(self):
        return self._parse_datetime("end_time")

    def get_area_def(self, key):
        """Construct SwathDefinition so Polar2Grid computes exact resampling geometry."""
        if "lat" not in self.nc or "lon" not in self.nc:
            raise NotImplementedError("File lacks lat/lon dimensions.")

        lats = self.nc["lat"].values
        lons = self.nc["lon"].values
        height, width = len(lats), len(lons)

        if lats.ndim == 1 and lons.ndim == 1:
            lons_2d, lats_2d = np.meshgrid(lons, lats)
        else:
            lats_2d, lons_2d = lats, lons

        lons_da = xr.DataArray(lons_2d, dims=("y", "x"))
        lats_da = xr.DataArray(lats_2d, dims=("y", "x"))

        swath_def = SwathDefinition(lons=lons_da, lats=lats_da)

        aoi_str = self.filename_info.get("aoi", "tile")
        area_name = f"viirs_flood_{aoi_str}"
        swath_def.name = area_name
        swath_def.area_id = area_name
        swath_def.height = height
        swath_def.width = width
        swath_def.crs_wkt = CRS.from_epsg(4326).to_wkt()

        return swath_def

    def get_dataset(self, key, info):
        """Extract dataset and forcefully inject AWIPS-compliant CF Grid Mapping."""
        dataset_name = key["name"]

        # 1. Handle Coordinate Queries
        if dataset_name in ["longitude", "latitude", "lon", "lat"]:
            if "lat" in self.nc and "lon" in self.nc:
                lats = self.nc["lat"].values
                lons = self.nc["lon"].values

                if lats.ndim == 1 and lons.ndim == 1:
                    lons_2d, lats_2d = np.meshgrid(lons, lats)
                    arr = lats_2d if dataset_name in ["latitude", "lat"] else lons_2d
                else:
                    arr = self.nc["lat"].values if dataset_name in ["latitude", "lat"] else self.nc["lon"].values

                clean_coords = xr.DataArray(arr, dims=("y", "x"))
                clean_coords.attrs.update(info)
                return clean_coords

        # 2. Handle Data Queries
        if dataset_name not in self.nc:
            logger.warning(f"Dataset {dataset_name} not found in file {self.filename}")
            return None

        raw_data = self.nc[dataset_name].load()
        dims = tuple("y" if d == "lat" else "x" if d == "lon" else d for d in raw_data.dims)

        # Pull raw values to ensure a naked array without legacy 'lambert_projection' pollution
        clean_data = xr.DataArray(raw_data.values, dims=dims, name=raw_data.name)

        # Selectively copy good attributes back, skipping toxic packing/mapping ones[cite: 1]
        toxic_keys = {'coordinates', 'grid_mapping', 'add_offset', 'scale_factor', '_FillValue', '_Unsigned'}
        for k, v in raw_data.attrs.items():
            if k not in toxic_keys:
                clean_data.attrs[k] = v
                
        # Invert the Y-axis if latitudes are ascending (South-to-North)
        if "lat" in self.nc:
            lats_raw = self.nc["lat"].values
            if len(lats_raw) > 1 and lats_raw[0] < lats_raw[-1]:
                if "y" in clean_data.dims:
                    clean_data = clean_data.isel(y=slice(None, None, -1))

        # =====================================================================
        # THE FIX: Inject a complete CF-compliant Grid Mapping variable so AWIPS 
        # fallback parser explicitly knows the latitude_of_projection_origin.
        # =====================================================================
        grid_mapping_attrs = {
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": 25.0,
            "longitude_of_central_meridian": -95.0,
            "latitude_of_projection_origin": 25.0,  # <-- The missing anchor!
            "false_easting": 0.0,
            "false_northing": 0.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245179,
            "inverse_flattening": 298.257223563,
            "short_name": "lcc_conus_300"
        }
        
        # Attach it to the coordinates dictionary so xarray saves it to NetCDF
        lambert_proj = xr.DataArray(np.int32(0), attrs=grid_mapping_attrs)
        clean_data = clean_data.assign_coords(lambert_projection=lambert_proj)
        clean_data.attrs["grid_mapping"] = "lambert_projection"
        
        # Add standard fields required by AWIPS templates
        if "standard_name" not in clean_data.attrs:
            clean_data.attrs["standard_name"] = "water_detection"
        if "units" not in clean_data.attrs:
            clean_data.attrs["units"] = "1"

        clean_data.attrs["area"] = self.get_area_def(key)
        clean_data.attrs.update(info)
        clean_data.attrs["start_time"] = self.start_time
        clean_data.attrs["end_time"] = self.end_time

        platform = (
            self.filename_info.get("platform_shortname")
            or self.filename_info.get("platform_name")
            or getattr(self.nc, "platform", "n20")
        )
        platform_map = {"n20": "noaa20", "n21": "noaa21", "j01": "noaa20", "j02": "noaa21", "npp": "snpp"}
        clean_data.attrs["platform_name"] = platform_map.get(str(platform).lower(), str(platform))
        clean_data.attrs["sensor"] = "viirs"

        return clean_data

    def close(self):
        """Close netCDF dataset handle."""
        if hasattr(self, "nc") and self.nc is not None:
            self.nc.close()
        super().close()
