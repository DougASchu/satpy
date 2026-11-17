#!/usr/bin/env python3
# Copyright (C) 2019-2026 Space Science and Engineering Center (SSEC),
#   University of Wisconsin-Madison.
#
# This file is part of the polar2grid software package.
"""Reader for VIIRS NetCDF Flood Granules."""

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
    """File handler for VIIRS NetCDF Flood granules."""

    # Uniform X width (> max single-granule width) so Satpy can concatenate
    # heterogeneous cropped flood granules along 'y' into one pass swath.
    MAX_SWATH_WIDTH = 26000

    def __init__(self, filename, filename_info, filetype_info):
        super().__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename, decode_times=False, mask_and_scale=False)
        self._cached_grid = None

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

    def _get_padded_grid(self):
        """Build North-to-South, pixel-center-aligned 2D lat/lon grids with fill pixels masked."""
        if self._cached_grid is not None:
            return self._cached_grid

        if "lat" not in self.nc or "lon" not in self.nc:
            raise NotImplementedError("Granule lacks 1D lat/lon dimensions.")

        lats = self.nc["lat"].values.astype(np.float64)
        lons = self.nc["lon"].values.astype(np.float64)
        height, width = len(lats), len(lons)

        wd_attrs = self.nc["WaterDetection"].attrs if "WaterDetection" in self.nc else {}
        nom_res = float(wd_attrs.get("ProjectionResolution", getattr(self.nc, "Resolution", 0.003372)))

        lat_res = abs(float(lats[-1] - lats[0])) / (height - 1) if height > 1 else nom_res
        lon_res = abs(float(lons[-1] - lons[0])) / (width - 1) if width > 1 else nom_res

        # Enforce North-to-South row ordering
        flip_y = height > 1 and lats[0] < lats[-1]
        if flip_y:
            lats = lats[::-1]

        # Shift outer pixel-corner coordinates to pixel centers
        min_lon_edge = wd_attrs.get("ProjectionMinLongitude", getattr(self.nc, "geospatial_lon_min", None))
        max_lat_edge = wd_attrs.get("ProjectionMaxLatitude", getattr(self.nc, "geospatial_lat_max", None))

        if min_lon_edge is not None and abs(lons[0] - float(min_lon_edge)) < 0.25 * lon_res:
            lons = lons + 0.5 * lon_res
        if max_lat_edge is not None and abs(lats[0] - float(max_lat_edge)) < 0.25 * lat_res:
            lats = lats - 0.5 * lat_res

        lons_2d, lats_2d = np.meshgrid(lons, lats)

        # Mask coordinates to NaN wherever WaterDetection == _FillValue (1 = no_valid_data)
        # so empty bounding-box corners and off-nadir sub-pixel gaps are ignored by the KD-tree.
        if "WaterDetection" in self.nc:
            wd_raw = self.nc["WaterDetection"]
            wd_vals = wd_raw.values
            wd_fill = int(wd_raw.attrs.get("_FillValue", 1))
            if flip_y:
                wd_vals = np.flip(wd_vals, axis=0)
            invalid_mask = wd_vals == wd_fill
            if not np.all(invalid_mask):
                lons_2d[invalid_mask] = np.nan
                lats_2d[invalid_mask] = np.nan

        # Pad X with NaNs so pyresample ignores padded cells while Satpy stacks granules along 'y'
        target_width = max(self.MAX_SWATH_WIDTH, width)
        if width < target_width:
            pad_cols = target_width - width
            lons_2d = np.pad(lons_2d, ((0, 0), (0, pad_cols)), constant_values=np.nan)
            lats_2d = np.pad(lats_2d, ((0, 0), (0, pad_cols)), constant_values=np.nan)

        self._cached_grid = (lats_2d, lons_2d, flip_y, target_width)
        return self._cached_grid

    def get_area_def(self, key):
        """Construct SwathDefinition matching the padded 2D pixel-center coordinates."""
        lats_2d, lons_2d, _, _ = self._get_padded_grid()
        height, width = lats_2d.shape

        lons_da = xr.DataArray(lons_2d, dims=("y", "x"))
        lats_da = xr.DataArray(lats_2d, dims=("y", "x"))

        swath_def = SwathDefinition(lons=lons_da, lats=lats_da)

        granule_id = self.filename_info.get("start_time", "granule")
        area_name = f"viirs_flood_{granule_id}"
        swath_def.name = area_name
        swath_def.area_id = area_name
        swath_def.height = height
        swath_def.width = width
        swath_def.crs_wkt = CRS.from_epsg(4326).to_wkt()

        return swath_def

    def get_dataset(self, key, info):
        """Extract granule dataset or coordinates as float32 NaN-masked 2D DataArrays."""
        dataset_name = key["name"]
        lats_2d, lons_2d, flip_y, target_width = self._get_padded_grid()

        # 1. Coordinate Queries
        if dataset_name in ["longitude", "latitude", "lon", "lat"]:
            arr = lats_2d if dataset_name in ["latitude", "lat"] else lons_2d
            clean_coords = xr.DataArray(arr, dims=("y", "x"))
            clean_coords.attrs.update(info)
            return clean_coords

        # 2. Data Queries
        if dataset_name not in self.nc:
            logger.warning(f"Dataset {dataset_name} not found in file {self.filename}")
            return None

        raw_data = self.nc[dataset_name].load()
        dims = tuple("y" if d == "lat" else "x" if d == "lon" else d for d in raw_data.dims)
        arr = raw_data.values.astype(np.float32)

        # Convert native _FillValue (1 for WaterDetection, 255 for QualityFlag) to NaN
        # so awips_tiled masks blank space cleanly and drops empty tiles.
        orig_fill = raw_data.attrs.get("_FillValue")
        if orig_fill is not None:
            arr = np.where(arr == float(orig_fill), np.nan, arr)

        # Flip Y if latitudes were South-to-North
        if flip_y and "y" in dims:
            arr = np.flip(arr, axis=dims.index("y"))

        # Pad X with NaN to match target_width
        if "x" in dims:
            x_axis = dims.index("x")
            cur_width = arr.shape[x_axis]
            if cur_width < target_width:
                pad_spec = [(0, 0)] * arr.ndim
                pad_spec[x_axis] = (0, target_width - cur_width)
                arr = np.pad(arr, pad_spec, constant_values=np.nan)

        clean_data = xr.DataArray(arr, dims=dims, name=raw_data.name)
        clean_data.encoding.clear()

        toxic_keys = {"coordinates", "grid_mapping", "add_offset", "scale_factor", "_FillValue", "_Unsigned"}
        for k, v in raw_data.attrs.items():
            if k not in toxic_keys:
                clean_data.attrs[k] = v

        clean_data.attrs.setdefault("standard_name", "water_detection")
        clean_data.attrs.setdefault("units", "1")
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
        self._cached_grid = None
        if hasattr(self, "nc") and self.nc is not None:
            self.nc.close()
        super().close()
