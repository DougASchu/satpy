"""Reader for VIIRS NetCDF Flood Granules."""

from __future__ import annotations

import logging
from datetime import datetime
import dask.array as da
import numpy as np
import xarray as xr
from pyproj import CRS
from pyresample.geometry import SwathDefinition
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()


class VIIRSEDRFlood(BaseFileHandler):
    """File handler for VIIRS NetCDF Flood granules."""

    # Tracks the maximum 'lon' width seen across instantiated granules per platform
    # so xarray.concat(..., dim='y') aligns CONUS, Alaska, and trans-polar passes dynamically.
    _max_width_by_platform: dict[str, int] = {}

    def __init__(self, filename, filename_info, filetype_info):
        super().__init__(filename, filename_info, filetype_info)
        # Open lazily with Dask chunking along 'lat' and single-chunk 'lon' (-1)
        # matching Satpy's native VIIRS readers for fast multi-worker streaming.
        self.nc = xr.open_dataset(
            self.filename,
            decode_times=False,
            mask_and_scale=False,
            chunks={"lat": CHUNK_SIZE, "lon": -1},
        )
        self._cached_grid = None

        platform = str(
            self.filename_info.get("platform_shortname")
            or self.filename_info.get("platform_name")
            or getattr(self.nc, "platform", "default")
        ).lower()
        self._platform_key = platform

        if "lon" in self.nc.dims:
            width = int(self.nc.dims["lon"])
            prev_max = self._max_width_by_platform.get(self._platform_key, 0)
            if width > prev_max:
                self._max_width_by_platform[self._platform_key] = width

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
        """Build lazy Dask 2D float32 lat/lon grids with pixel-center alignment and fill masking."""
        if self._cached_grid is not None:
            return self._cached_grid

        if "lat" not in self.nc or "lon" not in self.nc:
            raise NotImplementedError("Granule lacks 1D lat/lon dimensions.")

        # 1D coordinate vectors are tiny (~5k-25k floats); compute exact shifts in float64
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

        # Wrap 0..360 longitudes (e.g. 180..251.9 over Alaska) into [-180, 180)
        lons = ((lons + 180.0) % 360.0) - 180.0

        # Downcast 1D vectors to float32 before 2D broadcasting (halves coordinate RAM)
        lats_f32 = np.ascontiguousarray(lats, dtype=np.float32)
        lons_f32 = np.ascontiguousarray(lons, dtype=np.float32)
        nan_f32 = np.float32(np.nan)

        if "WaterDetection" in self.nc:
            wd_raw = self.nc["WaterDetection"]
            wd_data = wd_raw.data
            wd_fill = int(wd_raw.attrs.get("_FillValue", 1))
            if flip_y:
                wd_data = wd_data[::-1, :]

            row_chunks, col_chunks = wd_data.chunks
            lats_dask = da.from_array(lats_f32, chunks=(row_chunks,))
            lons_dask = da.from_array(lons_f32, chunks=(col_chunks,))
            lons_2d, lats_2d = da.meshgrid(lons_dask, lats_dask)

            invalid_mask = wd_data == wd_fill
            lons_2d = da.where(invalid_mask, nan_f32, lons_2d)
            lats_2d = da.where(invalid_mask, nan_f32, lats_2d)
        else:
            lats_dask = da.from_array(lats_f32, chunks=CHUNK_SIZE)
            lons_dask = da.from_array(lons_f32, chunks=-1)
            lons_2d, lats_2d = da.meshgrid(lons_dask, lats_dask)

        # Pad X lazily with NaNs to the widest granule in this pass and keep X single-chunked
        target_width = max(self._max_width_by_platform.get(self._platform_key, width), width)
        if width < target_width:
            pad_cols = target_width - width
            lons_2d = da.pad(lons_2d, ((0, 0), (0, pad_cols)), mode="constant", constant_values=nan_f32)
            lats_2d = da.pad(lats_2d, ((0, 0), (0, pad_cols)), mode="constant", constant_values=nan_f32)
        lons_2d = lons_2d.rechunk({1: -1})
        lats_2d = lats_2d.rechunk({1: -1})

        self._cached_grid = (lats_2d, lons_2d, flip_y, target_width)
        return self._cached_grid

    def get_area_def(self, key):
        """Construct SwathDefinition backed by lazy Dask 2D coordinate arrays."""
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
        """Extract granule dataset or coordinates as lazy Dask float32 DataArrays."""
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

        raw_data = self.nc[dataset_name]
        dims = tuple("y" if d == "lat" else "x" if d == "lon" else d for d in raw_data.dims)
        arr = raw_data.data.astype(np.float32)
        nan_f32 = np.float32(np.nan)

        # Convert native _FillValue (1 for WaterDetection, 255 for QualityFlag) to NaN
        orig_fill = raw_data.attrs.get("_FillValue")
        if orig_fill is not None:
            arr = da.where(arr == np.float32(orig_fill), nan_f32, arr)

        # Flip Y lazily if latitudes were South-to-North
        if flip_y and "y" in dims:
            y_axis = dims.index("y")
            slc = [slice(None)] * arr.ndim
            slc[y_axis] = slice(None, None, -1)
            arr = arr[tuple(slc)]

        # Pad X lazily with NaN to match target_width and keep X single-chunked
        if "x" in dims:
            x_axis = dims.index("x")
            cur_width = arr.shape[x_axis]
            if cur_width < target_width:
                pad_spec = [(0, 0)] * arr.ndim
                pad_spec[x_axis] = (0, target_width - cur_width)
                arr = da.pad(arr, pad_spec, mode="constant", constant_values=nan_f32)
            arr = arr.rechunk({x_axis: -1})

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
