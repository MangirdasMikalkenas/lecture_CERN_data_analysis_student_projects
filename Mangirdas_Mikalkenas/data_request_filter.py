import cdsapi
import zipfile
import xarray as xr
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from map import plot_nc_valid_coords
import copy
import dask

# Configure dask to run single-threaded
dask.config.set(scheduler="single-threaded")


# Maps variable names (same as map.py)
VARIABLE_MAPPING = {
    "10m_u_component_of_wind": "u10",
    "10m_u_component_of_neutral_wind": "u10n",
    "10m_v_component_of_wind": "v10",
    "10m_v_component_of_neutral_wind": "v10n",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "2m_dewpoint_temperature": "d2m",
    "2m_temperature": "t2m",
    "air_density_over_the_oceans": "rhoao",
    "angle_of_sub_gridscale_orography": "anor",
    "anisotropy_of_sub_gridscale_orography": "isor",
    "benjamin_feir_index": "bfi",
    "boundary_layer_height": "blh",
    "charnock": "chnk",
    "cloud_base_height": "cbh",
    "coefficient_of_drag_with_waves": "cdww",
    "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin",
    "convective_precipitation": "cp",
    "convective_rain_rate": "crr",
    "convective_snowfall": "csf",
    "convective_snowfall_rate_water_equivalent": "csfr",
    "duct_base_height": "dctb",
    "eastward_gravity_wave_surface_stress": "lgws",
    "maximum_individual_wave_height": "hmax",
    "lwe_thickness_of_water_evaporation_amount": "e",
    "forecast_albedo": "fal",
    "forecast_logarithm_of_surface_roughness_for_heat": "flsr",
    "forecast_surface_roughness": "fsr",
    "free_convective_velocity_over_the_oceans": "wstar",
    "friction_velocity": "zust",
    "geopotential": "z",
    "gravity_wave_dissipation": "gwd",
    "high_cloud_cover": "hcc",
    "high_vegetation_cover": "cvh",
    "ice_temperature_layer_1": "istl1",
    "ice_temperature_layer_2": "istl2",
    "ice_temperature_layer_3": "istl3",
    "ice_temperature_layer_4": "istl4",
    "instantaneous_10m_wind_gust": "i10fg",
    "instantaneous_eastward_turbulent_surface_stress": "iews",
    "instantaneous_large_scale_surface_precipitation_fraction": "ilspf",
    "instantaneous_moisture_flux": "ie",
    "instantaneous_northward_turbulent_surface_stress": "inss",
    "instantaneous_surface_sensible_heat_flux": "ishf",
    "k_index": "kx",
    "lake_bottom_temperature": "lblt",
    "lake_cover": "cl",
    "lake_ice_temperature": "lict",
    "lake_ice_depth": "licd",
    "lake_mix_layer_depth": "lmld",
    "lake_mix_layer_temperature": "lmlt",
    "lake_shape_factor": "lshf",
    "lake_depth": "dl",
    "lake_total_layer_temperature": "ltlt",
    "land_sea_mask": "lsm",
    "large_scale_precipitation_rate": "lsrr",
    "large_scale_snowfall_rate_water_equivalent": "lssfr",
    "large_scale_precipitation": "lsp",
    "large_scale_precipitation_fraction": "lspf",
    "large_scale_snowfall": "lsf",
    "leaf_area_index_high_vegetation": "lai_hv",
    "leaf_area_index_low_vegetation": "lai_lv",
    "low_cloud_cover": "lcc",
    "low_vegetation_cover": "cvl",
    "10m_wind_gust_since_previous_post_processing": "fg10",
    "maximum_2m_temperature_since_previous_post_processing": "mx2t",
    "maximum_total_precipitation_rate_since_previous_post_processing": "mxtpr",
    "mean_direction_of_total_swell": "mdts",
    "mean_direction_of_wind_waves": "mdww",
    "mean_period_of_total_swell": "mpts",
    "mean_period_of_wind_waves": "mpww",
    "mean_sea_level_pressure": "msl",
    "mean_square_slope_of_waves": "msqs",
    "mean_vertical_gradient_of_refractivity_inside_trapping_layer": "dndza",
    "mean_wave_direction": "mwd",
    "mean_wave_direction_of_first_swell_partition": "mwd1",
    "mean_wave_direction_of_second_swell_partition": "mwd2",
    "mean_wave_direction_of_third_swell_partition": "mwd3",
    "mean_wave_period": "mwp",
    "mean_wave_period_based_on_first_moment": "mp1",
    "mean_wave_period_based_on_first_moment_for_swell": "p1ps",
    "mean_wave_period_based_on_first_moment_for_wind_waves": "p1ww",
    "mean_wave_period_based_on_second_moment_for_swell": "p2ps",
    "mean_wave_period_based_on_second_moment_for_wind_waves": "p2ww",
    "mean_wave_period_of_first_swell_partition": "mwp1",
    "mean_wave_period_of_second_swell_partition": "mwp2",
    "mean_wave_period_of_third_swell_partition": "mwp3",
    "mean_zero_crossing_wave_period": "mp2",
    "medium_cloud_cover": "mcc",
    "minimum_2m_temperature_since_previous_post_processing": "mn2t",
    "minimum_total_precipitation_rate_since_previous_post_processing": "mntpr",
    "model_bathymetry": "wmb",
    "near_ir_albedo_for_diffuse_radiation": "alnid",
    "near_ir_albedo_for_direct_radiation": "alnip",
    "normalized_energy_flux_into_ocean": "phioc",
    "normalized_energy_flux_into_waves": "phiaw",
    "normalized_stress_into_ocean": "tauoc",
    "northward_gravity_wave_surface_stress": "mgws",
    "peak_wave_period": "pp1d",
    "period_corresponding_to_maximum_individual_wave_height": "tmax",
    "potential_evaporation": "pev",
    "precipitation_type": "ptype",
    "runoff": "ro",
    "sea_surface_temperature": "sst",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "significant_height_of_total_swell": "shts",
    "significant_height_of_wind_waves": "shww",
    "significant_wave_height_of_first_swell_partition": "swh1",
    "significant_wave_height_of_second_swell_partition": "swh2",
    "significant_wave_height_of_third_swell_partition": "swh3",
    "skin_reservoir_content": "src",
    "skin_temperature": "skt",
    "slope_of_sub_gridscale_orography": "slor",
    "snow_albedo": "asn",
    "snow_density": "rsn",
    "snow_depth": "sd",
    "snow_evaporation": "es",
    "snowfall": "sf",
    "snowmelt": "smlt",
    "soil_temperature_level_1": "stl1",
    "soil_temperature_level_2": "stl2",
    "soil_temperature_level_3": "stl3",
    "soil_temperature_level_4": "stl4",
    "soil_type": "slt",
    "standard_deviation_of_filtered_subgrid_orography": "sdfor",
    "sub_surface_runoff": "ssro",
    "surface_latent_heat_flux": "slhf",
    "surface_net_short_wave_radiation_flux": "ssr",
    "surface_net_short_wave_radiation_flux_clear_sky": "ssrc",
    "surface_pressure": "sp",
    "surface_runoff": "sro",
    "surface_sensible_heat_flux": "sshf",
    "temperature_of_snow_layer": "tsn",
    "mean_top_downward_short_wave_radiation_flux": "avg_tdswrf",
}


# ================= DOWNLOAD =================

def download_era5_zipped(request_obj: dict, out_dir: Path) -> list[Path]:
    # Downloads ERA5 data as a zipped NetCDF file.
    #
    # Parameters
    # ----------
    # request_obj : dict
    #     Data request dictionary including variables, date range, hours, and area.
    # out_dir : Path
    #     Directory to save the downloaded zip file.
    #
    # Returns
    # -------
    # list[Path]
    #     List containing the path to the downloaded zip file.

    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    variables = list(request_obj["variables"].keys())
    start = request_obj["time"]["range"]["start"]
    end = request_obj["time"]["range"]["end"]

    h0 = request_obj["time"]["filters"]["hours"]["start"]
    h1 = request_obj["time"]["filters"]["hours"]["end"]
    times = [f"{h:02d}:00" for h in range(h0, h1 + 1)]

    zip_file = out_dir / "era5.zip"

    # CDS API request
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": variables,
            "date": f"{start}/{end}",
            "time": times,
            "area": [
                request_obj["geo"]["north"],
                request_obj["geo"]["west"],
                request_obj["geo"]["south"],
                request_obj["geo"]["east"],
            ],
            "data_format": "netcdf",
            "download_format": "zip",
        },
        str(zip_file),
    )

    return [zip_file]


# ================= UNZIP =================
def unzip_files(zip_files, extract_dir):
    # Extracts NetCDF files from zip archives.
    #
    # Parameters
    # ----------
    # zip_files : list
    #     List of zip file paths.
    # extract_dir : Path
    #     Directory to extract files into.
    #
    # Returns
    # -------
    # list[Path]
    #     List of paths to extracted NetCDF files.

    nc_files = []
    for zf in zip_files:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(extract_dir)
            nc_files.extend([extract_dir / f for f in z.namelist() if f.endswith(".nc")])
    return nc_files


# ================= FILTER =================

def filter_nc_files_by_data_masked(nc_files: list[Path], request_obj: dict, output_dir: Path) -> list[Path]:
    # Applies min/max masking to NetCDF files and saves masked copies.
    #
    # Parameters
    # ----------
    # nc_files : list[Path]
    #     List of NetCDF file paths.
    # request_obj : dict
    #     Dictionary with variable limits.
    # output_dir : Path
    #     Directory to save masked NetCDF files.
    #
    # Returns
    # -------
    # list[Path]
    #     List of paths to masked NetCDF files.

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_files = []

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        ds_masked = ds.copy()

        for var_name, limits in request_obj["variables"].items():
            code = VARIABLE_MAPPING.get(var_name, var_name)
            if code not in ds_masked:
                continue

            da = ds_masked[code]

            # Pick first non-spatial dimension if it exists
            dims_to_index = [d for d in da.dims if d not in ("latitude", "longitude")]
            if dims_to_index:
                da = da.isel({dims_to_index[0]: 0})

            da_min = limits.get("min")
            da_max = limits.get("max")

            # Apply mask
            if da_min is not None:
                da = da.where(da >= da_min)
            if da_max is not None:
                da = da.where(da <= da_max)

            ds_masked[code] = da

        out_file = output_dir / nc_file.name
        ds_masked.to_netcdf(out_file)
        filtered_files.append(out_file)
        ds.close()
        ds_masked.close()

    return filtered_files

# ================= MAIN ENTRY (UPDATED) =================
def run_era5_filter_with_mask_maps(request_obj: dict, output_dir: Path = Path("results")) -> list[Path]:
    # High-level workflow:
    # 1. Downloads ERA5 data (seasonal or single download)
    # 2. Masks values outside requested limits
    # 3. Saves yearly or combined NetCDF files
    # 4. Plots valid coordinates map
    #
    # Parameters
    # ----------
    # request_obj : dict
    #     Data request dictionary including variables, limits, time range, and area.
    # output_dir : Path
    #     Directory to save outputs (masked NetCDF + maps).
    #
    # Returns
    # -------
    # list[Path]
    #     List of NetCDF file paths (combined dataset).
    results_dir = output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        time_mode = request_obj["time"]["mode"]
        start_date_str = request_obj["time"]["range"]["start"]
        end_date_str = request_obj["time"]["range"]["end"]

        start_year, start_mmdd = int(start_date_str[:4]), start_date_str[5:]
        end_year, end_mmdd = int(end_date_str[:4]), end_date_str[5:]

        yearly_files = []

        # ================= DOWNLOAD AND MASK PER YEAR =================
        # Download + mask data per year for seasonal requests
        if time_mode == "seasonal":
            for year in range(start_year, end_year + 1):
                year_request = copy.deepcopy(request_obj)
                year_request["time"]["range"] = {
                    "start": f"{year}-{start_mmdd}",
                    "end": f"{year}-{end_mmdd}"
                }

                year_dir = tmp_path / f"year_{year}"
                year_dir.mkdir(parents=True, exist_ok=True)

                zip_files = download_era5_zipped(year_request, year_dir)
                nc_files = unzip_files(zip_files, year_dir / "nc")

                if not nc_files:
                    continue

                # Combine per-year files with Dask chunking
                ds_year = xr.open_mfdataset(
                    nc_files,
                    combine="by_coords",
                    chunks={'time': 10},
                    engine="netcdf4",
                parallel=True
                )

                # Apply masks lazily
                for var_name, limits in year_request["variables"].items():
                    code = VARIABLE_MAPPING.get(var_name, var_name)
                    if code in ds_year:
                        da = ds_year[code]
                        min_val = limits.get("min", -np.inf)
                        max_val = limits.get("max", np.inf)
                        ds_year[code] = da.where((da >= min_val) & (da <= max_val))

                # Save masked yearly dataset
                year_file = results_dir / f"era5_masked_{year}.nc"
                ds_year.to_netcdf(year_file,    engine="netcdf4",
                     encoding={v: {'zlib': True} for v in ds_year.data_vars})
                ds_year.close()
                yearly_files.append(year_file)

        else:
            # Single download mode
            zip_files = download_era5_zipped(request_obj, tmp_path)
            nc_files = unzip_files(zip_files, tmp_path / "nc")
            if not nc_files:
                print("No NetCDF files downloaded.")
                return []

            ds_single = xr.open_mfdataset(
                nc_files,
                combine="by_coords",
                chunks={'time': 10},
                engine="netcdf4",
            parallel=True
            )

            for var_name, limits in request_obj["variables"].items():
                code = VARIABLE_MAPPING.get(var_name, var_name)
                if code in ds_single:
                    da = ds_single[code]
                    min_val = limits.get("min", -np.inf)
                    max_val = limits.get("max", np.inf)
                    ds_single[code] = da.where((da >= min_val) & (da <= max_val))

            combined_file = results_dir / "era5_masked.nc"
            ds_single.to_netcdf(combined_file, engine="netcdf4",
                 encoding={v: {'zlib': True} for v in ds_single.data_vars})
            ds_single.close()
            yearly_files.append(combined_file)

        # ================= COMBINE ALL YEARS =================
        # Combine all yearly files if multiple
        if len(yearly_files) > 1:
            print("Combining yearly datasets...")
            combined_ds = xr.open_mfdataset(
                yearly_files,
                combine="by_coords",
                chunks={'time': 10},
                engine="netcdf4",
            parallel=True
            )

            combined_file = results_dir / "combined.nc"
            combined_ds.to_netcdf(combined_file,     engine="netcdf4",
                encoding={v: {'zlib': True} for v in combined_ds.data_vars})
            combined_ds.close()
        else:
            combined_file = yearly_files[0]

        # ================= PLOT VALID COORDINATES =================
        bounds = request_obj.get("geo", {"north": 60, "south": 50, "west": -10, "east": 10})
        plot_nc_valid_coords(
            nc_file=str(combined_file),
            request_obj=request_obj,
            bounds=bounds,
            output_file=results_dir / "valid_coords_map_combined.png"
        )

        print(f"Results saved in: {results_dir}")
        return [combined_file]



