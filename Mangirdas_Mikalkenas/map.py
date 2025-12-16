import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Maps descriptive variable names to ERA5 dataset variable codes
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


def build_request_legend_text(request_obj: dict) -> str:

    # Generates a multi-line string describing the data request,
    # including time range, mode, hours, and variables with min/max.
    #
    # Parameters
    # ----------
    # request_obj : dict
    #     Dictionary describing the user's requested data (variables, limits, time, stride, etc.)
    #
    # Returns
    # -------
    # str
    #     Formatted legend text for plotting.

    t = request_obj["time"]
    lines = []

    # ---- Date range ----
    lines.append(
        f"Date range: {t['range']['start']} → {t['range']['end']}"
    )

    # ---- Mode ----
    if t["aggregation"]["type"] == "seasonal":
        lines.append("Mode: Seasonal aggregation")
    elif t["aggregation"]["type"] == "daily":
        lines.append("Mode: Daily")
    elif t["stride"]["hours"] is not None:
        lines.append(f"Mode: Every {t['stride']['hours']} hours")
    else:
        lines.append("Mode: Hourly (continuous)")

    # ---- Hour window ----
    h0 = t["filters"]["hours"]["start"]
    h1 = t["filters"]["hours"]["end"]
    lines.append(f"Hours: {h0:02d}:00–{h1:02d}:00")

    lines.append("")
    lines.append("Variables (min / max):")

    # Add each variable with its min/max limits
    for var, lim in request_obj["variables"].items():
        vmin = lim["min"] if lim["min"] is not None else "—"
        vmax = lim["max"] if lim["max"] is not None else "—"
        lines.append(f"• {var}: {vmin} / {vmax}")

    return "\n".join(lines)

def draw_text_legend(ax, text: str, title: str = "Legend"):

        # Draws a text legend on a Matplotlib axis.
        #
        # Parameters
        # ----------
        # ax : matplotlib.axes.Axes
        #     Axis to draw legend on.
        # text : str
        #     Text content for legend.
        # title : str
        #     Title of the legend.
        #

    ax.axis("off") #hide axis ticks
    ax.set_title(title, loc="left", fontsize=10, pad=6)
    ax.text(
        0.0, 1.0,
        text,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        wrap=True,
        transform=ax.transAxes
    )



def mask_nc_by_limits(nc_file: Path, request_obj: dict) -> np.ndarray:

    # Returns a boolean mask of coordinates where all requested variables
    # are within their limits.

    # Parameters
    # ----------
    # nc_file : Path
    #     Path to a NetCDF file.
    # request_obj : dict
    #     Dictionary with requested variables and their limits.
    #
    # Returns
    # -------
    # np.ndarray
    #     Boolean array of shape (latitude, longitude) indicating valid coordinates.


    with xr.open_dataset(nc_file) as ds:
        # Start with all True mask
        mask = np.ones((len(ds.latitude), len(ds.longitude)), dtype=bool)
        # Check each variable
        for var_name, limits in request_obj["variables"].items():
            code = VARIABLE_MAPPING.get(var_name, var_name)
            if code not in ds:
                continue

            da = ds[code]

            # Reduce extra dimensions to first index if present
            dims_to_index = [d for d in da.dims if d not in ("latitude", "longitude")]
            if dims_to_index:
                da = da.isel({dims_to_index[0]: 0})

            # Create mask for this variable
            var_mask = np.ones_like(mask)
            if "min" in limits:
                var_mask &= da >= limits["min"]
            if "max" in limits:
                var_mask &= da <= limits["max"]

            # Combine with global mask
            mask &= var_mask

        return mask


def plot_nc_valid_coords(
# Plots a map highlighting the coordinates where all variables are within limits.
#
#     Parameters
#     ----------
#     nc_file : str
#         Path to NetCDF file.
#     request_obj : dict
#         Dictionary describing the requested variables and limits.
#     bounds : dict
#         Geographic bounds: {"north": , "south": , "west": , "east": }
#     output_file : str
#         Path to save the figure. If None, displays it interactively.
    nc_file: str,
    request_obj: dict,
    bounds: dict,
    output_file: str = None
):
    mask = np.array(mask_nc_by_limits(Path(nc_file), request_obj))
    # Open dataset to get coordinates
    with xr.open_dataset(nc_file) as ds:
        lat = ds.latitude.values
        lon = ds.longitude.values

    # Select only coordinates within bounds
    lat_sel = (lat >= bounds["south"]) & (lat <= bounds["north"])
    lon_sel = (lon >= bounds["west"]) & (lon <= bounds["east"])

    final_mask = mask[np.ix_(lat_sel, lon_sel)]
    final_lat = lat[lat_sel]
    final_lon = lon[lon_sel]

    # Create figure
    fig, ax_map = plt.subplots(
        figsize=(13, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax_map.set_extent(
        [bounds["west"], bounds["east"], bounds["south"], bounds["north"]],
        crs=ccrs.PlateCarree()
    )
    # Plot valid coordinates
    ax_map.pcolormesh(
        final_lon,
        final_lat,
        final_mask,
        cmap="Greens",
        alpha=0.8
    )

    # Add map features
    ax_map.add_feature(cfeature.BORDERS, edgecolor="black")
    ax_map.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax_map.set_title("Coordinates within requested limits (marked green)")

    # ---- LEGEND AXIS ATTACHED TO MAP BORDER ----
    divider = make_axes_locatable(ax_map)

    ax_legend = divider.append_axes(
        "right",
        size="30%",
        pad=0.05,
        axes_class=plt.Axes
    )

    legend_text = build_request_legend_text(request_obj)
    draw_text_legend(ax_legend, legend_text)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)



