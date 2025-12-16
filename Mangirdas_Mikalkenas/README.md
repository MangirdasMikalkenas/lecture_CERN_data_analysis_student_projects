Climate Observation & Regional Earth eXplorer (COREX)
====================================================

Overview
--------
CORE is a Python-based GUI application for exploring and filtering climate and weather datasets. It allows users to:

- Select multiple weather variables from ERA5 datasets.
- Apply custom min/max thresholds for each variable.
- Define geographic regions using North/South/East/West coordinates.
- Choose time ranges, daily or seasonal aggregation, and hourly sampling.
- Apply pre-configured (or not) presets for variables and geographic areas.
- Visualize valid coordinates on a map and export filtered data.

The application features a modern dark-themed interface and optional background music.

Features
--------
- Variable Selection: Multi-select weather variables with individual min/max limits.
- Presets: Load variable/geo presets for quick selection.
- Geographic Filtering: Specify regions with NSEW bounds or use saved geo presets.
- Time Configuration:
  - Daily or seasonal aggregation
  - Every-Y-hours sampling
  - Full date and time range selection
- Interactive Map: Displays coordinates that match filter criteria.
- Background Music: Plays a looping soundtrack while using the GUI.

Installation
------------
Requirements:
- Python 3.11+
- Python packages:

    pip install PySide6 pygame xarray numpy matplotlib cartopy pathlib mpl_toolkits.axes_grid1 \
    dash[array] netcdf cdsapi zipfile tempfile map copy dask
Files:
- main.py           — application entry point
- gui.py            — GUI layout and logic
- map.py            — map visualization and variable filtering
- data_request_filter.py — ERA5 data filtering logic
- presets/          — default geo and weather presets
- cyberpunk.mp3     — background music
- logo.jpg          — application logo

Usage
-----
1. Run the application:

    python main.py

2. The main window will appear with:
    - Logo
    - Weather variable selection
    - Presets
    - NSEW coordinate inputs
    - Time and aggregation settings
    - Music controls

3. Configure your filters:
    - Select desired variables or apply presets.
    - Adjust min/max thresholds in the pop-up editor.
    - Choose a geographic region or preset.
    - Set time range, mode, and hourly stride if required.

4. Click RUN to process the ERA5 dataset with the selected filters.

5. Visualize results:
    - Matching coordinates are displayed on an interactive map.
    - Optionally save or export filtered results.

Presets
-------
- Stored in `presets/weather/` and `presets/geo/` as `.cfg` files.
- Presets allow quick selection of frequently used configurations.

Output
------
- Stored in `results/`

Notes
-----
- Ensure ERA5 NetCDF files are accessible for filtering.
- Music playback uses pygame; ensure audio hardware is available.
- Ensure you have configured your own `cdsapi.Client()` : https://cds.climate.copernicus.eu/how-to-api
- Designed for desktop environments; may not scale well on screens drastically different from 1920x1080.


