from PySide6.QtWidgets import (QApplication, QWidget, QPushButton,
                               QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QSlider, QComboBox, QLineEdit,
                               QSpacerItem, QSizePolicy, QScrollArea,
                               QDateEdit, QRadioButton, QButtonGroup,
                               QSpinBox, QCheckBox, QMessageBox,
                               QInputDialog
                               )

from PySide6.QtGui import (QGuiApplication, QPixmap, QStandardItemModel,
                           QStandardItem, QFontMetrics
                           )
from PySide6.QtCore import Qt, QPoint, QTimer
from pathlib import Path
import pygame
from data_request_filter import run_era5_filter_with_mask_maps
music_f = "cyberpunk.mp3"
logo_f = "logo.jpg"
BORDER_COLOR = "#1B1C1D"  # Dark grey
WEATHER_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_u_component_of_neutral_wind",
    "10m_v_component_of_wind",
    "10m_v_component_of_neutral_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "air_density_over_the_oceans",
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "benjamin_feir_index",
    "boundary_layer_height",
    "charnock",
    "cloud_base_height",
    "coefficient_of_drag_with_waves",
    "convective_available_potential_energy",
    "convective_inhibition",
    "convective_precipitation",
    "convective_rain_rate",
    "convective_snowfall",
    "convective_snowfall_rate_water_equivalent",
    "duct_base_height",
    "eastward_gravity_wave_surface_stress",
    "maximum_individual_wave_height",
    "lwe_thickness_of_water_evaporation_amount",
    "forecast_albedo",
    "forecast_logarithm_of_surface_roughness_for_heat",
    "forecast_surface_roughness",
    "free_convective_velocity_over_the_oceans",
    "friction_velocity",
    "geopotential",
    "gravity_wave_dissipation",
    "high_cloud_cover",
    "high_vegetation_cover",
    "ice_temperature_layer_1",
    "ice_temperature_layer_2",
    "ice_temperature_layer_3",
    "ice_temperature_layer_4",
    "instantaneous_10m_wind_gust",
    "instantaneous_eastward_turbulent_surface_stress",
    "instantaneous_large_scale_surface_precipitation_fraction",
    "instantaneous_moisture_flux",
    "instantaneous_northward_turbulent_surface_stress",
    "instantaneous_surface_sensible_heat_flux",
    "k_index",
    "lake_bottom_temperature",
    "lake_cover",
    "lake_ice_temperature",
    "lake_ice_depth",
    "lake_mix_layer_depth",
    "lake_mix_layer_temperature",
    "lake_shape_factor",
    "lake_depth",
    "lake_total_layer_temperature",
    "land_sea_mask",
    "large_scale_precipitation_rate",
    "large_scale_snowfall_rate_water_equivalent",
    "large_scale_precipitation",
    "large_scale_precipitation_fraction",
    "large_scale_snowfall",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "low_cloud_cover",
    "low_vegetation_cover",
    "10m_wind_gust_since_previous_post_processing",
    "maximum_2m_temperature_since_previous_post_processing",
    "maximum_total_precipitation_rate_since_previous_post_processing",
    "mean_direction_of_total_swell",
    "mean_direction_of_wind_waves",
    "mean_period_of_total_swell",
    "mean_period_of_wind_waves",
    "mean_sea_level_pressure",
    "mean_square_slope_of_waves",
    "mean_vertical_gradient_of_refractivity_inside_trapping_layer",
    "mean_wave_direction",
    "mean_wave_direction_of_first_swell_partition",
    "mean_wave_direction_of_second_swell_partition",
    "mean_wave_direction_of_third_swell_partition",
    "mean_wave_period",
    "mean_wave_period_based_on_first_moment",
    "mean_wave_period_based_on_first_moment_for_swell",
    "mean_wave_period_based_on_first_moment_for_wind_waves",
    "mean_wave_period_based_on_second_moment_for_swell",
    "mean_wave_period_based_on_second_moment_for_wind_waves",
    "mean_wave_period_of_first_swell_partition",
    "mean_wave_period_of_second_swell_partition",
    "mean_wave_period_of_third_swell_partition",
    "mean_zero_crossing_wave_period",
    "medium_cloud_cover",
    "minimum_2m_temperature_since_previous_post_processing",
    "minimum_total_precipitation_rate_since_previous_post_processing",
    "model_bathymetry",
    "near_ir_albedo_for_diffuse_radiation",
    "near_ir_albedo_for_direct_radiation",
    "normalized_energy_flux_into_ocean",
    "normalized_energy_flux_into_waves",
    "normalized_stress_into_ocean",
    "northward_gravity_wave_surface_stress",
    "peak_wave_period",
    "period_corresponding_to_maximum_individual_wave_height",
    "potential_evaporation",
    "precipitation_type",
    "runoff",
    "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
    "significant_height_of_total_swell",
    "significant_height_of_wind_waves",
    "significant_wave_height_of_first_swell_partition",
    "significant_wave_height_of_second_swell_partition",
    "significant_wave_height_of_third_swell_partition",
    "skin_reservoir_content",
    "skin_temperature",
    "slope_of_sub_gridscale_orography",
    "snow_albedo",
    "snow_density",
    "snow_depth",
    "snow_evaporation",
    "snowfall",
    "snowmelt",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "sub_surface_runoff",
    "surface_latent_heat_flux",
    "surface_net_short_wave_radiation_flux",
    "surface_net_short_wave_radiation_flux_clear_sky",
    "surface_pressure",
    "surface_runoff",
    "surface_sensible_heat_flux",
    "temperature_of_snow_layer",
    "mean_top_downward_short_wave_radiation_flux"
]

# ---------- Utility ----------

def get_scaled_window_size(width_ratio: float, height_ratio: float) -> tuple[int, int]:
    # Return window size scaled to screen dimensions.
    # width_ratio, height_ratio: fraction of screen width/height

    screen = QGuiApplication.primaryScreen()
    geometry = screen.availableGeometry()
    return (
        int(geometry.width() * width_ratio),
        int(geometry.height() * height_ratio)
    )


# ---------- Custom degree line edit for popup ----------
class PopupDegreeLineEdit(QLineEdit):
    """Numeric input without suffix, for popup min/max"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.allowed_chars = "0123456789+-."
        self.setAlignment(Qt.AlignRight)
        self.setFixedWidth(60)
        # Auto-correct comma to dot
        self.textChanged.connect(self._fix_comma)

    def _fix_comma(self, text):
        # Replace commas with dots in real-time
        if "," in text:
            self.setText(text.replace(",", "."))

    def keyPressEvent(self, event):
        # Allow only numeric input, plus/minus, dot, backspace/delete/arrow keys.
        # Overrides QLineEdit key event.
        key_text = event.text()
        if key_text == ",":
            key_text = "."
        if key_text not in self.allowed_chars and event.key() not in (
            Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right
        ):
            return # reject input
        super().keyPressEvent(event)

# ---------- Popup widget ----------
class WeatherPopup(QWidget):
    # Popup to allow user to set min/max for selected weather variables.
    # Maintains a scrollable list of inputs for each variable.
    def __init__(self, selected_items, minmax_values, parent=None, popup_width=400):
        super().__init__(parent, Qt.Popup)
        self.setStyleSheet("background-color: black;")
        self.minmax_values = minmax_values

        # Scroll area
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none;")
        self.container = QWidget()
        self.scroll.setWidget(self.container)

        self.items_layout = QVBoxLayout(self.container)
        self.items_layout.setContentsMargins(2, 2, 2, 2)
        self.items_layout.setSpacing(2)

        self.item_widgets = {}  # map: item_name -> (min_edit, max_edit)
        self.update_items(selected_items)

        # Popup layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.scroll)

        self.setFixedWidth(popup_width)
        self.adjust_height(selected_items)

    def adjust_height(self, items):
        # Resize popup based on number of items (max 400 px)
        height = min(400, 30 * len(items) + 10)
        self.setFixedHeight(height)
        self.scroll.setFixedHeight(height)

    def update_items(self, selected_items):
        # Add/remove rows for variables as selection changes.
        # Creates min/max line edits for each selected variable.        e

        existing_items = set(self.item_widgets.keys())
        new_items = set(selected_items)

        # Remove items no longer selected
        for item in existing_items - new_items:
            min_edit, max_edit = self.item_widgets[item]
            min_edit.parentWidget().deleteLater()
            del self.item_widgets[item]

        # Add new items
        for item in selected_items:
            if item in self.item_widgets:
                # Widget exists, just update text from minmax_values
                min_edit, max_edit = self.item_widgets[item]
                if item not in self.item_widgets:
                    pass  # create row and set text from minmax_values
                else:
                    # don't overwrite existing edits
                    pass
                continue

            # Create row
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)

            lbl = QLabel(item)
            lbl.setStyleSheet("color: white;")
            lbl.setFixedWidth(250)

            min_edit = PopupDegreeLineEdit()
            min_edit.setPlaceholderText("min")
            max_edit = PopupDegreeLineEdit()
            max_edit.setPlaceholderText("max")

            # Fill from minmax_values if exists
            if item in self.minmax_values:
                min_v, max_v = self.minmax_values[item]
                if min_v is not None:
                    min_edit.setText(str(min_v))
                if max_v is not None:
                    max_edit.setText(str(max_v))

            # Save values on change
            def save_value(name=item, min_e=min_edit, max_e=max_edit):
                try:
                    min_v = float(min_e.text())
                except ValueError:
                    min_v = None
                try:
                    max_v = float(max_e.text())
                except ValueError:
                    max_v = None
                self.minmax_values[name] = (min_v, max_v)

            min_edit.textChanged.connect(save_value)
            max_edit.textChanged.connect(save_value)

            row_layout.addWidget(lbl)
            row_layout.addWidget(min_edit)
            row_layout.addWidget(max_edit)

            self.items_layout.addWidget(row_widget)
            self.item_widgets[item] = (min_edit, max_edit)

        self.adjust_height(selected_items)

# ---------- Custom combo-like button ----------
class SelectedWeatherCombo(QWidget):
    # Custom widget that acts like a combo box with a popup to filter selected weather variables.
    def __init__(self, weather_model: QStandardItemModel):
        super().__init__()
        self.weather_model = weather_model
        self.setFixedHeight(25)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Button to show popup
        self.button = QPushButton("FILTER SELECTED VARIABLES")
        self.button.setStyleSheet("color: white; text-align: center;")
        self.button.clicked.connect(self.show_popup)
        self.layout.addWidget(self.button)

        # Persist min/max values per variable
        self.minmax_values = {}

        # Popup reference
        self.popup = None

        # Initialize min/max for all variables
        for i in range(weather_model.rowCount()):
            name = weather_model.item(i).text()
            self.minmax_values[name] = (None, None)

        # Update popup when check state changes
        self.weather_model.itemChanged.connect(self.update_popup)

    def get_selected_items(self):
        # Return currently checked variables
        return [
            self.weather_model.item(i).text()
            for i in range(self.weather_model.rowCount())
            if self.weather_model.item(i).checkState() == Qt.Checked
        ]

    def update_popup(self):
        # Only update popup if visible
        items = self.get_selected_items()
        if self.popup and self.popup.isVisible():
            self.popup.update_items(items)

    def show_popup(self):
        # Show or refresh the popup widget
        items = self.get_selected_items()
        if not items:
            return

        popup_width = max(self.width(), 400)
        if self.popup is None:
            self.popup = WeatherPopup(items, self.minmax_values, self, popup_width)
        else:
            self.popup.update_items(items)

        pos = self.mapToGlobal(QPoint(0, self.height()))
        self.popup.setGeometry(pos.x(), pos.y(), popup_width, min(400, 30 * len(items) + 10))
        self.popup.show()



# ---------- Container creation (EMPTY) ----------

def create_container(style: str = None) -> tuple[QWidget, QVBoxLayout]:
# Return a QWidget container with QVBoxLayout and optional border style

    container = QWidget()
    if style is None:
        style = f"border: 2px solid {BORDER_COLOR}; padding: 5px;"
    container.setStyleSheet(style)
    layout = QVBoxLayout(container)
    return container, layout

# ---------- Widget creation ----------
def set_combo_width_for_text(combo: QComboBox, sample_text: str, extra_digits: int = 3):
    # Set QComboBox width to fit sample text plus extra space

    fm = QFontMetrics(combo.font())
    width = fm.horizontalAdvance(sample_text + "0" * extra_digits)
    combo.setMinimumWidth(width + 30)  # padding for arrow + margins


# ---------- Reusable button style ----------
BUTTON_STYLE_ROW = f"""
QPushButton {{
    color: white;
    background-color: black;
    border: 2px solid {BORDER_COLOR};
    border-radius: 0px;
}}
QPushButton:hover {{
    background-color: #1B1C1D;
}}
QPushButton:pressed {{
    background-color: #111111;
}}
"""


# Logo implementation

def create_logo(path: Path, height: int) -> QLabel:
    logo = QLabel()
    logo.setPixmap(QPixmap(str(path)))
    logo.setScaledContents(True)
    logo.setFixedHeight(height)
    return logo

# ---------- Path getters ----------

def get_script_directory() -> Path:
    return Path(__file__).parent

PRESETS_DIR = get_script_directory() / "presets"
GEO_DIR = PRESETS_DIR / "geo"
WEATHER_DIR = PRESETS_DIR / "weather"

def get_path(filename) -> Path:
    f_path = get_script_directory() / filename
    if not f_path.exists():
        raise FileNotFoundError(f"File not found: {f_path}")
    return f_path

def load_geo_cfg(file_path: Path) -> dict:
    """Load north/south/east/west values from geo .cfg"""
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=")
                data[key.strip()] = float(val.strip())
    return data

def load_weather_cfg(file_path: Path) -> set:
    # Load variable names from weather .cfg
    vars_set = set()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                vars_set.add(line)
    return vars_set


def save_preset(filename: str, directory: Path, content: str):
    path = directory / f"{filename}.cfg"
    with open(path, "w") as f:
        f.write(content)

# Store variables added by presets
preset_selected_vars = set()

def apply_weather_presets(selected_presets: list[str], weather_model: QStandardItemModel, update_label):
    global preset_selected_vars

    combined_vars = set()

    # Load variables for all selected presets
    for preset_name in selected_presets:
        file_path = WEATHER_DIR / f"{preset_name}.cfg"
        if file_path.exists():
            combined_vars |= load_weather_cfg(file_path)

    # Block signals while changing check states
    weather_model.blockSignals(True)

    # Uncheck variables no longer selected
    for var in preset_selected_vars - combined_vars:
        for i in range(weather_model.rowCount()):
            item = weather_model.item(i)
            if item.text() == var and item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)

    # Check variables in currently selected presets
    for i in range(weather_model.rowCount()):
        item = weather_model.item(i)
        if item.text() in combined_vars:
            item.setCheckState(Qt.Checked)

    weather_model.blockSignals(False)

    # Update the combo box label after all changes
    QTimer.singleShot(0, update_label)

    preset_selected_vars = combined_vars


# ---------- Music functions ----------

def init_music_player(music_path: Path):
    pygame.mixer.init()
    pygame.mixer.music.load(str(music_path))
    pygame.mixer.music.play(-1)  # Loop indefinitely

def create_music_controls(layout: QVBoxLayout, reference_combo: QComboBox = None):
    music_container = QWidget()
    music_layout = QHBoxLayout(music_container)
    music_layout.setContentsMargins(0, 0, 0, 0)
    music_layout.setSpacing(4)
    music_layout.setAlignment(Qt.AlignLeft)

    # Container height matches reference combo
    container_height = reference_combo.sizeHint().height() if reference_combo else 25
    music_container.setFixedHeight(container_height)

    # Buttons can be smaller independently
    btn_size = container_height // 2  # smaller than container
    play_btn = QPushButton("►")
    pause_btn = QPushButton("❚❚")
    for btn in (play_btn, pause_btn):
        btn.setFixedSize(btn_size, btn_size)
        btn.setStyleSheet(f"""
            QPushButton {{
                color: white;
                background-color: transparent;
                border: none;
                font-size: {btn_size}px;
                padding: 0;
                margin: 0;
            }}
            QPushButton:hover {{
                background-color: #2A2B2C;
            }}
            QPushButton:pressed {{
                background-color: #111111;
            }}
        """)

    # Volume slider styling
    volume_slider = QSlider(Qt.Horizontal)
    volume_slider.setFixedHeight(container_height // 3)
    volume_slider.setFixedWidth(container_height * 6)
    volume_slider.setRange(0, 100)
    volume_slider.setValue(50)

    volume_slider.setStyleSheet(f"""
        QSlider {{
            background: transparent;
        }}
        QSlider::groove:horizontal {{
            border: 1px solid #2A2B2C;
            height: {container_height // 8}px;
            background: #1B1C1D;
            border-radius: {container_height // 10}px;
        }}
        QSlider::handle:horizontal {{
            background: white;
            border: none;
            width: {container_height // 4}px;
            margin: -{container_height // 10}px 0;
            border-radius: {container_height // 4}px;
        }}
        QSlider::sub-page:horizontal {{
            background: white;
            border-radius: {container_height // 10}px;
        }}
        QSlider::add-page:horizontal {{
            background: #1B1C1D;
            border-radius: {container_height // 10}px;
        }}
    """)

    music_layout.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Fixed, QSizePolicy.Minimum))
    music_layout.addWidget(play_btn)
    music_layout.addWidget(pause_btn)
    music_layout.addWidget(volume_slider)

    layout.addWidget(music_container)

    # Connect buttons
    play_btn.clicked.connect(lambda: pygame.mixer.music.unpause())
    pause_btn.clicked.connect(lambda: pygame.mixer.music.pause())
    volume_slider.valueChanged.connect(lambda val: pygame.mixer.music.set_volume(val / 100))




# ---------- Subcontainers ----------

class DegreeLineEdit(QLineEdit):
    """Strict numeric input: 0–9 + - .  (comma auto-converts to dot)"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.suffix = "°"
        self.allowed_chars = "0123456789+-."

        self.setAlignment(Qt.AlignRight)
        self.setStyleSheet("color: white;")
        self.setFixedWidth(60)

        self.setText(self.suffix)
        self.setCursorPosition(0)

        self.textChanged.connect(self._enforce_suffix)

    def _enforce_suffix(self, text: str):
        if not text.endswith(self.suffix):
            clean = text.replace(self.suffix, "")
            self.blockSignals(True)
            self.setText(clean + self.suffix)
            self.blockSignals(False)

        self.setCursorPosition(len(self.text()) - 1)

    def keyPressEvent(self, event):
        key = event.key()
        key_text = event.text()
        pos = self.cursorPosition()
        text = self.text()
        limit = len(text) - 1  # position of °

        # Prevent moving cursor into °
        if key == Qt.Key_Right and pos >= limit:
            return

        # Handle Backspace
        if key == Qt.Key_Backspace:
            if pos == 0:
                return
            clean = text[:pos - 1] + text[pos:]
            clean = clean.replace(self.suffix, "")
            self.setText(clean + self.suffix)
            self.setCursorPosition(pos - 1)
            return

        # Handle Delete
        if key == Qt.Key_Delete:
            if pos >= limit:
                return
            clean = text[:pos] + text[pos + 1:]
            clean = clean.replace(self.suffix, "")
            self.setText(clean + self.suffix)
            self.setCursorPosition(pos)
            return

        # Allow navigation keys
        if not key_text:
            super().keyPressEvent(event)
            return

        # Convert comma to dot
        if key_text == ",":
            key_text = "."
        # Allow only one decimal point
        clean = text.replace(self.suffix, "")
        if key_text == "." and "." in clean:
            return

        # Reject invalid symbols
        if key_text not in self.allowed_chars:
            return

        clean = text.replace(self.suffix, "")
        clean = clean[:pos] + key_text + clean[pos:]
        self.setText(clean + self.suffix)
        self.setCursorPosition(pos + 1)


def create_subcontainer1() -> QWidget:
    container, outer_layout = create_container(
        f"border: 2px solid {BORDER_COLOR}; padding: 0px;"
    )
    container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(0)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    # ---------- Weather combo ----------
    weather_combo = QComboBox()
    weather_combo.setEditable(True)
    weather_combo.lineEdit().setReadOnly(True)
    w_model = QStandardItemModel(weather_combo)
    weather_combo.setModel(w_model)
    set_combo_width_for_text(weather_combo, "Selected weather criteria: ", 3)

    for v in WEATHER_VARIABLES:
        item = QStandardItem(v)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        w_model.appendRow(item)

    def update_weather_count():
        n = sum(
            w_model.item(i).checkState() == Qt.Checked
            for i in range(w_model.rowCount())
        )
        weather_combo.lineEdit().setText(f"Selected weather criteria: {n}")

    # Connect with deferred call
    w_model.itemChanged.connect(lambda _: QTimer.singleShot(0, update_weather_count))


    # Connect with deferred call to avoid race conditions
    w_model.itemChanged.connect(lambda _: QTimer.singleShot(0, update_weather_count))
    update_weather_count()

    # ---------- Preset combo ----------
    preset_combo = QComboBox()
    preset_combo.setEditable(True)
    preset_combo.lineEdit().setReadOnly(True)
    p_model = QStandardItemModel(preset_combo)
    preset_combo.setModel(p_model)
    set_combo_width_for_text(preset_combo, "Selected criteria presets: ", 3)

    # Populate with available .cfg files from presets/weather
    for cfg_file in WEATHER_DIR.glob("*.cfg"):
        it = QStandardItem(cfg_file.stem)
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Unchecked)
        p_model.appendRow(it)

    def update_presets_and_apply():
        selected_presets = [
            p_model.item(i).text()
            for i in range(p_model.rowCount())
            if p_model.item(i).checkState() == Qt.Checked
        ]
        preset_combo.lineEdit().setText(f"Selected criteria presets: {len(selected_presets)}")
        apply_weather_presets(selected_presets, w_model, update_weather_count)  # ✅ pass callback

    p_model.itemChanged.connect(lambda _: QTimer.singleShot(0, update_presets_and_apply))
    update_presets_and_apply()

    # ---------- Music widget ----------
    music = QWidget()
    music_layout = QHBoxLayout(music)
    music_layout.setContentsMargins(0, 0, 0, 0)
    music_layout.setSpacing(2)

    left_stack = QVBoxLayout()
    left_stack.setContentsMargins(0, 0, 0, 0)
    left_stack.setSpacing(2)
    left_stack.addWidget(weather_combo)
    left_stack.addWidget(preset_combo)
    create_music_controls(left_stack, reference_combo=preset_combo)
    left_stack.addWidget(music)
    grid.addLayout(left_stack, 0, 0, Qt.AlignLeft | Qt.AlignTop)

    # ---------- NSEW inputs ----------
    total_height = weather_combo.sizeHint().height()
    nsew_inputs = {}
    nsew_layout = QHBoxLayout()
    nsew_layout.setSpacing(2)
    nsew_layout.setContentsMargins(0, 0, 0, 0)

    for d, key in zip("NSEW", ["north", "south", "east", "west"]):
        v = QVBoxLayout()
        v.setSpacing(2)
        v.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(d)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(total_height - 2)
        degree_input = DegreeLineEdit()
        degree_input.setFixedHeight(total_height - 2)
        nsew_inputs[key] = degree_input
        v.addWidget(lbl)
        v.addWidget(degree_input)
        nsew_layout.addLayout(v)

    # ---------- Geo combo ----------
    geo_combo = QComboBox()
    geo_combo.setFixedHeight(preset_combo.sizeHint().height())
    geo_combo.setEditable(True)
    geo_combo.lineEdit().setReadOnly(True)

    geo_combo.addItem("None")
    for cfg_file in GEO_DIR.glob("*.cfg"):
        geo_combo.addItem(cfg_file.stem)
    geo_combo.lineEdit().setText("Selected geo preset: None")
    set_combo_width_for_text(geo_combo, "Selected geo preset: ", 3)

    def apply_geo_from_combo(index):
        preset_name = geo_combo.itemText(index)
        geo_combo.lineEdit().setText(f"Selected geo preset: {preset_name}")
        if preset_name == "None":
            for widget in nsew_inputs.values():
                widget.setText("")
            return
        file_path = GEO_DIR / f"{preset_name}.cfg"
        if file_path.exists():
            data = load_geo_cfg(file_path)
            for k, widget in nsew_inputs.items():
                if k in data:
                    val = data[k]
                    if val == int(val):
                        val = int(val)
                    widget.setText(f"{val}°")

    geo_combo.activated.connect(apply_geo_from_combo)

    right_stack = QVBoxLayout()
    right_stack.setContentsMargins(0, 0, 0, 0)
    right_stack.setSpacing(2)
    right_stack.addLayout(nsew_layout)
    right_stack.addWidget(geo_combo, alignment=Qt.AlignTop)

    grid.addLayout(right_stack, 0, 1, Qt.AlignRight | Qt.AlignTop)
    outer_layout.addLayout(grid)

    container.nsew_inputs = nsew_inputs
    container.weather_model = w_model
    container.preset_model = p_model
    container.geo_combo = geo_combo

    return container


def create_subcontainer2(weather_model: QStandardItemModel) -> QWidget:
    subcontainer, sub_layout = create_container()
    combo = SelectedWeatherCombo(weather_model)
    sub_layout.addWidget(combo)

    subcontainer.weather_combo = combo  # ✅ store reference
    return subcontainer


def create_subcontainer3() -> QWidget:
    subcontainer, layout = create_container()

    # ---------- DATE ROW ----------
    date_row = QHBoxLayout()

    date_label_start = QLabel("Start")
    date_label_end = QLabel("End")

    start_date_day = QDateEdit()
    end_date_day = QDateEdit()
    start_date_day.setCalendarPopup(True)
    end_date_day.setCalendarPopup(True)
    start_date_day.setDisplayFormat("yyyy-MM-dd")
    end_date_day.setDisplayFormat("yyyy-MM-dd")

    date_row.addWidget(date_label_start)
    date_row.addWidget(start_date_day)
    date_row.addSpacing(10)
    date_row.addWidget(date_label_end)
    date_row.addWidget(end_date_day)

    layout.addLayout(date_row)

    # ---------- MODE ROW ----------
    daily_rb = QRadioButton("Daily")
    seasonal_rb = QRadioButton("Seasonal inte.")

    every_y_hour_cb = QCheckBox("Every Y hours")
    hour_spin = QSpinBox()
    hour_spin.setRange(1, 23)
    hour_spin.setValue(1)
    hour_spin.setFixedWidth(60)
    hour_spin.setEnabled(False)

    def update_states():
        daily_selected = daily_rb.isChecked()
        seasonal_selected = seasonal_rb.isChecked()
        every_y_hour_cb.setEnabled(not (daily_selected or seasonal_selected))
        hour_spin.setEnabled(every_y_hour_cb.isChecked())

    for w in (daily_rb, seasonal_rb, every_y_hour_cb):
        w.toggled.connect(update_states)

    mode_row = QHBoxLayout()
    mode_row.addWidget(daily_rb)
    mode_row.addWidget(seasonal_rb)
    mode_row.addSpacing(10)
    mode_row.addWidget(every_y_hour_cb)
    mode_row.addWidget(hour_spin)
    mode_row.addStretch()
    layout.addLayout(mode_row)
    update_states()

    # ---------- TIME RANGE ----------
    time_row = QHBoxLayout()
    start_hour = QSpinBox()
    start_hour.setRange(0, 23)
    start_hour.setValue(0)
    start_hour.setSuffix(":00")

    end_hour = QSpinBox()
    end_hour.setRange(0, 23)
    end_hour.setValue(23)
    end_hour.setSuffix(":00")

    time_row.addWidget(QLabel("Start time"))
    time_row.addWidget(start_hour)
    time_row.addSpacing(10)
    time_row.addWidget(QLabel("End time"))
    time_row.addWidget(end_hour)
    layout.addLayout(time_row)

    # Store widgets on subcontainer for later reference
    subcontainer.start_hour = start_hour
    subcontainer.end_hour = end_hour
    subcontainer.hour_spin = hour_spin
    subcontainer.start_date = start_date_day
    subcontainer.end_date = end_date_day
    subcontainer.daily_rb = daily_rb
    subcontainer.seasonal_rb = seasonal_rb
    subcontainer.every_y_hour_cb = every_y_hour_cb

    return subcontainer



# ---------- Button creators ----------

def create_run(sub1=None, sub2=None, sub3=None) -> QPushButton:
    btn = QPushButton("RUN")
    btn.setStyleSheet(BUTTON_STYLE_ROW)
    btn.setFixedHeight(30)

    if sub1 and sub3:
        btn.clicked.connect(lambda: run_era5_filter_with_mask_maps(build_era5_object(sub1, sub2, sub3)))

    return btn


# input object
def build_era5_object(sub1: QWidget, sub2: QWidget, sub3: QWidget) -> dict:
    """
    Build canonical ERA5 request object from GUI state using direct widget references.
    Monthly mode removed; only daily, seasonal, and every-Y-hours retained.
    """

    # -----------------------------
    # VARIABLES
    # -----------------------------
    variables = {}
    weather_model = sub1.weather_model
    combo = sub2.weather_combo

    for i in range(weather_model.rowCount()):
        item = weather_model.item(i)
        if item.checkState() == Qt.Checked:
            name = item.text()
            if combo.popup and name in combo.popup.item_widgets:
                min_edit, max_edit = combo.popup.item_widgets[name]
                try:
                    min_v = float(min_edit.text())
                except ValueError:
                    min_v = None
                try:
                    max_v = float(max_edit.text())
                except ValueError:
                    max_v = None
            else:
                min_v, max_v = combo.minmax_values.get(name, (None, None))

            variables[name] = {"min": min_v, "max": max_v}

    # -----------------------------
    # GEO
    # -----------------------------
    geo = {}
    for k, widget in sub1.nsew_inputs.items():
        text = widget.text().replace("°", "").strip()
        geo[k] = float(text) if text else None

    # -----------------------------
    # TIME DEFAULTS
    # -----------------------------
    time = {
        "mode": None,
        "range": {"start": None, "end": None},
        "stride": {"hours": None},
        "filters": {"hours": {"start": 0, "end": 23}},
        "aggregation": {"type": None}
    }

    # -----------------------------
    # MODE WIDGETS
    # -----------------------------
    daily_rb = sub3.daily_rb
    seasonal_rb = sub3.seasonal_rb
    every_y_hour_cb = sub3.every_y_hour_cb

    start_hour = sub3.start_hour
    end_hour = sub3.end_hour
    hour_spin = sub3.hour_spin

    start_date = sub3.start_date
    end_date = sub3.end_date

    # -----------------------------
    # MODE LOGIC
    # -----------------------------
    if daily_rb.isChecked():
        time["mode"] = "daily"
        time["aggregation"]["type"] = "daily"
    elif seasonal_rb.isChecked():
        time["mode"] = "seasonal"
        time["aggregation"]["type"] = "seasonal"

    # -----------------------------
    # HOURS
    # -----------------------------
    time["filters"]["hours"] = {"start": start_hour.value(), "end": end_hour.value()}
    if every_y_hour_cb.isChecked():
        time["stride"]["hours"] = hour_spin.value()

    # -----------------------------
    # DATE RANGE
    # -----------------------------
    time["range"]["start"] = start_date.date().toString("yyyy-MM-dd")
    time["range"]["end"] = end_date.date().toString("yyyy-MM-dd")

    return {
        "variables": variables,
        "geo": geo,
        "time": time
    }


# ---------- Main window ----------

def create_main_window() -> QWidget:
    window = QWidget()
    window.setStyleSheet("background-color: black;")
    width, height = get_scaled_window_size(0.3, 0.5)
    window.setFixedSize(width, height)
    main_layout = QVBoxLayout(window)

    # Logo
    logo_path = get_path(logo_f)
    main_layout.addWidget(create_logo(logo_path, 80))

    # Subcontainer 1
    sub1 = create_subcontainer1()
    main_layout.addWidget(sub1)

    # Subcontainer 2
    weather_combo = sub1.findChild(QComboBox)
    w_model = weather_combo.model() if weather_combo else None
    if w_model:
        sub2 = create_subcontainer2(w_model)
        main_layout.addWidget(sub2)

    # Subcontainer 3
    sub3 = create_subcontainer3()
    main_layout.addWidget(sub3)

    # ---------- Responsive buttons row below subcontainer3 ----------
    button_row = QHBoxLayout()
    button_row.setContentsMargins(10, 10, 0, 0)
    button_row.setSpacing(10)

    run_btn = create_run(sub1, sub2, sub3)
    run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    button_row.addWidget(run_btn)

    main_layout.addLayout(button_row)

    window.setWindowTitle("Climate Observation & Regional Earth eXplorer")
    return window


