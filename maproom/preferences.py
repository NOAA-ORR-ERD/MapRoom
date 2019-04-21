import os

from sawx import persistence
from sawx.preferences import SawxEditorPreferences

# fixme:
# Some hard_coded stuff just to put it in a central place -- should be handled smarter

# EPSG:3857 now the default projection due to the use of WMS and Tile servers
DEFAULT_PROJECTION_STRING = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +over +no_defs"


# Add unique MapRoom preferences types

import wx
from sawx.ui.prefs_dialog import InfoField, register_preference_field
from .library.colormap.ui_combobox import ColormapComboBox


class ColormapField(InfoField):
    default_width = 300

    def fill_data(self):
        name = self.get_value()
        self.ctrl.rebuild_colormap_list()
        self.ctrl.set_selection_by_name(name)

    def create_control(self, settings):
        c = ColormapComboBox(self.container, -1, "", size=(self.default_width, -1), popup_width=300)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        name = self.ctrl.get_selected_name()
        setattr(self.prefs, self.attrib_name, name)

register_preference_field("colormap", ColormapField)


class MaproomPreferences(SawxEditorPreferences):
    # Lat/lon degree display format
    coordinate_display_format = [
        "decimal degrees",
        "degrees decimal minutes",
        "degrees minutes seconds",
    ]

    # mouse wheel zoom speed
    zoom_speed = [
        "Slow",
        "Medium",
        "Fast",
    ]

    display_order = [
        ("coordinate_display_format", coordinate_display_format),
        ("zoom_speed", zoom_speed),
        ("show_scale", "bool"),
        ("check_errors_on_save", "bool"),
        ("identify_layers", "bool", "Blink newly selected layer when switching"),
        ("grid_spacing", "intrange:25-200"),
        ("download_directory", "directory"),
        ("bsb_directory", "directory"),
        ("colormap_name", "colormap", "Default colormap"),
    ]

    def set_defaults(self):
        self.coordinate_display_format = "degrees decimal minutes"

        self.zoom_speed = "Slow"

        # display scale legend by default
        self.show_scale = True

        # check for layer errors on file save
        self.check_errors_on_save = True

        # blink newly selected layer when switching
        self.identify_layers = True

        # minimum number of pixels between grid lines
        self.grid_spacing = 100

        self.download_directory = persistence.get_user_dir("Downloads")

        self.bsb_directory = persistence.get_user_dir("BSB")

        self.colormap_name = "gist_heat"
