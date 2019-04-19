import os

from sawx import persistence
from sawx.preferences import SawxPreferences

# fixme:
# Some hard_coded stuff just to put it in a central place -- should be handled smarter

# EPSG:3857 now the default projection due to the use of WMS and Tile servers
DEFAULT_PROJECTION_STRING = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +over +no_defs"


class MaproomPreferences(SawxPreferences):
    # Lat/lon degree display format
    coordinate_display_format = [
        "degrees decimal minutes",
        "degrees minutes seconds",
        "decimal degrees",
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
        ("colormap_name", "colormap"),
    ]

    def __init__(self):
        self.coordinate_display_format = "degrees decimal minutes"

        self.zoom_speed = "Slow"

        # display scale legend by default
        self.show_scale = True

        # check for layer errors on file save
        self.check_errors_on_save = True

        # blink newly selected layer when switching
        self.identify_layers = True

        # minimum number of pixels between grid lines
        self.grid_spacing_low = 25
        self.grid_spacing_high = 200
        self.grid_spacing = 100

        self.download_directory = persistence.get_user_dir("Downloads")

        self.bsb_directory = persistence.get_user_dir("BSB")

        self.colormap_name = "gist_heat"


# class MaproomPreferencesPane(PreferencesPane):
#     """ The preferences pane for the Framework application.
#     """

#     # 'PreferencesPane' interface ##########################################

#     # The factory to use for creating the preferences model object.
#     model_factory = MaproomPreferences

#     category = Str('MapRoom')

#     # 'FrameworkPreferencesPane' interface ################################

#     view = View(
#         VGroup(HGroup(Item('coordinate_display_format'),
#                       Label('Coordinate Display Format'),
#                       show_labels=False),
#                HGroup(Item('zoom_speed'),
#                       Label('Scroll Zoom Speed'),
#                       show_labels=False),
#                HGroup(Item('grid_spacing', editor=RangeEditor(mode="spinner", is_float=False, low_name='grid_spacing_low', high_name='grid_spacing_high')),
#                       Label('Minimum Number of Pixels Between Grid Lines'),
#                       show_labels=False),
#                HGroup(Item('show_scale'),
#                       Label('Show Scale Layer'),
#                       show_labels=False),
#                HGroup(Item('check_errors_on_save'),
#                       Label('Check for layer errors on save'),
#                       show_labels=False),
#                HGroup(Item('identify_layers'),
#                       Label('Briefly highlight the selected layer when chosen'),
#                       show_labels=False),
#                HGroup(Item('download_directory'),
#                       Label('Download directory'),
#                       show_labels=False),
#                HGroup(Item('bsb_directory'),
#                       Label('BSB storage directory'),
#                       show_labels=False),
#                label='MapRoom'),
#         resizable=True)
