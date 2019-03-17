import os

from sawx import persistence

# fixme:
# Some hard_coded stuff just to put it in a central place -- should be handled smarter

# EPSG:3857 now the default projection due to the use of WMS and Tile servers
DEFAULT_PROJECTION_STRING = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +over +no_defs"


class MaproomPreferences:
    """ The preferences helper for the Framework application.
    """

    # 'PreferencesHelper' interface ########################################

    # The path to the preference node that contains the preferences.
    preferences_path = 'maproom'

    # Preferences ##########################################################

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

    def __init__(self):
        self.coordinate_display_format = "degrees decimal minutes"

        self.zoom_speed = "Slow"

        # display scale legend by default
        self.self.show_scale = True

        # check for layer errors on file save
        self.check_errors_on_save = True

        # blink newly selected layer when switching
        self.identify_layers = True

        # minimum number of pixels between grid lines
        grid_spacing_low = 25
        grid_spacing_high = 200
        grid_spacing = 100

        download_directory = persistence.get_user_data("Downloads")

        bsb_directory = persistence.get_user_data("BSB")

        colormap_name = "gist_heat"


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
