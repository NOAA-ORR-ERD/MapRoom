# Enthought library imports.
from envisage.ui.tasks.api import PreferencesPane, TaskFactory
from apptools.preferences.api import PreferencesHelper
from traits.api import Bool, Dict, Enum, List, Str, Unicode
from traitsui.api import EnumEditor, HGroup, VGroup, Item, Label, \
    View

## fixme:
## Some hard_coded stuff just to put it in a central place -- should be handled smarter

# EPSG:3857 now the default projection due to the use of WMS and Tile servers
DEFAULT_PROJECTION_STRING = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +over +no_defs"


class MaproomPreferences(PreferencesHelper):
    """ The preferences helper for the Framework application.
    """

    #### 'PreferencesHelper' interface ########################################

    # The path to the preference node that contains the preferences.
    preferences_path = 'maproom'

    #### Preferences ##########################################################

    # Lat/lon degree display format
    coordinate_display_format = Enum(
        "degrees decimal minutes",
        "degrees minutes seconds",
        "decimal degrees",
        )

    # mouse wheel zoom speed
    zoom_speed = Enum(
        "Slow",
        "Medium",
        "Fast",
        )

    # display scale legend by default
    show_scale = Bool(True)

    # check for layer errors on file save
    check_errors_on_save = Bool(True)


class MaproomPreferencesPane(PreferencesPane):
    """ The preferences pane for the Framework application.
    """

    #### 'PreferencesPane' interface ##########################################

    # The factory to use for creating the preferences model object.
    model_factory = MaproomPreferences

    category = Str('MapRoom')

    #### 'FrameworkPreferencesPane' interface ################################

    view = View(
        VGroup(HGroup(Item('coordinate_display_format'),
                      Label('Coordinate Display Format'),
                      show_labels = False),
               HGroup(Item('zoom_speed'),
                      Label('Scroll Zoom Speed'),
                      show_labels = False),
               HGroup(Item('show_scale'),
                      Label('Show Scale Layer'),
                      show_labels = False),
               HGroup(Item('check_errors_on_save'),
                      Label('Check for layer errors on save'),
                      show_labels = False),
               label='MapRoom'),
        resizable=True)
