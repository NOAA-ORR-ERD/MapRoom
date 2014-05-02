# Enthought library imports.
from envisage.ui.tasks.api import PreferencesPane, TaskFactory
from apptools.preferences.api import PreferencesHelper
from traits.api import Bool, Dict, Enum, List, Str, Unicode
from traitsui.api import EnumEditor, HGroup, VGroup, Item, Label, \
    View


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
               label='MapRoom'),
        resizable=True)
