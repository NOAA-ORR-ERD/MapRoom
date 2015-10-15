# Standard library imports.
import os.path

# Enthought library imports.
from envisage.api import ExtensionPoint
from envisage.ui.tasks.api import TaskFactory
from traits.api import List

from omnimon.framework.plugin import FrameworkPlugin

from preferences import MaproomPreferencesPane

class MaproomPlugin(FrameworkPlugin):
    """ The sample framework plugin.
    """

    # Extension point IDs.
    TASK_FACTORIES  = 'envisage.ui.tasks.tasks'
    PREFERENCES_PANES = 'envisage.ui.tasks.preferences_panes'

    #### 'IPlugin' interface ##################################################

    # The plugin's unique identifier.
    id = 'omnimon.tasks'

    # The plugin's name (suitable for displaying to the user).
    name = 'MapRoom'

    #### Contributions to extension points made by this plugin ################

    task_factories = List(contributes_to=TASK_FACTORIES)
    preferences_panes = List(contributes_to=PREFERENCES_PANES)

    ###########################################################################
    # Protected interface.
    ###########################################################################

    def _preferences_panes_default(self):
        return [ MaproomPreferencesPane ]

    def _task_factories_default(self):
        from maproom.task import MaproomProjectTask

        return self.task_factories_from_tasks([
                MaproomProjectTask,
            ])

    def start(self):
        pass
