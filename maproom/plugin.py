# Standard library imports.
import os.path

# Enthought library imports.
from envisage.api import ExtensionPoint, Plugin
from envisage.ui.tasks.api import TaskFactory
from traits.api import List


class MaproomPlugin(Plugin):
    """ The sample framework plugin.
    """

    # Extension point IDs.
    TASK_FACTORIES  = 'envisage.ui.tasks.tasks'

    #### 'IPlugin' interface ##################################################

    # The plugin's unique identifier.
    id = 'peppy2.tasks'

    # The plugin's name (suitable for displaying to the user).
    name = 'Maproom'

    #### Contributions to extension points made by this plugin ################

    task_factories = List(contributes_to=TASK_FACTORIES)
    recognizer = List(contributes_to='peppy2.file_recognizer')

    ###########################################################################
    # Protected interface.
    ###########################################################################

    def _task_factories_default(self):
        from maproom.task import MaproomProjectTask

        return [ 
            TaskFactory(id = 'maproom.project',
                        name = 'Maproom Project Task',
                        factory = MaproomProjectTask),
            ]

    def _recognizer_default(self):
        from maproom.file_type.text import VerdatRecognizer, BNARecognizer
        from maproom.file_type.image import GDALRecognizer
        return [VerdatRecognizer(), BNARecognizer(), GDALRecognizer()]
