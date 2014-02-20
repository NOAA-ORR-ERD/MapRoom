# Maproom application using the Peppy2 framework

# Standard library imports.
from pkg_resources import Environment, working_set
import logging

# Enthought library imports.
from traits.etsconfig.api import ETSConfig
from envisage.api import PluginManager
from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin
 
# Local imports.
from peppy2.framework.plugin import FrameworkPlugin
from peppy2.file_type.plugin import FileTypePlugin
from peppy2.framework.application import FrameworkApplication
from maproom.plugin import MaproomPlugin

def main(argv):
    """ Run the application.
    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    plugins = [ CorePlugin(), TasksPlugin(), FrameworkPlugin(), FileTypePlugin(), MaproomPlugin() ]
    
    import peppy2.file_type.recognizers
    plugins.extend(peppy2.file_type.recognizers.plugins)
    
    default = PluginManager(
        plugins = plugins,
    )
    plugin_manager = default
    
    app = FrameworkApplication(plugin_manager=plugin_manager)
    
    app.run()

    logging.shutdown()


if __name__ == '__main__':
    import sys
    
    main(sys.argv)
