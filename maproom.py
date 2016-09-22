# Maproom application using the Peppy2 framework

# Uncomment to enable OpenGL command tracing (very slow).  This must occur
# before any import of OpenGL which is why it's here.  :)
#import OpenGL
#OpenGL.FULL_LOGGING = True

# Standard library imports.
import logging

# Must set these environmental vars early, before any of the Enthought
# libraries are loaded, because some are only used at module load time.
import os

# Workaround for unknown locale bug on OS X; importing docutils here prevents
# the failure when loading docutils.core in the markup support for the text
# vector objects
import docutils
from docutils.core import publish_parts

# Debugging turned on for readable exceptions on Enthought ui module import
os.environ["ETS_DEBUG"] = "True"

# Framework imports.
from omnivore import get_image_path
from omnivore.framework.application import run, setup_frozen_logging

# Local imports.
from maproom.pane_layout import task_id_with_pane_layout
from maproom.plugin import MaproomPlugin

# Imports for py2exe/py2app
import wx
import multiprocessing
multiprocessing.freeze_support()
import markdown.util as markdown_utils

def main(argv):
    """ Run the application.
    """
    logging.basicConfig(level=logging.WARNING)
    if "-d" in argv:
        i = argv.index("-d")
        argv.pop(i)  # discard -d
        next = argv.pop(i)  # discard next
        if next == "all":
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
        else:
            loggers = next.split(",")
            for name in loggers:
                log = logging.getLogger(name)
                log.setLevel(logging.DEBUG)

    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    for toolkit in ['pyface', 'envisage', 'traits', 'traitsui', 'apptools']:
        _ = logging.getLogger(toolkit)
        _.setLevel(logging.WARNING)

    plugins = [ MaproomPlugin() ]
    
    import maproom.file_type
    plugins.extend(maproom.file_type.plugins)
    
    import maproom
    image_path = [get_image_path("icons", maproom)]
    run(plugins=plugins, image_path=image_path, use_eggs=False, startup_task=task_id_with_pane_layout, application_name="MapRoom")

    logging.shutdown()


if __name__ == '__main__':
    import sys
    
    setup_frozen_logging()
    main(sys.argv)
