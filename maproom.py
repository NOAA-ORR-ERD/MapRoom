# Maproom application using the Peppy2 framework

# Standard library imports.
import logging

# Must set these environmental vars early, before any of the Enthought
# libraries are loaded, because some are only used at module load time.
import os

# Debugging turned on for readable exceptions on Enthought ui module import
os.environ["ETS_DEBUG"] = "True"

# Use the agw.aui toolbar because the standard toolbar in AUI doesn't refresh
# properly on Mac
os.environ["ETS_AUI_TOOLBAR"] = "True"

# Framework imports.
from peppy2 import get_image_path
from peppy2.framework.application import run

# Local imports.
from maproom.plugin import MaproomPlugin

# Imports for py2exe/py2app
import wx
import multiprocessing
multiprocessing.freeze_support()

def main(argv):
    """ Run the application.
    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for toolkit in ['pyface', 'envisage', 'traits', 'traitsui', 'apptools']:
        _ = logging.getLogger(toolkit)
        _.setLevel(logging.WARNING)

    plugins = [ MaproomPlugin() ]
    
    import maproom.file_type
    plugins.extend(maproom.file_type.plugins)
    
    import maproom
    image_path = [get_image_path("icons", maproom)]
    run(plugins=plugins, image_path=image_path, use_eggs=False, startup_task="maproom.project.v4")

    logging.shutdown()


if __name__ == '__main__':
    import sys
    
    main(sys.argv)
