# Maproom application using the Peppy2 framework

# Standard library imports.
import logging

# Framework imports.
from peppy2 import get_image_path
from peppy2.framework.application import run

# Local imports.
from maproom.plugin import MaproomPlugin

# Imports for py2exe/py2app
import wx
import multiprocessing
multiprocessing.freeze_support()

import os
os.environ["ETS_DEBUG"] = "True"

def main(argv):
    """ Run the application.
    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    plugins = [ MaproomPlugin() ]
    
    import maproom
    image_path = [get_image_path("icons", maproom)]
    run(plugins=plugins, image_path=image_path, use_eggs=False, startup_task="maproom.project")

    logging.shutdown()


if __name__ == '__main__':
    import sys
    
    main(sys.argv)
