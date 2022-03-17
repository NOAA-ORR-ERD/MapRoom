# MapRoom application main script

# Uncomment to enable OpenGL command tracing (very slow).  This must occur
# before any import of OpenGL which is why it's here.  :)
#import OpenGL
#OpenGL.FULL_LOGGING = True

# Standard library imports.
import logging

# Must set these environmental vars early, before any of the Enthought
# libraries are loaded, because some are only used at module load time.
import os
import sys

# Workaround for unknown locale bug on OS X; importing docutils here prevents
# the failure when loading docutils.core in the markup support for the text
# vector objects
import docutils
from docutils.core import publish_parts

# Set GDAL_DATA environment variable, needed on windows to find support files
# to use when converting coordinate systems upon load of shapefiles
gdal_path = None
proj_path = None
if "CONDA_PREFIX" in os.environ:
    gdal_path = os.path.join(os.environ["CONDA_PREFIX"], "Library/share/gdal")
    proj_path = os.path.join(os.environ["CONDA_PREFIX"], "Library/share")
else:
    import osgeo.ogr
    gdal_path = os.path.join(os.path.dirname(osgeo.ogr.__file__), "data")
    if not os.path.exists(gdal_path):
        try:
            import fiona
        except ImportError:
            gdal_path = None
        else:
            gdal_path = os.path.join(os.path.dirname(fiona.__file__), "gdal_data")
if gdal_path is not None:
    os.environ["GDAL_DATA"] = gdal_path
elif sys.platform.startswith("win"):
    print("ERROR: GDAL_DATA environment not set; will have errors loading some shapfiles")
if proj_path is not None:
    os.environ["PROJ_LIB"] = proj_path

# Framework imports.
from maproom.app_framework.application import MafApp
from maproom.app_framework.filesystem import get_image_path
from maproom.app_framework.startup import run, setup_frozen_logging

# Imports for py2exe/py2app
import wx
import multiprocessing
multiprocessing.freeze_support()
import markdown.util as markdown_utils

def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        # Ignore write() calls from print statements
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    caller = frame.f_back
    caller_line_no = caller.f_lineno if caller is not None else "<none>"
    caller_filename = caller.f_code.co_filename if caller is not None else "<none>"
    f = caller_filename + func_filename
    if "trait_notifiers.py" in f or "trait_handlers" in f or "has_traits" in f or "logging" in f or "envisage" in f or "sre_" in f:
        return
    print('Call to %s on line %s of %s from line %s of %s' % \
        (func_name, func_line_no, func_filename,
         caller_line_no, caller_filename))
    return


from maproom._version import __version__

class MapRoomApp(MafApp):
    app_version = __version__

    def shutdown_subprocesses(self):
        from maproom.servers import stop_threaded_processing
        stop_threaded_processing()
        MafApp.shutdown_subprocesses(self)


def main(argv):
    """ Run the application.
    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    if "--trace" in argv:
        import sys
        i = argv.index("--trace")
        argv.pop(i)
        sys.settrace(trace_calls)

    import maproom
    image_paths = [get_image_path("icons", maproom)]
    template_paths = [get_image_path("templates", maproom)]
    help_paths = ["maproom/help"]
    run(MapRoomApp, image_paths, template_paths, help_paths)

    logging.shutdown()


if __name__ == '__main__':
    import sys

    setup_frozen_logging()
    main(sys.argv)
