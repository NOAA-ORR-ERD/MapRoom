# Maproom application using the sawx framework

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

# Set GDAL_DATA environment variable, needed on windows to find support files
# to use whhen converting coordinate systems upon load of shapefiles
import osgeo.ogr
found = os.path.join(os.path.dirname(osgeo.ogr.__file__), "data")
if not os.path.exists(found):
    try:
        import fiona
    except ImportError:
        found = None
    else:
        found = os.path.join(os.path.dirname(fiona.__file__), "gdal_data")
if found is not None:
    os.environ["GDAL_DATA"] = found

# Framework imports.
from sawx.application import SawxApp
from sawx.filesystem import get_image_path
from sawx.startup import run, setup_frozen_logging

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
    caller_line_no = caller.f_lineno
    caller_filename = caller.f_code.co_filename
    f = caller_filename + func_filename
    if "trait_notifiers.py" in f or "trait_handlers" in f or "has_traits" in f or "logging" in f or "envisage" in f or "sre_" in f:
        return
    print('Call to %s on line %s of %s from line %s of %s' % \
        (func_name, func_line_no, func_filename,
         caller_line_no, caller_filename))
    return

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

    from maproom._version import __version__
    SawxApp.app_name = "MapRoom"
    SawxApp.app_version = __version__
    SawxApp.app_description = "High-performance 2d mapping"
    SawxApp.app_icon = "icon://maproom.ico"
    SawxApp.app_website = "http://www.noaa.gov"
    SawxApp.default_uri = "template://default_project.maproom"
    SawxApp.about_image = "icon://maproom_large.png"
    SawxApp.about_html = f"""<h2>{SawxApp.app_name} {SawxApp.app_version}</h2>

<h3>{SawxApp.app_description}</h3>

<p><img src="{SawxApp.about_image}">"""
    run(SawxApp, image_paths, template_paths, help_paths)

    logging.shutdown()


if __name__ == '__main__':
    import sys
    
    setup_frozen_logging()
    main(sys.argv)
