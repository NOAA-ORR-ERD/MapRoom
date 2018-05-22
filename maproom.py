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
from omnivore.app_init import run, setup_frozen_logging

# Local imports.
from maproom.pane_layout import task_id_with_pane_layout
from maproom.plugin import MaproomPlugin

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
    for toolkit in ['pyface', 'envisage', 'traits', 'traitsui', 'apptools']:
        _ = logging.getLogger(toolkit)
        _.setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Override location of help pages so we don't get the framework user guide
    from traits.trait_base import get_resource_path
    import omnivore.help
    omnivore.help.help_dirs = ["maproom/help"]
    omnivore.help.root_resource_path = get_resource_path(1)

    plugins = [ MaproomPlugin() ]
    
    import maproom.file_type
    plugins.extend(maproom.file_type.plugins)

    if "--trace" in argv:
        import sys
        i = argv.index("--trace")
        argv.pop(i)
        sys.settrace(trace_calls)

    import maproom
    image_path = [get_image_path("icons", maproom)]
    template_path = [get_image_path("templates", maproom)]
    run(plugins=plugins, image_path=image_path, template_path=template_path, use_eggs=False, startup_task=task_id_with_pane_layout, application_name="MapRoom")

    logging.shutdown()


if __name__ == '__main__':
    import sys
    
    setup_frozen_logging()
    main(sys.argv)
