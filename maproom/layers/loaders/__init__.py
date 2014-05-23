# All built-in recognizer plugins should be listed in this file so that the
# application can import this single file and determine the default plugins.
# In addition, a list of plugins is expected so that the framework can import
# all built-in plugins at once.
#
# A cog script is included to automatically generate the expected code. Test with::
#
#    cog.py __init__.py
#
# which prints to standard output. To place the generated code in the file, use::
#
#    cog.py -r __init__.py
#
# Note that the cog script works when called from the top directory (.i.e.  at
# the same level as the peppy2 directory) or this directory.

# [[[cog
#import os
#import sys
#import inspect
#import imp
#
#from envisage.api import Plugin
#
#cwd = os.getcwd()
#cog.msg("working dir : %s" % cwd)
#path = os.path.dirname(os.path.join(cwd, cog.inFile))
#cog.msg("scanning dir: %s" % path)
#top = os.path.abspath(os.path.join(path, "../../..")) # so absolute imports of peppy2 will work
#sys.path.append(top)
#cog.msg("top dir     : %s" % top)
#import glob
#cog.outl("loaders = []")
#for filename in glob.iglob(os.path.join(path, "*.py")):
#    if filename.endswith("__init__.py"):
#        continue
#    modname = filename.rstrip(".py").split("/")[-1]
#    module = imp.load_source(modname, filename)
#    members = inspect.getmembers(module, inspect.isclass)
#    names = []
#    for name, cls in members:
#        if hasattr(cls, 'can_load'):
#            # make sure class is from this module and not an imported dependency
#            if cls.__module__.startswith(modname):
#                names.append(name)
#    if names:
#       cog.outl("from %s import %s" % (modname, ", ".join(names)))
#       for name in names:
#           cog.outl("loaders.append(%s())" % name)
# ]]]*/
loaders = []
from bna import BNALoader
loaders.append(BNALoader())
from verdat import VerdatLoader
loaders.append(VerdatLoader())
# [[[end]]]

def load_layer(metadata, manager=None):
    for loader in loaders:
        if loader.can_load(metadata):
            layer = loader.load(metadata, manager=manager)
            return layer
    return None

def check_layer(layer, info_handler, error_handler):
    saver = None
    for loader in loaders:
        # FIXME: only the first loader that can check the layer is used.  How
        # would we present results of multiple loaders?
        if loader.can_save(layer):
            loader.check(layer)
            return

def save_layer(layer, uri):
    saver = None
    for loader in loaders:
        if loader.can_save(layer):
            saver = loader
    if not saver:
        return "Layer type %s cannot be saved." % layer.type
    
    if uri is None:
        uri = layer.file_path
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, os.path.basename(uri))

    f = open(temp_file, "w")
    error = ""
    had_error = False
    try:
        saver.save_to_file(f, layer)
        if layer.get_num_points_selected(STATE_FLAGGED):
            layer.clear_all_selections(STATE_FLAGGED)
            layer.manager.dispatch_event('refresh_needed')
    except Exception as e:
        import traceback
        
        had_error = True
        print traceback.format_exc(e)
        if hasattr(e, "points") and e.points != None:
            layer.clear_all_selections(STATE_FLAGGED)
            for p in e.points:
                layer.select_point(p, STATE_FLAGGED)
            layer.manager.dispatch_event('refresh_needed')
        error = e.message
    finally:
        f.close()
    if (not had_error and temp_file and os.path.exists(temp_file)):
        try:
            shutil.copy(temp_file, uri)
            layer.file_path = uri
        except Exception as e:
            import traceback
        
            error = "Unable to save file to disk. Make sure you have write permissions to the file.\n\nSystem error was: %s" % e.message
            print traceback.format_exc(e)
    return error
