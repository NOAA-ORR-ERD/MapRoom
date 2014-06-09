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
#        if hasattr(cls, 'can_load') and name != "BaseLoader":
#            # make sure class is from this module and not an imported dependency
#            if cls.__module__.startswith(modname):
#                names.append(name)
#    if names:
#       cog.outl("from %s import %s" % (modname, ", ".join(names)))
#       for name in names:
#           cog.outl("loaders.append(%s())" % name)
# ]]]*/
loaders = []
from ugrid import UGridLoader
loaders.append(UGridLoader())
from bna import BNALoader
loaders.append(BNALoader())
from gdal import GDALLoader
loaders.append(GDALLoader())
from verdat import VerdatLoader
loaders.append(VerdatLoader())
# [[[end]]]

import os

def load_layers(metadata, manager=None):
    for loader in loaders:
        print "trying loader %s" % loader.name
        if loader.can_load(metadata):
            print " loading using loader %s!" % loader.name
            layers = loader.load(metadata, manager=manager)
            print " loaded layers: \n  %s" % "\n  ".join([str(a) for a in layers])
            return layers
    return None

def check_layer(layer):
    for loader in loaders:
        # FIXME: only the first loader that can check the layer is used.  How
        # would we present results of multiple loaders?
        if loader.can_save(layer):
            loader.check(layer)
            return "This is a valid %s file" % loader.name

def find_best_saver(savers, uri):
    name, ext = os.path.splitext(uri)
    if len(savers) > 1:
        for saver in savers:
            if saver.is_valid_extension(ext):
                return saver
    return savers[0]

def save_layer(layer, uri):
    savers = []
    for loader in loaders:
        if loader.can_save(layer):
            savers.append(loader)
    
    if uri is None:
        uri = layer.file_path
    
    saver = find_best_saver(savers, uri)
    if not saver:
        return "Layer type %s cannot be saved." % layer.type
    
    error = saver.save(uri, layer)
    return error
