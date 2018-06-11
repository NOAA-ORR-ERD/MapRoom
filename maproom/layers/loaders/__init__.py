# flake8: noqa  # cog scripts won't work with E265: extra space after the "#"

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
# the same level as the omnivore directory) or this directory.

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
#sys.path[0:0] = [path] # include current dir so it finds local gdal.py file
#cog.msg("scanning dir: %s" % path)
#top = os.path.abspath(os.path.join(path, "../../..")) # so absolute imports of omnivore will work
#sys.path.append(top)
#cog.msg("top dir     : %s" % top)
#import glob
#cog.outl("loaders = []")
#source_files = glob.glob(os.path.join(path, "*.py"))
#source_files.sort()
#for filename in source_files:
#    if filename.endswith("__init__.py"):
#        continue
#    modname = filename.rstrip(".py").split("/")[-1]
#    module = imp.load_source(modname, filename)
#    members = inspect.getmembers(module, inspect.isclass)
#    names = []
#    for name, cls in members:
#        if hasattr(cls, 'can_load') and not name.startswith("Base"):
#            # make sure class is from this module and not an imported dependency
#            if cls.__module__.startswith(modname):
#                names.append(name)
#    if names:
#       cog.outl("from %s import %s" % (modname, ", ".join(names)))
#       for name in names:
#           cog.outl("loaders.append(%s())" % name)
# ]]]*/
loaders = []
from .bna import RNCLoader
loaders.append(RNCLoader())
from .bsb import BSBLoader
loaders.append(BSBLoader())
from .gdal import GDALLoader
loaders.append(GDALLoader())
from .logfile import CommandLogLoader
loaders.append(CommandLogLoader())
from .nc_particles import ParticleLoader
loaders.append(ParticleLoader())
from .project import ProjectLoader, ZipProjectLoader
loaders.append(ProjectLoader())
loaders.append(ZipProjectLoader())
from .shapefile import ShapefileLoader, BNAShapefileLoader
loaders.append(ShapefileLoader())
loaders.append(BNAShapefileLoader())
from .text import LatLonTextLoader, LonLatTextLoader
loaders.append(LatLonTextLoader())
loaders.append(LonLatTextLoader())
from .ugrid import UGridLoader
loaders.append(UGridLoader())
from .verdat import VerdatLoader
loaders.append(VerdatLoader())
# [[[end]]]

import os

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def load_layers_from_url(url, mime, manager=None):
    from omnivore.utils.file_guess import FileGuess

    guess = FileGuess(url)
    guess.metadata.mime = mime
    metadata = guess.get_metadata()
    return load_layers(metadata, manager)


def get_loader(metadata):
    for loader in loaders:
        log.debug("trying loader %s" % loader.name)
        if loader.can_load(metadata):
            log.debug(" loading using loader %s!" % loader.name)
            return loader
    return None


def load_layers(metadata, manager=None, **kwargs):
    for loader in loaders:
        log.debug("trying loader %s" % loader.name)
        if loader.can_load(metadata):
            log.debug(" loading using loader %s!" % loader.name)
            layers = loader.load_layers(metadata, manager=manager, **kwargs)
            log.debug(" loaded layers: \n  %s" % "\n  ".join([str(a) for a in layers]))
            return loader, layers
    return None, None


def valid_save_formats(layer):
    valid = []
    for loader in loaders:
        if loader.can_save_layer(layer):
            valid.append((loader, "%s: %s" % (loader.name, loader.get_pretty_extension_list())))
    return valid


def get_valid_string(valid, capitalize=True):

    return "This layer can be saved in the following formats:\n(with allowed filename extensions)\n\n" + "\n\n".join(v[1] for v in valid)


def find_best_saver(savers, ext):
    for saver in savers:
        if saver.is_valid_extension(ext):
            return saver
    return None


def save_layer(layer, uri, saver=None):
    if uri is None:
        uri = layer.file_path

    name, ext = os.path.splitext(uri)

    if not saver:
        savers = []
        for loader in loaders:
            if loader.can_save_layer(layer):
                savers.append(loader)
        saver = find_best_saver(savers, ext)

    if not saver:
        valid = valid_save_formats(layer)
        if valid:
            return "The extension '%s' doesn't correspond to any format\nthat can save the '%s' layer type.\n\n%s" % (ext, layer.type, get_valid_string(valid))

    progress_log.info("TITLE=Saving %s" % uri)
    error = saver.save_layer(uri, layer)
    return error
