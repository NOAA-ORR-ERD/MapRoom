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
#cog.msg("scanning dir: %s" % path)
#top = os.path.abspath(os.path.join(path, "../..")) # so absolute imports of maproom will work
#sys.path.append(top)
#cog.msg("top dir     : %s" % top)
#import glob
#cog.outl("recognizers = []")
#source_files = glob.glob(os.path.join(path, "*.py"))
#source_files.sort()
#for filename in source_files:
#    if filename.endswith("__init__.py"):
#        continue
#    modname = filename.split(".py")[0].split("/")[-1]
#    module = imp.load_source(modname, filename)
#    members = inspect.getmembers(module, inspect.isclass)
#    names = []
#    for name, cls in members:
#        if hasattr(cls, 'identify'):
#            # make sure class is from this module and not an imported dependency
#            if cls.__module__.startswith(modname):
#                names.append(name)
#    if names:
#       cog.outl("from %s import %s" % (modname, ", ".join(names)))
#       for name in names:
#           cog.outl("recognizers.append(%s())" % name)
# ]]]*/
recognizers = []
from .binary import MapRoomZipProjectRecognizer, NC_ParticleRecognizer, UGRID_Recognizer
recognizers.append(MapRoomZipProjectRecognizer())
recognizers.append(NC_ParticleRecognizer())
recognizers.append(UGRID_Recognizer())
from .image import GDALRecognizer
recognizers.append(GDALRecognizer())
from .text import BNARecognizer, BSBRecognizer, MapRoomCommandRecognizer, MapRoomProjectRecognizer, PlainTextRecognizer, VerdatRecognizer
recognizers.append(BNARecognizer())
recognizers.append(BSBRecognizer())
recognizers.append(MapRoomCommandRecognizer())
recognizers.append(MapRoomProjectRecognizer())
recognizers.append(PlainTextRecognizer())
recognizers.append(VerdatRecognizer())
from .vector import OGRRecognizer
recognizers.append(OGRRecognizer())
# [[[end]]]

from envisage.api import Plugin
from traits.api import List


class MaproomFileRecognizerPlugin(Plugin):
    """ A plugin that contributes to the omnivore_framework.file_type.recognizer extension point. """

    # 'IPlugin' interface ##################################################

    # The plugin's unique identifier.
    id = 'maproom.file_type.builtin'

    # The plugin's name (suitable for displaying to the user).
    name = 'MapRoom File Recognizer Plugin'

    # This tells us that the plugin contributes the value of this trait to the
    # 'greetings' extension point.
    recognizer = List(recognizers, contributes_to='omnivore_framework.file_recognizer')


plugins = [MaproomFileRecognizerPlugin()]
