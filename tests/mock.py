import os

# Include maproom directory so that maproom modules can be imported normally
import sys
maproom_dir = os.path.realpath(os.path.abspath(".."))
if maproom_dir not in sys.path:
    sys.path.insert(0, maproom_dir)

from nose.tools import *

import numpy as np

from pyugrid.ugrid import UGrid

from peppy2.utils.file_guess import FileGuess
from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries
from maproom.command import UndoStack

class MockControl(object):
    def __init__(self):
        pass
        ## projection_is_identity is no more.
        #self.projection_is_identity = True

class MockProject(object):
    def __init__(self):
        self.control = MockControl()
        self.layer_manager = None
        self.window = None
    
    def load_file(self, path, mime):
        guess = FileGuess(os.path.realpath(path))
        guess.metadata.mime = mime
        metadata = guess.get_metadata()
        print metadata
        loader = loaders.get_loader(metadata)
        print loader
        if hasattr(loader, "load_project"):
            print "FIXME: Add load project command that clears all layers"
        elif hasattr(loader, "iter_log"):
            line = 0
            for cmd in loader.iter_log(metadata, self.layer_manager):
                line += 1
                errors = None
                if cmd.short_name == "load":
                    print cmd.metadata
                    if cmd.metadata.uri.startswith("TestData"):
                        cmd.metadata.uri = "../" + cmd.metadata.uri
                try:
                    undo = self.process_command(cmd)
                    if not undo.flags.success:
                        errors = undo.errors
                        break
                except Exception, e:
                    #errors = [str(e)]
                    #break
                    raise
            if errors is not None:
                text = "\n".join(errors)
                raise RuntimeError(text)
        else:
            cmd = LoadLayersCommand(metadata)
            self.process_command(cmd)
    
    def undo(self):
        undo = self.layer_manager.undo_stack.undo(self)
        self.process_flags(undo.flags)
    
    def redo(self):
        undo = self.layer_manager.undo_stack.redo(self)
        self.process_flags(undo.flags)
    
    def process_command(self, command):
        print "processing command %s" % command.short_name
        undo = self.layer_manager.undo_stack.perform(command, self)
        self.process_flags(undo.flags)
        return undo
    
    def process_flags(self, f):
        pass

class MockManager(object):
    def __init__(self):
        self.project = MockProject()
        self.project.layer_manager = self
        self.undo_stack = UndoStack()
    
    def dispatch_event(self, event, value=True):
        pass

    def load_all_layers(self, uri, mime):
        guess = FileGuess(uri)
        guess.metadata.mime = mime
        print guess
        print guess.metadata
        loader, layers = loaders.load_layers(guess.metadata, manager=self)
        return layers

    def load_first_layer(self, uri, mime):
        return self.load_all_layers(uri, mime)[0]
    
    def get_outer_boundary(self, layer):
        boundaries = Boundaries(layer)
        return boundaries.get_outer_boundary()
