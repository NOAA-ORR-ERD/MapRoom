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
    
    def undo(self):
        undo = self.layer_manager.undo_stack.undo(self)
        self.process_flags(undo.flags)
    
    def redo(self):
        undo = self.layer_manager.undo_stack.redo(self)
        self.process_flags(undo.flags)
    
    def process_command(self, command):
        if command is None:
            return
        undo = command.perform(self)
        self.layer_manager.undo_stack.add_command(command)
        self.process_flags(undo.flags)
    
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
