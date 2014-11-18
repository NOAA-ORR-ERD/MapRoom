import os

from nose.tools import *

import numpy as np

from pyugrid.ugrid import UGrid

from peppy2.utils.file_guess import FileGuess
from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries

class MockControl(object):
    def __init__(self):
        pass
        ## projection_is_identity is no more.
        #self.projection_is_identity = True

class MockProject(object):
    def __init__(self):
        self.control = MockControl()

class MockManager(object):
    def __init__(self):
        self.project = MockProject()
    
    def dispatch_event(self, event, value=True):
        pass
    
    def add_undo_operation_to_operation_batch(self, op, layer, index, values):
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
