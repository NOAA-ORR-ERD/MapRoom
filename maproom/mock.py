import os

from omnivore.utils.file_guess import FileGuess

from layer_manager import LayerManager
from layers import loaders, TriangleLayer, Layer
from library.Boundary import Boundaries
from library.projection import NullProjection
from library.host_utils import HostCache
from library.known_hosts import default_tile_hosts
from command import UndoStack, BatchStatus
from menu_commands import *

class MockApplication(object):
    command_line_args = []

class MockCanvas(object):
    def __init__(self):
        self.projection = NullProjection()

class MockWindow(object):
    def __init__(self):
        self.application = MockApplication()
        
    def error(self, *args, **kwargs):
        pass

class MockTask(object):
    def __init__(self, window):
        self.window = window
        HostCache.set_known_hosts(default_tile_hosts)

    def get_tile_server_id_from_url(self, url):
        index, host = HostCache.get_host_by_url(url)
        return index

    def get_tile_server_by_id(self, id):
        return HostCache.get_known_hosts()[id]

class MockTree(object):
    def __init__(self):
        self.layer = Layer()
    
    def get_selected_layer(self):
        return self.layer

class MockProject(object):
    def __init__(self, add_tree_control=False):
        self.window = MockWindow()
        self.task = MockTask(self.window)
        self.layer_canvas = MockCanvas()
        self.layer_manager = LayerManager.create(self)
        if add_tree_control:
            self.layer_tree_control = MockTree()
            self.layer_manager.insert_layer([2], self.layer_tree_control.layer)

    def raw_load_all_layers(self, uri, mime):
        guess = FileGuess(uri)
        guess.metadata.mime = mime
        print guess
        print guess.metadata
        loader, layers = loaders.load_layers(guess.metadata, manager=self.layer_manager)
        return layers

    def raw_load_first_layer(self, uri, mime):
        return self.raw_load_all_layers(uri, mime)[0]
    
    def load_file(self, path, mime):
        guess = FileGuess(os.path.realpath(path))
        guess.metadata.mime = mime
        metadata = guess.get_metadata()
        print metadata
        loader = loaders.get_loader(metadata)
        print loader
        batch_flags = BatchStatus()
        if hasattr(loader, "load_project"):
            print "FIXME: Add load project command that clears all layers"
            loader.load_project(metadata, self.layer_manager, batch_flags)
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
            return cmd
        return None
    
    def undo(self, count=1):
        while count > 0:
            undo = self.layer_manager.undo_stack.undo(self)
            self.process_flags(undo.flags)
            count -= 1
    
    def redo(self, count=1):
        while count > 0:
            undo = self.layer_manager.undo_stack.redo(self)
            self.process_flags(undo.flags)
            count -= 1
    
    def process_command(self, command, new_mouse_mode=None, override_editable_properties_changed=None):
        print "processing command %s" % command.short_name
        undo = self.layer_manager.undo_stack.perform(command, self)
        self.process_flags(undo.flags)
        return undo
    
    def process_flags(self, f):
        print "in process_flags"
    
    def get_outer_boundary(self, layer):
        boundaries = Boundaries(layer)
        return boundaries.get_outer_boundary()

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
