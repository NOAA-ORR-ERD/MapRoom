import os

from sawx.loader import identify_file

from . import loaders
from .styles import LayerStyle
from .layer_manager import LayerManager
from .layers import Layer
from .library.Boundary import Boundaries
from .library.projection import Projection
from .library.host_utils import HostCache
from .servers import default_tile_hosts
from .command import UndoStack, BatchStatus
from .menu_commands import LoadLayersCommand


class MockApplication(object):
    command_line_args = []


class MockFrame(object):
    def status_message(self, message, debug=False):
        pass


class MockCanvas(object):
    def __init__(self):
        self.projection = Projection("+proj=merc +units=m +over")


class MockWindow(object):
    def __init__(self):
        self.application = MockApplication()

    def error(self, *args, **kwargs):
        pass


class MockPreferences(object):
    def __init__(self):
        pass


class MockTask(object):
    def __init__(self, window, default_styles=None):
        self.window = window
        if default_styles is None:
            default_styles = {"other": LayerStyle()}
        self.default_styles = default_styles
        HostCache.set_known_hosts(default_tile_hosts)
        self.preferences = MockPreferences()

    def get_tile_server_id_from_url(self, url):
        index, host = HostCache.get_host_by_url(url)
        return index

    def get_tile_server_by_id(self, id):
        return HostCache.get_known_hosts()[id]


class MockTree(object):
    def __init__(self, manager):
        self.layer = Layer(manager)

    def get_edit_layer(self):
        return self.layer


class MockProject(object):
    def __init__(self, add_tree_control=False, default_styles=None):
        self.window = MockWindow()
#        self.task = MockTask(self.window, default_styles)
        self.layer_canvas = MockCanvas()
        self.layer_manager = LayerManager(None)
        if add_tree_control:
            self.layer_tree_control = MockTree(self.layer_manager)
            self.layer_manager.insert_layer([2], self.layer_tree_control.layer)
        self.frame = MockFrame()

    @property
    def current_layer(self):
        return self.layer_tree_control.get_edit_layer()

    def raw_load_all_layers(self, uri, mime):
        file_metadata = identify_file(os.path.realpath(uri))
        print(file_metadata)
        loader = file_metadata["loader"]
        layers = loader.load_layers(uri, manager=self.layer_manager)
        return layers

    def raw_load_first_layer(self, uri, mime):
        return self.raw_load_all_layers(uri, mime)[0]

    def load_file(self, path, mime):
        file_metadata = identify_file(os.path.realpath(path))
        print(file_metadata)
        loader = file_metadata["loader"]
        batch_flags = BatchStatus()
        if hasattr(loader, "load_project"):
            print("FIXME: Add load project command that clears all layers")
            loader.load_project(file_metadata["uri"], self.layer_manager, batch_flags)
        elif hasattr(loader, "iter_log"):
            line = 0
            for cmd in loader.iter_log(metadata, self.layer_manager):
                line += 1
                errors = None
                if cmd.short_name == "load":
                    print(cmd.metadata)
                    if cmd.metadata.uri.startswith("TestData"):
                        cmd.metadata.uri = "../" + cmd.metadata.uri
                try:
                    undo = self.process_command(cmd)
                    if not undo.flags.success:
                        errors = undo.errors
                        break
                except Exception:
                    # errors = [str(e)]
                    # break
                    raise
            if errors is not None:
                text = "\n".join(errors)
                raise RuntimeError(text)
        else:
            cmd = LoadLayersCommand(file_metadata["uri"], loader)
            self.process_command(cmd)
            return cmd
        return None

    def load_success(self, uri):
        pass

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
        print("processing command %s" % command.short_name)
        undo = self.layer_manager.undo_stack.perform(command, self)
        self.process_flags(undo.flags)
        return undo

    def process_flags(self, f):
        print("in process_flags")

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
        print(guess)
        print(guess.metadata)
        loader, layers = loaders.load_layers(guess.metadata, manager=self)
        return layers

    def load_first_layer(self, uri, mime):
        return self.load_all_layers(uri, mime)[0]

    def get_outer_boundary(self, layer):
        boundaries = Boundaries(layer)
        return boundaries.get_outer_boundary()

    def get_default_style_for(self, layer):
        return LayerStyle()
