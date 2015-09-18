import os

from peppy2.utils.runtime import get_all_subclasses


class UndoStack(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.insert_index = 0
        self.save_point_index = 0
        self.batch = self
    
    def is_dirty(self):
        return self.insert_index != self.save_point_index
    
    def set_save_point(self):
        self.save_point_index = self.insert_index
    
    def perform(self, cmd, editor):
        if cmd is None:
            return UndoInfo()
        undo_info = cmd.perform(editor)
        if undo_info.flags.success:
            self.add_command(cmd)
        cmd.last_flags = undo_info.flags
        return undo_info

    def can_undo(self):
        return self.insert_index > 0
    
    def get_undo_command(self):
        if self.can_undo():
            return self[self.insert_index - 1]
    
    def undo(self, editor):
        cmd = self.get_undo_command()
        if cmd is None:
            return UndoInfo()
        undo_info = cmd.undo(editor)
        if undo_info.flags.success:
            self.insert_index -= 1
        cmd.last_flags = undo_info.flags
        return undo_info
    
    def can_redo(self):
        return self.insert_index < len(self)
    
    def get_redo_command(self):
        if self.can_redo():
            return self[self.insert_index]
    
    def redo(self, editor):
        cmd = self.get_redo_command()
        if cmd is None:
            return UndoInfo()
        undo_info = cmd.perform(editor)
        if undo_info.flags.success:
            self.insert_index += 1
        cmd.last_flags = undo_info.flags
        return undo_info
    
    def start_batch(self):
        if self.batch == self:
            self.batch = Batch()
    
    def end_batch(self):
        self.batch = self

    def add_command(self, command):
        self.batch.insert_at_index(command)
    
    def insert_at_index(self, command):
        last = self.get_undo_command()
        if last is not None and last.coalesce(command):
            return
        if command.is_recordable():
            self[self.insert_index:] = [command]
            self.insert_index += 1

    def pop_command(self):
        last = self.get_undo_command()
        if last is not None:
            self.insert_index -= 1
            self[self.insert_index:self.insert_index + 1] = []
        return last
    
    def history_list(self):
        h = [str(c) for c in self]
        return h
    
    def serialize(self):
        from serializer import Serializer
        s = Serializer()
        for c in self:
            s.add(c)
        return s
    
    def unserialize_text(self, text, manager):
        from serializer import TextDeserializer
        
        offset = manager.get_invariant_offset()
        s = TextDeserializer(text, offset)
        for cmd in s.iter_cmds(manager):
            yield cmd
    
    def find_most_recent(self, cmdcls):
        for cmd in reversed(self):
            if isinstance(cmd, cmdcls):
                return cmd
        return None


class LayerStatus(object):
    def __init__(self, layer):
        self.layer = layer  # FIXME: should be a weakref (or maybe token)
        
        # True if items within the layer have changed position only
        self.layer_items_moved = False
        
        # True if items have been added to the layer
        self.layer_contents_added = False
        
        # True if items have been removed from the layer
        self.layer_contents_deleted = False
        
        # True if drawing properties changed (color, line width, etc.)
        self.layer_display_properties_changed = False
        
        # True if layer properties changed (layer name, etc.)
        self.layer_metadata_changed = False
        
        # True if this layer was loaded and needs to be broadcast to all views
        self.layer_loaded = False
        
        # True if a message should be displayed if it isn't the top layer and
        # it was edited
        self.hidden_layer_check = False
        
        # ...needs to be selected after the command finishes
        self.select_layer = False


class BatchStatus(object):
    def __init__(self):
        # list of all layers processed
        self.layers = []
        
        # rebuild flags for each layer; value is whether or not it needs full
        # refresh (False) or in-place, fast refresh (True)
        self.need_rebuild = {}
        
        # The last layer marked as selected will show up here
        self.select_layer = None
        
        self.layers_changed = False
        
        self.metadata_changed = False
        
        # If any editable properties have changed (those show up in the
        # InfoPanel)
        self.editable_properties_changed = False
        
        self.refresh_needed = False
        
        self.fast_viewport_refresh_needed = False
        
        # Any (error) messages will be added to this list
        self.messages = []


class CommandStatus(object):
    def __init__(self):
        # True if command successfully completes, must set to False on failure
        self.success = True
        
        # List of errors encountered
        self.errors = []
        
        # Message displayed to the user
        self.message = None
        
        # True if all controls & map window need to be redrawn
        self.refresh_needed = False
        
        # True if only map window redraw needed
        self.fast_viewport_refresh_needed = False
        
        # True if projection changed on any layer
        self.projection_changed = False
        
        # True if layers have been added, removed, renamed or reordered in the
        # layer manager
        self.layers_changed = False
        
        self.layer_flags = []
    
    def add_layer_flags(self, layer):
        lf = LayerStatus(layer)
        self.layer_flags.append(lf)
        return lf


class UndoInfo(object):
    def __init__(self):
        self.index = -1
        self.data = None
        self.flags = CommandStatus()
    
    def __str__(self):
        return "index=%d, flags=%s" % (self.index, str(dir(self.flags)))
    
    def affected_layers(self):
        layers = []
        for lf in self.flags.layer_flags:
            layers.append(lf.layer)
        return layers


class Command(object):
    short_name = None
    serialize_order = [
        ('layer', 'layer'),
        ]
    
    def __init__(self, layer=None):
        # Instead of storing a reference to the layer itself, an invariant
        # is stored instead.  This invariant is basically a counter in the
        # layer manager that uniquely identifies the layer in the order it
        # was loaded.  Referencing by name doesn't work because layers can be
        # renamed and therefore not unique.  Storing the layer itself results
        # in problems when a layer gets removed and then re-added (by an undo
        # then redo) because a new layer is created for that command.  The
        # layers saved in redos beyond that command are pointing to the old
        # layer, not the newly created layer, so stepping into the future
        # operates on the old layer that isn't displayed.
        if layer is not None:
            self.layer = layer.invariant
        else:
            self.layer = None
        
        self.undo_info = None
        self.last_flags = None

    def __str__(self):
        return "<unnamed command>"
    
    def get_serialized_name(self):
        if self.short_name is None:
            return self.__class__.__name__
        return self.short_name
    
    def coalesce(self, next_command):
        return False
    
    def is_recordable(self):
        return True
    
    def perform(self, editor):
        pass
    
    def undo(self, editor):
        pass


class Batch(Command):
    """A batch is immutable once created, so there's no need to allow
    intermediate index points.
    """
    def __str__(self):
        return "<batch>"
    
    def perform(self, editor):
        for c in self:
            c.perform(editor)
    
    def undo(self, editor):
        for c in reversed(self):
            c.undo(editor)


def get_known_commands():
    s = get_all_subclasses(Command)
    c = {}
    for cls in s:
        if cls.short_name is not None:
            c[cls.short_name] = cls
    return c
