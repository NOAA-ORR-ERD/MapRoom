
from sawx.utils.runtime import get_all_subclasses

import logging
log = logging.getLogger(__name__)


class UndoStack(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.insert_index = 0
        self.save_point_index = 0
        self.batch = self

    def debug_structure(self, indent=""):
        s = self.serialize()
        lines = ["command history:"]
        for c in s.serialized_commands:
            lines.append(str(c))
        return ("\n" + indent).join(lines)

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

    @property
    def can_undo(self):
        return self.insert_index > 0

    def get_undo_command(self):
        if self.can_undo:
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

    @property
    def can_redo(self):
        return self.insert_index < len(self)

    def get_redo_command(self):
        if self.can_redo:
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
        from .serializer import Serializer
        s = Serializer()
        for c in self:
            s.add(c)
        return s

    def unserialize_text(self, text, manager):
        from .serializer import TextDeserializer

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

        # List of points that have been changed, if command supports it
        self.indexes_of_points_affected = []

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

        # True if the item should be shown collapsed in the tree list
        self.collapse = False

    def __str__(self):
        flags = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            val = getattr(self, name)
            if val is None or not val or hasattr(val, "__call__"):
                continue
            flags.append("%s=%s" % (name, val))
        return ", ".join(flags)


class BatchStatus(object):
    def __init__(self):
        # list of all layers processed
        self.layers = []

        # rebuild flags for each layer; value is whether or not it needs full
        # refresh (False) or in-place, fast refresh (True)
        self.need_rebuild = {}

        # force the layer to be collapsed when inserted into the tree list
        self.collapse = {}

        # The last layer marked as selected will show up here
        self.select_layer = None

        self.layers_changed = False

        self.metadata_changed = False

        # If any editable properties have changed (those show up in the
        # InfoPanel)
        self.editable_properties_changed = False

        self.refresh_needed = False

        self.immediate_refresh_needed = False

        # Any info messages will be added to this list
        self.messages = []

        # Any error messages will be added to this list
        self.errors = []

    def __str__(self):
        flags = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            val = getattr(self, name)
            if val is None or not val or hasattr(val, "__call__"):
                continue
            flags.append("%s=%s" % (name, val))
        return ", ".join(flags)

    __repr__ = __str__


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
        self.immediate_refresh_needed = False

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

    def __str__(self):
        flags = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            val = getattr(self, name)
            if val is None or not val or hasattr(val, "__call__"):
                continue
            flags.append("%s=%s" % (name, val))
        return ", ".join(flags)


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

    def get_layer_in_layer_manager(self, layer_manager):
        return layer_manager.get_layer_by_invariant(self.layer)

    def get_serialized_name(self):
        if self.short_name is None:
            return self.__class__.__name__
        return self.short_name

    def can_coalesce(self, next_command):
        return False

    def coalesce_merge(self, next_command):
        raise NotImplementedError

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__ and next_command.layer == self.layer and self.can_coalesce(next_command):
            self.coalesce_merge(next_command)
            return True

    def is_recordable(self):
        return True

    def perform_on_layer(self, editor, layer, lm, lf):
        raise NotImplementedError

    def perform_on_parent(self, editor, layer, lm, lf):
        return self.perform_on_layer(editor, layer, lm, lf)

    def get_affected_layers(self, layer, lm):
        """If any layers are affected by the change to this layer, return them
        here in the order that they should be changed.

        Defaults to all descendants of the layer.
        """
        return lm.get_layer_descendants(layer)

    def perform(self, editor):
        lm = editor.layer_manager
        self.undo_info = undo = UndoInfo()
        undo.data = []
        layer = lm.get_layer_by_invariant(self.layer)
        lf = undo.flags.add_layer_flags(layer)
        log.debug("%s: perform_on_parent: %s" % (self, layer))
        layer_undo_info = self.perform_on_parent(editor, layer, lm, lf)
        undo.data.append((layer.invariant, layer_undo_info))
        log.debug("%s: parent undo: %s" % (self, undo.data[-1]))
        children = self.get_affected_layers(layer, lm)
        for layer in children:
            log.debug("%s: perform_on_layer: %s" % (self, layer))
            lf = undo.flags.add_layer_flags(layer)
            layer_undo_info = self.perform_on_layer(editor, layer, lm, lf)
            undo.data.append((layer.invariant, layer_undo_info))
            log.debug("%s: layer undo: %s" % (self, undo.data[-1]))
        return undo

    def undo_on_layer(self, editor, layer, lm, layer_undo_info):
        raise NotImplementedError

    def undo_on_parent(self, editor, layer, lm, layer_undo_info):
        return self.undo_on_layer(editor, layer, lm, lf)

    def undo(self, editor):
        lm = editor.layer_manager

        # Do the layers in reverse order, as that seems the most common case.
        # Special cases will require subclassing
        for invariant, layer_undo_info in reversed(self.undo_info.data):
            layer = lm.get_layer_by_invariant(invariant)
            if invariant == self.layer:
                self.undo_on_parent(editor, layer, lm, layer_undo_info)
            else:
                self.undo_on_layer(editor, layer, lm, layer_undo_info)
        return self.undo_info


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
