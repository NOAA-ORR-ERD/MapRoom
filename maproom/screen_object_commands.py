
from .command import Command, UndoInfo

import logging
log = logging.getLogger(__name__)


class MoveStickyLayerCommand(Command):
    short_name = "move_sticky"
    serialize_order = [
        ('layer', 'layer'),
        ('dx', 'float'),
        ('dy', 'float'),
    ]

    def __init__(self, layer, dx, dy):
        Command.__init__(self, layer)
        self.dx = dx
        self.dy = dy
        self.start_x_percentage = layer.x_percentage
        self.start_y_percentage = layer.y_percentage

    def __str__(self):
        return "Move Layer"

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer:
                self.dx += next_command.dx
                self.dy += next_command.dy
                return True

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        lf = undo.flags.add_layer_flags(layer)
        undo.data = layer.get_undo_info()

        layer.move_layer(self.start_x_percentage, self.start_y_percentage, self.dx, self.dy)
        editor.update_info_panels(layer, True)
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        layer.restore_undo_info(self.undo_info)
        return self.undo_info
