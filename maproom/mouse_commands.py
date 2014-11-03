from command import Command

class InsertPointCommand(Command):
    def __init__(self, layer, world_point):
        self.layer = layer
        self.world_point = world_point
        self.undo_info = None
    
    def __str__(self):
        return "Add Point #%d" % self.undo_info.index
    
    def perform(self, editor):
        self.undo_info = self.layer.insert_point(self.world_point)
        self.layer.select_point(self.undo_info.index)
        self.layer.update_bounds()
        vis = editor.layer_visibility[self.layer]['layer']
        if not vis:
            self.undo_info.message = "Added point to hidden layer %s" % layer.name
        return self.undo_info

    def undo(self, editor):
        undo_info = self.layer.delete_point(self.undo_info.index)
        self.layer.update_bounds()
        return undo_info
