# Enthought library imports.
from traits.api import HasTraits, Int, Any, List, Set, Bool, Event

OP_ADD_POINT = 1  # params = None
OP_DELETE_POINT = 2  # params = ( ( x, y ), z, color, state )
OP_ADD_LINE = 3  # params = None
OP_DELETE_LINE = 4  # params = ( point_index_1, point_index_2, color, state )
OP_MOVE_POINT = 5  # params = ( d_lon, d_lat )
OP_CHANGE_POINT_DEPTH = 6  # params = ( old_depth, new_depth )
OP_ADD_TRIANGLE = 7  # params = ( point_index_1, point_index_2, point_index_3, color, state )
OP_DELETE_TRIANGLE = 8  # params = ( point_index_1, point_index_2, point_index_3, color, state )

import logging
log = logging.getLogger(__name__)

class LayerUndo(HasTraits):
    
    #### Traits definitions
    
    # the undo stack is a list of objects, where each object is of the form:
    # { "n" : <int operation number>, "op" : <int operation enum>, "l" :
    # <layer>, "i" : <point or line index in layer>, "p" : ( params depending
    # on op enum ) }
    undo_stack = List(Any)
    
    undo_stack_next_index = Int
    
    undo_stack_next_operation_number = Int
    
    undo_stack_changed = Event

    def start_batch_events(self):
        self.batch = True
        self.events = []
    
    def end_batch_events(self):
        self.batch = False
        order = []
        seen = set()
        for event in self.events:
            if event in seen:
                continue
            seen.add(event)
            order.append(event)
        for name, value in order:
            setattr(self, name, value)
    
    def dispatch_event(self, event, value=True):
        log.debug("batch=%s: dispatching event %s = %s" % (self.batch, event, value))
        if self.batch:
            self.events.append((event, value))
        else:
            setattr(self, event, value)

    def add_undo_operation_to_operation_batch(self, op, layer, index, params):
        self.clear_undo_stack_forward()
        self.undo_stack.append({"n": self.undo_stack_next_operation_number, "op": op, "l": layer, "i": index, "p": params})
        self.undo_stack_next_index = len(self.undo_stack)

    def end_operation_batch(self, refresh=True):
        self.show_undo_redo_debug_dump("end_operation_batch()")
        self.undo_stack_next_operation_number += 1
        if refresh:
            self.dispatch_event('refresh_needed')

    def delete_undo_operations_for_layer(self, layer):
        self.clear_undo_stack_forward()
        new_stack = []
        for o in self.undo_stack:
            if (o["l"] != layer):
                new_stack.append(o)
        self.undo_stack = new_stack
        self.undo_stack_next_index = len(self.undo_stack)

    def clear_undo_stack_forward(self):
        if (len(self.undo_stack) > self.undo_stack_next_index):
            self.undo_stack = self.undo_stack[0: self.undo_stack_next_index]

    def get_current_undoable_operation_text(self):
        if (self.undo_stack_next_index == 0):
            return ""

        op = self.undo_stack[self.undo_stack_next_index - 1]["op"]

        return self.get_undo_redo_operation_text(op)

    def get_current_redoable_operation_text(self):
        if (self.undo_stack_next_index >= len(self.undo_stack)):
            return ""

        op = self.undo_stack[self.undo_stack_next_index]["op"]

        return self.get_undo_redo_operation_text(op)

    def get_undo_redo_operation_text(self, op):
        if (op == OP_ADD_POINT or op == OP_ADD_LINE or op == OP_ADD_TRIANGLE):
            return "Add"
        elif (op == OP_DELETE_POINT or op == OP_DELETE_LINE or op == OP_DELETE_TRIANGLE):
            return "Delete"
        elif (op == OP_MOVE_POINT):
            return "Move"
        elif (op == OP_CHANGE_POINT_DEPTH):
            return "Depth Change"

        return ""

    def undo(self):
        if (self.undo_stack_next_index == 0):
            return
        operation_number = self.undo_stack[self.undo_stack_next_index - 1]["n"]
        # here we assume that all operations in the batch are actually from the same layer
        # we also assume that as point and line deletions are undone, the objects come back already in selected state,
        # which is true because they had to be in selected state at the time they were deleted
        layer = self.undo_stack[self.undo_stack_next_index - 1]["l"]
        layer.clear_all_selections()
        # walk backward until we get to a different operation number or hit the start of the stack
        affected_layers = set()
        while (True):
            if (self.undo_stack_next_index == 0 or self.undo_stack[self.undo_stack_next_index - 1]["n"] != operation_number):
                break
            self.undo_operation(self.undo_stack[self.undo_stack_next_index - 1], affected_layers)
            self.undo_stack_next_index -= 1
        log.debug("affected layers: %s" % str(affected_layers))
        for layer in affected_layers:
            self.dispatch_event('layer_contents_changed', layer)
        self.show_undo_redo_debug_dump("undo() done")
        self.undo_stack_changed = True

    def undo_operation(self, o, affected_layers):
        operation_number = o["n"]
        op = o["op"]
        layer = o["l"]
        index = o["i"]
        params = o["p"]

        affected_layers.add(layer)

        if (op == OP_ADD_POINT):
            layer.delete_point(index, False)
        elif (op == OP_ADD_LINE):
            layer.delete_line_segment(index, False)
        elif (op == OP_DELETE_POINT):
            ((x, y), z, color, state) = params
            layer.insert_point_at_index(index, (x, y), z, color, state, False)
        elif (op == OP_DELETE_LINE):
            (point_index_1, point_index_2, color, state) = params
            layer.insert_line_segment_at_index(index, point_index_1, point_index_2, color, state, False)
        elif (op == OP_MOVE_POINT):
            (world_d_x, world_d_y) = params
            layer.offset_point(index, world_d_x, world_d_y, False)
        elif (op == OP_CHANGE_POINT_DEPTH):
            (old_depth, new_depth) = params
            layer.points.z[index] = old_depth
        elif (op == OP_ADD_TRIANGLE):
            layer.delete_triangle(index, False)
        elif (op == OP_DELETE_TRIANGLE):
            layer.insert_triangle_at_index(index, params, False)

    def redo(self):
        if (self.undo_stack_next_index >= len(self.undo_stack)):
            return
        operation_number = self.undo_stack[self.undo_stack_next_index]["n"]
        # walk forward until we get to a different operation number or hit the end of the stack
        affected_layers = set()
        while (True):
            if (self.undo_stack_next_index == len(self.undo_stack) or self.undo_stack[self.undo_stack_next_index]["n"] != operation_number):
                break
            self.redo_operation(self.undo_stack[self.undo_stack_next_index], affected_layers)
            self.undo_stack_next_index += 1
        for layer in affected_layers:
            self.dispatch_event('layer_contents_changed', layer)
        self.show_undo_redo_debug_dump("redo() done")
        self.undo_stack_changed = True

    def redo_operation(self, o, affected_layers):
        operation_number = o["n"]
        op = o["op"]
        layer = o["l"]
        index = o["i"]
        params = o["p"]

        affected_layers.add(layer)

        if (op == OP_ADD_POINT):
            ((x, y), z, color, state) = params
            layer.insert_point_at_index(index, (x, y), z, color, state, False)
        elif (op == OP_ADD_LINE):
            (point_index_1, point_index_2, color, state) = params
            layer.insert_line_segment_at_index(index, point_index_1, point_index_2, color, state, False)
        elif (op == OP_DELETE_POINT):
            layer.delete_point(index, False)
        elif (op == OP_DELETE_LINE):
            layer.delete_line_segment(index, False)
        elif (op == OP_MOVE_POINT):
            (world_d_x, world_d_y) = params
            layer.offset_point(index, -world_d_x, -world_d_y, False)
        elif (op == OP_CHANGE_POINT_DEPTH):
            (old_depth, new_depth) = params
            layer.points.z[index] = new_depth
        elif (op == OP_ADD_TRIANGLE):
            layer.insert_triangle_at_index(index, params, False)
        elif (op == OP_DELETE_TRIANGLE):
            layer.delete_triangle(index, False)

    def show_undo_redo_debug_dump(self, location_message):
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(location_message + ": the undo_stack is now: ")
            if (len(self.undo_stack) <= 100):
                for item in self.undo_stack:
                    print "    " + str(item)
            else:
                print "    longer than 100 items"
            print "undo_stack_next_index is now: " + str(self.undo_stack_next_index)
