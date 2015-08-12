class NullPicker(object):
    is_active = False
    
    @classmethod
    def get_picker_index_base(cls, layer_index):
        return layer_index
    
    def prepare_to_render(self, screen_rect):
        pass

    def done_rendering(self):
        pass

    def render_picker_to_screen(self):
        pass

    def bind_picker_colors_for_lines(self, layer_index, object_count):
        pass

    def bind_picker_colors_for_points(self, layer_index, object_count):
        pass

    def bind_picker_colors(self, layer_index, object_count, doubled=False):
        pass

    def unbind_picker_colors(self, layer_index, object_count, doubled=False):
        pass

    def get_object_at_mouse_position(self, screen_point):
        """returns ( layer_index, object_index ) or None
        """
        return None
    
    @staticmethod
    def is_ugrid_point(obj):
        return False
    
    @staticmethod
    def is_ugrid_point_type(type):
        return False

    @staticmethod
    def is_ugrid_line(obj):
        return False

    @staticmethod
    def is_polygon_fill(obj):
        return False

    @staticmethod
    def parse_clickable_object(o):
        return (None, None, None, None)
