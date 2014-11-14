import Point_and_line_set_renderer
import Polygon_set_renderer
from layer_renderer import LayerRenderer

def is_ugrid_point(obj):
    (layer_index, type, subtype, object_index) = parse_clickable_object(obj)
    #
    return type == LayerRenderer.POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET and subtype == Point_and_line_set_renderer.POINTS_SUB_LAYER_PICKER_OFFSET

def is_ugrid_line(obj):
    (layer_index, type, subtype, object_index) = parse_clickable_object(obj)
    #
    return type == LayerRenderer.POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET and subtype == Point_and_line_set_renderer.LINES_SUB_LAYER_PICKER_OFFSET

def is_polygon_fill(self):
    (layer_index, type, subtype, object_index) = parse_clickable_object(obj)
    #
    return type == LayerRenderer.POLYGONS_SUB_LAYER_PICKER_OFFSET and subtype == Polygon_set_renderer.FILL_SUB_LAYER_PICKER_OFFSET

def is_polygon_point(self):
    (layer_index, type, subtype, object_index) = parse_clickable_object(obj)
    #
    return type == LayerRenderer.POLYGONS_SUB_LAYER_PICKER_OFFSET and subtype == Point_and_line_set_renderer.POINTS_SUB_LAYER_PICKER_OFFSET

def parse_clickable_object(o):
    if (o is None):
        return (None, None, None, None)

    # see Layer.py for layer types
    # see Point_and_line_set_renderer.py and Polygon_set_renderer.py for subtypes
    ## fixme: OMG! I can't believe how hard-coded this is!!!
    (layer_index, object_index) = o
    type_and_subtype = layer_index % 10
    type = (type_and_subtype / 5) * 5
    subtype = type_and_subtype % 5
    layer_index = layer_index / 10
    # print str( obj ) + "," + str( ( layer_index, type, subtype ) )
    #
    return (layer_index, type, subtype, object_index)
