import os
import os.path
import time
import sys
import math

import numpy as np

import wx

# Enthought library imports.
from traits.api import on_trait_change, Unicode, Str, Any, Float, Bool, Int
from pyface.api import YES

from ..library import rect
from ..mouse_commands import MoveControlPointCommand
from ..renderer import color_floats_to_int, int_to_color_floats, int_to_html_color_string, alpha_from_int, ImageData

from line import LineLayer
from constants import *

import logging
log = logging.getLogger(__name__)


class VectorObjectLayer(LineLayer):
    """Layer for a vector object
    
    Vector objects have control points (the points that can be moved by the
    user) and rasterized points, the points created by the control points
    in order to generate whatever shape is needed.  E.g.  a simple spline
    might have 4 control points, but only those endpoints on the drawn object
    actually go through the control points so the rasterized points are
    computed from those control points.
    
    The self.points array contains the control points
    """
    name = Unicode("Vector Object Layer")

    type = Str("vector_object")
    
    mouse_mode_toolbar = Str("AnnotationLayerToolBar")
    
    rebuild_needed = Bool(False)
    
    def set_layer_style_defaults(self):
        self.style.line_color = self.manager.default_style.line_color

    def set_visibility_when_selected(self, layer_visibility):
        layer_visibility['points'] = True

    def clear_visibility_when_deselected(self, layer_visibility):
        layer_visibility['points'] = False
    
    def rebuild_image(self):
        """Hook for image-based renderers to rebuild image data
        
        """
        pass
    
    def rebuild_renderer(self, in_place=False):
        """Update renderer
        
        """
        projected_point_data = self.compute_projected_point_data()
        r, g, b, a = int_to_color_floats(self.style.line_color)
        point_color = color_floats_to_int(r, g, b, 1.0)
#        self.rasterize(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        self.rasterize(projected_point_data, self.points.z, point_color, self.style.line_color)
        self.rebuild_image()
        self.rebuild_needed = False

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        """Renders the outline of the vector object.
        
        If the vector object subclass is fillable, subclass from
        FillableVectorObject instead of this base class.
        """
        log.log(5, "Rendering vector object %s!!! visible=%s, pick=%s" % (self.name, layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return
        if self.rebuild_needed:
            self.rebuild_renderer()
        self.renderer.outline_object(layer_index_base, picker, self.style)
        if layer_visibility["points"] or picker.is_active:
            self.renderer.draw_points(layer_index_base, picker, self.point_size)

    def render_control_points_only(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        """Renders the outline of the vector object.
        
        If the vector object subclass is fillable, subclass from
        FillableVectorObject instead of this base class.
        """
        log.log(5, "Rendering vector object control points %s!!!" % (self.name))
        if (not layer_visibility["layer"]):
            return
        self.renderer.draw_points(layer_index_base, picker, self.point_size)


class LineVectorObject(VectorObjectLayer):
    """Line uses 3 control points in the self.points array.  The midpoint is an
    additional control point, which is constrained and not independent of the
    ends.  This is used as the control point when translating.
    
    """
    name = Unicode("Line")
    
    type = Str("line_obj")
    
    layer_info_panel = ["Layer name", "Line Style", "Line Width", "Line Color", "Start Marker", "End Marker", "Line Transparency"]
    
    selection_info_panel = ["Point coordinates"]

    corners_from_flat = np.asarray((0, 1, 2, 3), dtype=np.uint8)
    lines = np.asarray(((0, 1),), dtype=np.uint8)
    num_corners = 2
    center_point_index = 2
    display_center_control_point = False
    
    # return the anchor point of the index point. E.g. anchor_of[0] = 1
    anchor_of = np.asarray((1, 0, 2), dtype=np.uint8)
    
    # anchor modification array: apply dx,dy values to each control point based
    # on the anchor point.  Used when moving/resizing
    anchor_dxdy = np.asarray((
        ((0,0), (1,1), (0.5,0.5)), # anchor point is 0 (drag point is 1)
        ((1,1), (0,0), (0.5,0.5)), # anchor point is 1, etc.
        ((1,1), (1,1), (1,1)), # center point acts as rigid move
        ), dtype=np.float32)

    def set_opposite_corners(self, p1, p2):
        p = np.concatenate((p1, p2), 0)  # flatten to 1D
        c = p[self.corners_from_flat].reshape(-1,2)
        cp = self.get_control_points_from_corners(c)
        self.set_data(cp, 0.0, self.lines)
    
    def copy_control_point_from(self, cp, other_layer, other_cp):
        log.log(5, "copy control point from %s %s to %s %s" % (other_layer.name, other_cp, self.name, cp))
        x = self.points.x[cp]
        y = self.points.y[cp]
        x1 = other_layer.points.x[other_cp]
        y1 = other_layer.points.y[other_cp]
        self.move_control_point(cp, self.anchor_of[cp], x1 - x, y1 - y)
    
    def get_control_points_from_corners(self, c):
        num_cp = self.center_point_index + 1
        cp = np.empty((num_cp,2), dtype=np.float32)
        cp[0:self.num_corners] = c
        cp[self.center_point_index] = c.mean(0)
        self.compute_constrained_control_points(cp)
        return cp
    
    def compute_constrained_control_points(self, cp):
        pass

    def find_anchor_of(self, point_index):
        if point_index > self.center_point_index:
            self.anchor_point = point_index
        else:
            self.anchor_point = self.anchor_of[point_index]

    def set_anchor_point(self, point_index, maintain_aspect=False):
        self.clear_all_selections()
        self.select_point(point_index)
        self.drag_point = point_index
        self.find_anchor_of(point_index)
        
    def dragging_selected_objects(self, world_dx, world_dy, snapped_layer, snapped_cp):
        cmd = MoveControlPointCommand(self, self.drag_point, self.anchor_point, world_dx, world_dy, snapped_layer, snapped_cp)
        return cmd
    
    def move_control_point(self, drag, anchor, dx, dy):
        """Moving the control point changes the size of the bounding rectangle.
        
        Assuming the drag point is one of the corners and the anchor is the
        opposite corner, the points are constrained as follows: the drag point
        moves by both dx & dy.  The anchor point doesn't move at all, and of
        the other points: one only uses dx and the other dy.
        """
        if drag == anchor and drag != self.center_point_index:
            self.move_polyline_point(anchor, dx, dy)
        else:
            self.move_bounding_box_point(drag, anchor, dx, dy)
    
    def move_polyline_point(self, anchor, dx, dy):
        pass
    
    def remove_from_master_control_points(self, drag, anchor):
        # if the item is moved and it's linked to a master control point,
        # detatch it.  Moving dependent points will not update the master
        # point.
        if drag != anchor:
            remove = drag
        else:
            remove = -1
        return self.manager.remove_control_point_links(self, remove)
    
    def move_bounding_box_point(self, drag, anchor, dx, dy):
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        old_origin = np.copy(p.xy[0])  # without copy it will be changed below
        orig_wh = p.xy[self.anchor_of[0]] - old_origin

        scale = self.anchor_dxdy[anchor]
        xoffset = scale.T[0] * dx
        yoffset = scale.T[1] * dy
        
        # Only scale the bounding box control points & center because
        # subclasses may have additional points in the list
        offset = self.center_point_index + 1
        
        # FIXME: Why does specifying .x work for a range, but not for a single
        # element? Have to use the dict notation for a single element.
        self.points[0:offset].x += xoffset
        self.points[0:offset].y += yoffset
        
        new_origin = np.copy(p.xy[0])  # see above re use of copy
        scaled_wh = p.xy[self.anchor_of[0]] - new_origin
        scale = scaled_wh / orig_wh
        self.rescale_after_bounding_box_change(old_origin, new_origin, scale)
    
    def rescale_after_bounding_box_change(self, old_origin, new_origin, scale):
        pass
    
    def rasterize(self, projected_point_data, z, cp_color, line_color):
        n = np.alen(self.points)
        if not self.display_center_control_point:
            n -= 1
        colors = np.empty(n, dtype=np.uint32)
        colors.fill(cp_color)
        self.renderer.set_points(projected_point_data, z, colors, num_points=n)
        colors = np.empty(np.alen(self.line_segment_indexes), dtype=np.uint32)
        colors.fill(line_color)
        self.renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], colors)

    def get_marker_points(self):
        """Return a tuple of point indexes for each marker.
        
        The first index is the point where the marker will be drawn.  The
        second is the other end of the line which is used to align the marker
        in the proper direction.
        """
        return ((0, 1, self.style.line_start_marker),
                (1, 0, self.style.line_end_marker))

    def render_screen(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        """Marker rendering occurs in screen coordinates
        
        It doesn't scale with the image, it scales with the line size on screen
        """
        if (not layer_visibility["layer"] or picker.is_active):
            return
        log.log(5, "Rendering markers!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        c = self.renderer.canvas
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        markers = []
        for start, end, marker in self.get_marker_points():
            markers.append((p[start]['xy'], p[end]['xy'], marker))
        self.renderer.draw_screen_markers(markers, self.style)


class FillableVectorObject(LineVectorObject):
    # Fillable objects should (in general) display their center control point
    display_center_control_point = True
    
    def set_layer_style_defaults(self):
        self.style.line_color = self.manager.default_style.line_color
        self.style.fill_color = self.manager.default_style.fill_color

    def remove_from_master_control_points(self, drag, anchor):
        # linked control points only possible with lines, so skip the test to
        # save time
        return []

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        log.log(5, "Rendering vector object %s!!! visible=%s, pick=%s" % (self.name, layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return
        if self.rebuild_needed:
            self.rebuild_renderer()
        self.renderer.fill_object(layer_index_base, picker, self.style)
        self.renderer.outline_object(layer_index_base, picker, self.style)
        if layer_visibility["points"] or picker.is_active:
            self.renderer.draw_points(layer_index_base, picker, self.point_size)
            

class RectangleMixin(object):
    """Rectangle uses 4 control points in the self.points array, and nothing in
    the polygon points array.  All corner points can be used as control points.
    
    The center is an additional control point, which is constrained and not
    independent of the corners.
    
     3     6     2
      o----o----o
      |         |
    7 o    o 8  o 5
      |         |
      o----o----o
     0     4     1
    """
    layer_info_panel = ["Layer name", "Line Style", "Line Width", "Line Color", "Line Transparency", "Fill Style", "Fill Color", "Fill Transparency"]
    
    selection_info_panel = ["Point coordinates"]

    corners_from_flat = np.asarray((0, 1, 2, 1, 2, 3, 0, 3), dtype=np.uint8)
    lines = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.uint8)
    num_corners = 4
    center_point_index = 8
    
    # return the anchor point of the index point. E.g. anchor_of[0] = 2
    anchor_of = np.asarray((2, 3, 0, 1, 6, 7, 4, 5, 8), dtype=np.uint8)
    
    # anchor modification array: apply dx,dy values to each control point based
    # on the anchor point.  Used when moving/resizing
    anchor_dxdy = np.asarray((
        ((0,0), (1,0), (1,1), (0,1), (.5,0), (1,.5), (.5,1), (0,.5), (.5,.5)), # anchor point is 0 (drag point is 2)
        ((1,0), (0,0), (0,1), (1,1), (.5,0), (0,.5), (.5,1), (1,.5), (.5,.5)), # anchor point is 1 (drag is 3)
        ((1,1), (0,1), (0,0), (1,0), (.5,1), (0,.5), (.5,0), (1,.5), (.5,.5)), # anchor point is 2, etc.
        ((0,1), (1,1), (1,0), (0,0), (.5,1), (1,.5), (.5,0), (0,.5), (.5,.5)),
        ((0,0), (0,0), (0,1), (0,1), (0,0), (0,.5), (0,1), (0,.5), (0,.5)), # edges start here
        ((1,0), (0,0), (0,0), (1,0), (.5,0), (0,0), (.5,0), (1,0), (.5,0)),
        ((0,1), (0,1), (0,0), (0,0), (0,1), (0,.5), (0,0), (0,.5), (0,.5)),
        ((0,0), (1,0), (1,0), (0,0), (.5,0), (1,0), (.5,0), (0,0), (.5,0)),
        ((1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1)), # center point acts as rigid move
        ), dtype=np.float32)
    
    def compute_constrained_control_points(self, cp):
        x1 = cp[0,0]
        x2 = cp[1,0]
        xm = (x1 + x2)*.5
        y1 = cp[0,1]
        y2 = cp[2,1]
        ym = (y1 + y2)*.5
        cp[4] = (xm, y1)
        cp[5] = (x2, ym)
        cp[6] = (xm, y2)
        cp[7] = (x1, ym)

class RectangleVectorObject(RectangleMixin, FillableVectorObject):
    name = Unicode("Rectangle")
    
    type = Str("rectangle_obj")
    
    def get_marker_points(self):
        return []


class EllipseVectorObject(RectangleVectorObject):
    """Rectangle uses 4 control points in the self.points array, and nothing in
    the polygon points array.  All corner points can be used as control points.
    
    """
    name = Unicode("Ellipse")
    
    type = Str("ellipse_obj")
    
    def rasterize(self, projected_point_data, z, cp_color, line_color):
        colors = np.empty(np.alen(self.points), dtype=np.uint32)
        colors.fill(cp_color)
        self.renderer.set_points(projected_point_data, z, colors)
        p = projected_point_data
        
        # FIXME: this only supports axis aligned ellipses
        width = p[1][0] - p[0][0]
        height = p[2][1] - p[1][1]
        sx = width / 2
        sy = height / 2
        cx = p[self.center_point_index][0]
        cy = p[self.center_point_index][1]
         
        num_segments = 128
        xy = np.zeros((num_segments, 2), dtype=np.float32)
        
        dtheta = 2 * 3.1415926 / num_segments
        theta = 0.0
        x = sx # we start at angle = 0 
        y = 0
        i = 0
        while i < num_segments:
            xy[i] = (cx + sx*math.cos(theta), cy + sy*math.sin(theta))
            theta += dtheta
            i += 1
        
        # create line segment list from one point to the next
        i1 = np.arange(num_segments, dtype=np.uint32)
        i2 = np.arange(1, num_segments+1, dtype=np.uint32)
        i2[-1] = 0
        lsi = np.vstack((i1, i2)).T  # zip arrays to get line segment indexes
        
        # set_lines expects a color list for each point, not a single color
        colors = np.empty(num_segments, dtype=np.uint32)
        colors.fill(line_color)
        self.renderer.set_lines(xy, lsi, colors)


class ScaledImageObject(RectangleVectorObject):
    """Texture mapped image object that scales to the lat/lon view
    
    Image uses 4 control points like the rectangle, but uses a texture
    object as the foreground.  The background color will show through in
    trasparent pixels.
    
    """
    name = Unicode("Image")
    
    type = Str("scaled_image_obj")
    
    layer_info_panel = ["Layer name", "Transparency"]
    
    image_data = Any
    
    def get_image_array(self):
        from maproom.library.numpy_images import get_square
        return get_square(100)
    
    def move_control_point(self, drag, anchor, dx, dy):
        RectangleVectorObject.move_control_point(self, drag, anchor, dx, dy)
        if self.image_data is not None:
            projection = self.manager.project.layer_canvas.projection
            self.image_data.set_control_points(self.points, projection)
            self.renderer.set_image_projection(self.image_data, projection)

    def rebuild_image(self):
        """Update renderer
        
        """
        projection = self.manager.project.layer_canvas.projection
        if self.image_data is None:
            raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[0], raw.shape[1])
            self.image_data.load_numpy_array(self.points, raw, projection)
        self.renderer.set_image_projection(self.image_data, projection)

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        """Renders the outline of the vector object.
        
        If the vector object subclass is fillable, subclass from
        FillableVectorObject instead of this base class.
        """
        log.log(5, "Rendering vector object %s!!! visible=%s, pick=%s" % (self.name, layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return
        if self.rebuild_needed:
            self.rebuild_renderer()
        alpha = alpha_from_int(self.style.line_color)
        self.renderer.draw_image(layer_index_base, picker, alpha)
        if layer_visibility["points"] or picker.is_active:
            self.renderer.draw_points(layer_index_base, picker, self.point_size)


class OverlayImageObject(RectangleVectorObject):
    """Texture mapped image object that is fixed in size relative to the screen
    
    Image uses 4 control points like the rectangle, but uses a texture
    object as the foreground.  The background color will show through in
    trasparent pixels.
    
    """
    name = Unicode("Overlay Image")
    
    type = Str("overlay_image_obj")
    
    layer_info_panel = ["Layer name", "Transparency"]
    
    image_data = Any
    
    screen_offset_from_center = np.asarray(
        ((-0.5,-0.5), (0.5,-0.5), (0.5,0.5), (-0.5,0.5), (0,-0.5), (0.5,0), (0,0.5), (-0.5,0), (0,0)),
        dtype=np.float32)

    def get_image_array(self):
        from maproom.library.numpy_images import get_numpy_from_marplot_icon
        return get_numpy_from_marplot_icon('marplot_drum.png')

    def set_location(self, p1):
        p = np.concatenate((p1, p1), 0)  # flatten to 1D
        c = p[self.corners_from_flat].reshape(-1,2)
        cp = self.get_control_points_from_corners(c)
        self.set_data(cp, 0.0, self.lines)
    
    def move_control_point(self, drag, anchor, dx, dy):
        RectangleVectorObject.move_control_point(self, drag, anchor, dx, dy)
        if self.image_data is not None:
            projection = self.manager.project.layer_canvas.projection
            self.image_data.set_control_points(self.points, projection)
            self.renderer.set_image_projection(self.image_data, projection)
    
    def update_world_control_points(self):
        pass

    def rebuild_image(self):
        """Update renderer
        
        """
        if self.rebuild_needed:
            self.renderer.release_textures()
            self.image_data = None
        if self.image_data is None:
            raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[0], raw.shape[1])
            self.image_data.load_numpy_array(self.points, raw)
        self.renderer.set_image_screen(self.image_data)

    def pre_render(self):
        if self.rebuild_needed:
            self.rebuild_renderer()
        self.update_world_control_points()

    def render_screen(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        """Marker rendering occurs in screen coordinates
        
        It doesn't scale with the image, it scales with the line size on screen
        """
        if (not layer_visibility["layer"]):
            return
        log.log(5, "Rendering overlay image!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        c = self.renderer.canvas
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        center = c.get_numpy_screen_point_from_world_point(p[self.center_point_index]['xy'])
        self.renderer.image_textures.center_at_screen_point(self.image_data, center, rect.height(c.screen_rect))
        self.render_overlay(w_r, p_r, s_r, layer_visibility, layer_index_base, picker)
    
    def render_overlay(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        alpha = alpha_from_int(self.style.line_color)
        self.renderer.draw_image(layer_index_base, picker, alpha)

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        # without this, the superclass method from VectorObjectLayer will get
        # called too
        pass

class OverlayTextObject(OverlayImageObject):
    """Texture mapped image object that is fixed in size relative to the screen
    
    Image uses 4 control points like the rectangle, but uses a texture
    object as the foreground.  The background color will show through in
    trasparent pixels.
    
    """
    name = Unicode("Text")
    
    type = Str("overlay_text_obj")
    
    user_text = Unicode("<b>New Label</b>")
    
    text_width = Float(200)
    
    text_height = Float(50)
    
    border_width = Int(10)
    
    layer_info_panel = ["Layer name", "Text Color", "Font", "Font Size", "Text Transparency", "Line Style", "Line Width", "Line Color", "Line Transparency", "Fill Style", "Fill Color", "Fill Transparency"]
    
    selection_info_panel = ["Text Format", "Overlay Text"]
    
    def user_text_to_json(self):
        return self.user_text.encode("utf-8")

    def user_text_from_json(self, json_data):
        self.user_text = json_data['user_text'].decode('utf-8')
    
    def text_width_to_json(self):
        return self.text_width

    def text_width_from_json(self, json_data):
        self.text_width = json_data['text_width']
    
    def text_height_to_json(self):
        return self.text_height

    def text_height_from_json(self, json_data):
        self.text_height = json_data['text_height']
    
    def set_style(self, style):
        OverlayImageObject.set_style(self, style)
        self.rebuild_needed = True  # Force rebuild to change image color
    
    def get_image_array(self):
        from maproom.library.numpy_images import OffScreenHTML
        h = OffScreenHTML()
        c = int_to_html_color_string(self.style.text_color)
        arr = h.get_numpy(self.user_text, c, self.style.font, self.style.font_size, self.style.text_format, self.text_width)
        return arr

    def update_world_control_points(self):
        h, w = self.text_height + (2 * self.border_width), self.text_width + (2 * self.border_width)  # numpy image dimensions are reversed
        c = self.renderer.canvas
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        center = c.get_numpy_screen_point_from_world_point(p[self.center_point_index]['xy'])
        
        scale = self.screen_offset_from_center.T
        xoffset = scale[0] * w + center[0]
        yoffset = scale[1] * h + center[1]
        
        for i in range(self.center_point_index):
            w = c.get_numpy_world_point_from_screen_point((xoffset[i], yoffset[i]))
            # p[i]['xy'] = w  # Doesn't work!
            self.points.x[i] = w[0]
            self.points.y[i] = w[1]
        
        projected_point_data = self.compute_projected_point_data()
        self.renderer.set_points(projected_point_data, None, None)
        self.renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], None)
    
    def move_control_point(self, drag, anchor, dx, dy):
        # Note: center point drag is rigid body move so text box size is only
        # recalculated if dragging some other control point
        if drag < self.center_point_index:
            c = self.renderer.canvas
            p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
            d = np.copy(p.xy[drag])
            d += (dx, dy)
            a = np.copy(p.xy[anchor])
            
            d_s = c.get_numpy_screen_point_from_world_point(d)
            a_s = c.get_numpy_screen_point_from_world_point(a)

            if drag < self.num_corners:
                # Dragging a corner changes both width and heiht
                self.text_width = abs(d_s[0] - a_s[0]) - (2 * self.border_width)
                self.text_height = abs(d_s[1] - a_s[1]) - (2 * self.border_width)
            else:
                # Dragging an edge only changes one dimension
                oc = self.screen_offset_from_center[drag]
                if abs(oc[1]) > 0:
                    self.text_height = abs(d_s[1] - a_s[1]) - (2 * self.border_width)
                else:
                    self.text_width = abs(d_s[0] - a_s[0]) - (2 * self.border_width)

        self.move_bounding_box_point(drag, anchor, dx, dy)
    
    def render_overlay(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        self.renderer.prepare_to_render_projected_objects()
        self.renderer.fill_object(layer_index_base, picker, self.style)
        self.renderer.outline_object(layer_index_base, picker, self.style)
        if layer_visibility["points"] or picker.is_active:
            self.renderer.draw_points(layer_index_base, picker, self.point_size)
        self.renderer.prepare_to_render_screen_objects()
        alpha = alpha_from_int(self.style.text_color)
        self.renderer.draw_image(layer_index_base, picker, alpha)

class OverlayIconObject(OverlayImageObject):
    """Texture mapped Marplot icon object that is fixed in size relative to the screen
    
    Uses the Marplot category icons.
    """
    name = Unicode("Icon")
    
    type = Str("overlay_icon_obj")
    
    layer_info_panel = ["Layer name", "Marplot Icon", "Color", "Transparency"]
    
    def get_image_array(self):
        return self.style.get_numpy_image_from_icon()
    
    def set_style(self, style):
        OverlayImageObject.set_style(self, style)
        self.rebuild_needed = True  # Force rebuild to change image color


class PolylineObject(RectangleMixin, FillableVectorObject):
    """Polyline uses 4 control points in the self.points array as the control
    points for the bounding box, one center point, and subsequent points as
    the list of points that define the segmented line.
    
    Adjusting the corner control points will resize or move the entire
    polyline.  The center is an additional control point, which is constrained
    and not independent of the corners.  Note that the control points that
    represent the line don't have to start or end at one of the corners; the
    bounding box points are calculated every time a point is added or removed
    from the polyline.
    
     3           2
      o---------o
      |         |
      |    o 4  |
      |         |
      o---------o
     0           1
    """
    name = Unicode("Polyline")
    
    type = Str("polyline_obj")
    
    def set_points(self, points):
        points = np.asarray(points)
        
        # initialize boundary box control points with zeros; will be filled in
        # with call to recalc_bounding_box below
        cp = np.zeros((self.center_point_index + 1,2), dtype=np.float32)
        
        p = np.concatenate((cp, points), 0)  # flatten to 1D
        num_points = np.alen(points)
        offset = self.center_point_index + 1
        lines = zip(range(offset, offset + num_points - 1), range(offset + 1, offset + num_points))
        self.set_data(p, 0.0, np.asarray(lines, dtype=np.uint32))
        self.recalc_bounding_box()
    
    def move_polyline_point(self, anchor, dx, dy):
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy
        points[anchor] += (dx, dy)
        self.recalc_bounding_box()
    
    def recalc_bounding_box(self):
        offset = self.center_point_index + 1
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy
        r = rect.get_rect_of_points(points[offset:])
        corners = np.empty((4,2), dtype=np.float32)
        corners[0] = r[0]
        corners[1] = (r[1][0], r[0][1])
        corners[2] = r[1]
        corners[3] = (r[0][0], r[1][1])
        cp = self.get_control_points_from_corners(corners)
        points[0:offset] = cp
        self.update_bounds()
    
    def rescale_after_bounding_box_change(self, old_origin, new_origin, scale):
        offset = self.center_point_index + 1
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        points = ((p.xy[offset:] - old_origin) * scale) + new_origin
        p.xy[offset:] = points
        
    def rasterize(self, projected_point_data, z, cp_color, line_color):
        n = np.alen(self.points)
        if not self.display_center_control_point:
            n -= 1
        colors = np.empty(n, dtype=np.uint32)
        colors.fill(cp_color)
        self.renderer.set_points(projected_point_data, z, colors, num_points=n)
        colors = np.empty(np.alen(self.line_segment_indexes), dtype=np.uint32)
        colors.fill(line_color)
        self.renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], colors)

    def get_marker_points(self):
        # Markers are only used on the first and last segments of the line
        indexes = self.line_segment_indexes
        if len(indexes) > 0:
            return (
                (indexes.point1[0], indexes.point2[0], self.style.line_start_marker),
                (indexes.point2[-1], indexes.point1[-1], self.style.line_end_marker))
