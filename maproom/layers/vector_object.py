import math

import numpy as np

from ..library import rect
from ..library.coordinates import haversine, distance_bearing, haversine_at_const_lat, haversine_list, km_to_rounded_string, km2_to_rounded_string, mi_to_rounded_string, mi2_to_rounded_string
from ..library.Boundary import Boundary
from ..renderer import color_floats_to_int, int_to_color_floats, int_to_color_uint8, int_to_html_color_string, alpha_from_int, ImageData, data_types
from .. import styles

from .line import LineLayer
from .folder import BoundedFolder

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

    NOTE: Each subclass of VectorObjectLayer must have a unique string used for
    the class attribute 'type'.  This string is what's used when deserializing
    from a saved project file, and if multiple objects use the same string,
    the class used for deserializing will be randomly chosen, which is bad.
    """
    name = "Vector Object"

    type = "vector_object"

    mouse_mode_toolbar = "AnnotationLayerToolBar"

    has_control_points = True

    use_color_cycling = False

    center_point_index = 0

    display_center_control_point = False

    control_point_color = color_floats_to_int(0, 0, 0, 1.0)

    control_point_names = [""]

    def __init__(self, manager):
        super().__init__(manager)

        self.rotation = 0.0
        self.border_width = 10

    def __str__(self):
        return LineLayer.__str__(self) + ", bb: %s" % str(self.bounds)

    @property
    def can_rotate(self):
        return False

    def can_copy(self):
        return True

    def can_reparent_to(self, potential_parent_layer):
        return potential_parent_layer.can_contain_annotations

    def check_for_problems(self):
        pass

    def rotation_to_json(self):
        return self.rotation

    def rotation_from_json(self, json_data):
        # Ignore when rotation isn't present
        try:
            self.rotation = json_data['rotation']
        except KeyError:
            self.rotation = 0.0

    def border_width_to_json(self):
        return self.border_width

    def border_width_from_json(self, json_data):
        # Ignore when border width isn't present
        try:
            self.border_width = json_data['border_width']
        except KeyError:
            self.border_width = 10  # FIXME: add application preference

# control point links are not used when restoring from disk, only on copied
# layers when pasting them back.

#    def control_point_links_to_json(self):
#        return self.control_point_links

    def control_point_links_from_json(self, json_data):
        try:
            self.control_point_links = json_data['control_point_links']
        except KeyError:
            raise TypeError("optional control_point_links data not present; skipping")

    def get_control_point_link(self, point_index):
        for dep_cp, truth_inv, truth_cp, locked in self.manager.get_control_point_links(self):
            if point_index == dep_cp:
                return self.manager.get_layer_by_invariant(truth_inv), truth_cp
        return None, None

    def get_info_panel_text(self, prop):
        if prop == "Path length":
            km = self.calculate_distances()
            return "%s, %s" % (km_to_rounded_string(km), mi_to_rounded_string(km * .621371))
        return LineLayer.get_info_panel_text(self, prop)

    def set_border_width(self, width):
        self.border_width = width

    def set_visibility_when_selected(self, layer_visibility):
        layer_visibility['points'] = True

    def clear_visibility_when_deselected(self, layer_visibility):
        layer_visibility['points'] = False

    def has_points(self):
        # Points on vector layers aren't individually editable
        return False

    def has_selection(self):
        # "selection" on a vector layer is coerced to mean that when a vector
        # layer is selected in the layer tree control, the entire layer is
        # selected
        return True

    def delete_all_selected_objects(self):
        from ..menu_commands import DeleteLayerCommand
        return DeleteLayerCommand(self)

    def calculate_distances(self, cp):
        return 0.0

    def children_affected_by_move(self):
        """ Returns a list of layers that will be affected by moving a control
        point.  This is used for layer groups; moving a control point of a
        group will affect all the layers in the group.
        """
        return [self]

    def parents_affected_by_move(self):
        """ Returns a list of layers that might need to have boundaries
        recalculated after moving this layer.
        """
        affected = self.manager.get_layer_parents(self)
        return affected

    def rebuild_image(self, renderer):
        """Hook for image-based renderers to rebuild image data

        """

    def get_renderer_colors(self):
        """Hook to allow subclasses to override style colors
        """
        line_color = self.style.line_color
        r, g, b, a = int_to_color_floats(line_color)
        point_color = color_floats_to_int(r, g, b, 1.0)
        return point_color, line_color

    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer

        """
        projected_point_data = self.compute_projected_point_data(renderer.canvas.projection)
        r, g, b, a = int_to_color_floats(self.style.line_color)
        point_color, line_color = self.get_renderer_colors()
#        self.rasterize(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        self.rasterize(renderer, projected_point_data, self.points.z, point_color, line_color)
        self.rebuild_image(renderer)
        self.rebuild_needed = False

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        """Renders the outline of the vector object.

        If the vector object subclass is fillable, subclass from
        FillableVectorObject instead of this base class.
        """
        log.log(5, "Rendering vector object %s!!! pick=%s" % (self.name, picker))
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        renderer.outline_object(self, picker, self.style)

    def render_control_points_only(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        """Renders the outline of the vector object.

        If the vector object subclass is fillable, subclass from
        FillableVectorObject instead of this base class.
        """
        log.log(5, "Rendering vector object control points %s!!!" % (self.name))
        renderer.draw_points(self, picker, self.point_size)


class LineVectorObject(VectorObjectLayer):
    """Line uses 3 control points in the self.points array.  The midpoint is an
    additional control point, which is constrained and not independent of the
    ends.  This is used as the control point when translating.

    """
    name = "Line"

    type = "line_obj"

    layer_info_panel = ["Line style", "Line width", "Line color", "Start marker", "End marker"]

    selection_info_panel = ["Anchor latitude", "Anchor longitude", "Path length"]

    corners_from_flat = np.asarray((0, 1, 2, 3), dtype=np.uint8)
    lines = np.asarray(((0, 1),), dtype=np.uint8)
    num_corners = 2
    center_point_index = 2

    control_point_names = ["start", "end", "middle"]

    # return the anchor point of the index point, the point that should remain
    # in place when dragging the index point. So, for instance, dragging
    # control point zero would use zero as the index value into this array,
    # e.g. anchor_of[0] = 1. This means control point 1 remains in place as
    # control point 0 is dragged.
    anchor_of = np.asarray((1, 0, 2), dtype=np.uint8)

    # anchor modification array: apply dx,dy values to each control point based
    # on the anchor point.  Used when moving/resizing
    anchor_dxdy = np.asarray((
        ((0, 0), (1, 1), (0.5, 0.5)),  # anchor point is 0 (drag point is 1)
        ((1, 1), (0, 0), (0.5, 0.5)),  # anchor point is 1, etc.
        ((1, 1), (1, 1), (1, 1)),  # center point acts as rigid move
    ), dtype=np.float32)

    def calculate_distances(self):
        return haversine_list(self.points[0:self.num_corners])

    def set_opposite_corners(self, p1, p2, update_bounds=True):
        p = np.concatenate((p1, p2), 0)  # flatten to 1D
        c = p[self.corners_from_flat].reshape(-1, 2)
        cp = self.get_control_points_from_corners(c)
        self.set_data(cp, 0.0, self.lines, update_bounds)

    def copy_control_point_from(self, cp, other_layer, other_cp):
        log.log(5, "copy control point from %s %s to %s %s" % (other_layer.name, other_cp, self.name, cp))
        x = self.points.x[cp]
        y = self.points.y[cp]
        x1 = other_layer.points.x[other_cp]
        y1 = other_layer.points.y[other_cp]
        self.move_control_point(cp, self.anchor_of[cp], x1 - x, y1 - y)

    def get_control_points_from_corners(self, c):
        num_cp = self.center_point_index + 1
        cp = np.empty((num_cp, 2), dtype=np.float32)
        cp[0:self.num_corners] = c
        cp[self.center_point_index] = c.mean(0)
        self.compute_constrained_control_points(cp)
        return cp

    def compute_constrained_control_points(self, cp):
        pass

    def find_nearest_corner(self, world_pt):
        x = np.copy(self.points.x[0:self.num_corners]) - world_pt[0]
        y = np.copy(self.points.y[0:self.num_corners]) - world_pt[1]
        d = (x * x) + (y * y)
        cp = np.argmin(d)
        return int(cp)

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
        self.set_initial_rotation()

    def set_initial_rotation(self):
        self.initial_rotation = self.rotation

    def dragging_selected_objects(self, world_dx, world_dy, snapped_layer, snapped_cp, about_center=False):
        from ..vector_object_commands import MoveControlPointCommand
        cmd = MoveControlPointCommand(self, self.drag_point, self.anchor_point, world_dx, world_dy, snapped_layer, snapped_cp, about_center)
        return cmd

    def move_control_point(self, drag, anchor, dx, dy, about_center=False):
        """Moving the control point changes the size of the bounding rectangle.

        Assuming the drag point is one of the corners and the anchor is the
        opposite corner, the points are constrained as follows: the drag point
        moves by both dx & dy.  The anchor point doesn't move at all, and of
        the other points: one only uses dx and the other dy.
        """
        if drag == anchor and drag != self.center_point_index:
            self.move_polyline_point(anchor, dx, dy)
        else:
            self.move_bounding_box_point(drag, anchor, dx, dy, about_center)

    def move_polyline_point(self, anchor, dx, dy):
        pass

    def remove_from_master_control_points(self, drag, anchor, force=False):
        # if the item is moved and it's linked to a master control point,
        # detatch it.  Moving dependent points will not update the master
        # point.
        if drag != anchor:
            remove = drag
        else:
            remove = -1
        return self.manager.remove_control_point_links(self, remove, force)

    def move_bounding_box_point(self, drag, anchor, dx, dy, about_center=False, ax=0.0, ay=0.0):
        """ Adjust points within object after bounding box has been resized

        Returns a list of affected layers (child layers can be resized as a
        side effect!)
        """
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)

        if about_center:
            dx *= 2
            dy *= 2
            orig_center = np.copy(p.xy[self.center_point_index])

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

        # Optionally, the anchor point can also move, so scale again if needed
        scale = self.anchor_dxdy[drag]
        xoffset = scale.T[0] * ax
        yoffset = scale.T[1] * ay
        self.points[0:offset].x += xoffset
        self.points[0:offset].y += yoffset

        if about_center:
            new_center = np.copy(p.xy[self.center_point_index])
            self.points[0:offset].x -= new_center[0] - orig_center[0]
            self.points[0:offset].y -= new_center[1] - orig_center[1]

        # force children to fit inside the new bounding box
        temp_bounds = self.compute_bounding_rect_from_points(self.points[0:offset])
        self.fit_to_bounding_box(self.bounds, temp_bounds)

    def fit_to_bounding_box(self, current_bounds, new_bounds):
        pass

    def rotating_selected_objects(self, world_dx, world_dy):
        from ..vector_object_commands import RotateObjectCommand
        cmd = RotateObjectCommand(self, self.drag_point, world_dx, world_dy)
        return cmd

    def rotate_point(self, drag, dx, dy):
        """Rotate the object (about the center) using the x & y offset from the
        dragged point to calculate the angle of rotation
        """
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        c = p[self.center_point_index]
        p1 = p.xy[drag] - c.xy
        a1 = np.arctan2(*p1)
        p2 = p1 + (dx, dy)
        a2 = np.arctan2(*p2)
        delta = (a2 - a1) * 180.0 / np.pi
        self.rotation = self.initial_rotation + delta
        self.update_bounds_after_rotation()

    def update_bounds_after_rotation(self):
        pass

    def rasterize_points(self, renderer, projected_point_data, z, cp_color):
        n = np.alen(self.points)
        if not self.display_center_control_point:
            n -= 1
        colors = np.empty(n, dtype=np.uint32)
        colors.fill(cp_color)
        num_cp = self.center_point_index
        if self.display_center_control_point:
            num_cp += 1
        colors[0:num_cp] = self.control_point_color
        renderer.set_points(projected_point_data, z, colors, num_points=n)

    def rotate_points(self, projected_point_data, center=None):
        if center is None:
            center = projected_point_data[self.center_point_index]
        r = self.rotation / 180.0 * np.pi
        rot = np.array(((np.cos(r), -np.sin(r)), (np.sin(r), np.cos(r))), dtype=np.float32)
        p = np.dot(projected_point_data - center, rot) + center
        return p

    def rasterize_lines(self, renderer, projected_point_data, colors):
        points = self.rotate_points(projected_point_data)
        renderer.set_lines(points, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], colors)

    def rasterize(self, renderer, projected_point_data, z, cp_color, line_color):
        self.rasterize_points(renderer, projected_point_data, z, cp_color)
        colors = np.empty(np.alen(self.line_segment_indexes), dtype=np.uint32)
        colors.fill(line_color)
        self.rasterize_lines(renderer, projected_point_data, colors)

    def get_marker_points(self):
        """Return a tuple of point indexes for each marker.

        The first index is the point where the marker will be drawn.  The
        second is the other end of the line which is used to align the marker
        in the proper direction.
        """
        return ((0, 1, self.style.line_start_marker),
                (1, 0, self.style.line_end_marker))

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        """Marker rendering occurs in screen coordinates

        It doesn't scale with the image, it scales with the line size on screen
        """
        log.log(5, "Rendering markers!!! pick=%s" % (picker))
        if picker.is_active:
            # don't render markers on the pick screen: these points don't have
            # a corresponding entry into the pick_layer_index_map
            return
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        markers = []
        for start, end, marker in self.get_marker_points():
            markers.append((p[start]['xy'], p[end]['xy'], marker))
        renderer.draw_screen_markers(markers, self.style)


class FillableVectorObject(LineVectorObject):
    name = "Fillable"

    type = "fillable_obj"

    # Fillable objects should (in general) display their center control point
    display_center_control_point = True

    def remove_from_master_control_points(self, drag, anchor):
        # linked control points only possible with lines, so skip the test to
        # save time
        return []

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering vector object %s!!! pick=%s" % (self.name, picker))
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        renderer.fill_object(self, picker, self.style)
        renderer.outline_object(self, picker, self.style)


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
    corners_from_flat = np.asarray((0, 1, 2, 1, 2, 3, 0, 3), dtype=np.uint8)
    lines = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.uint8)
    num_corners = 4
    center_point_index = 8

    control_point_names = ["lower left", "lower right", "upper right", "upper left", "center bottom", "center right", "center top", "center left", "center"]

    # return the anchor point of the index point. E.g. anchor_of[0] = 2
    anchor_of = np.asarray((2, 3, 0, 1, 6, 7, 4, 5, 8), dtype=np.uint8)

    # anchor modification array: apply dx,dy values to each control point based
    # on the anchor point.  Used when moving/resizing
    anchor_dxdy = np.asarray((
        ((0, 0), (1, 0), (1, 1), (0, 1), (.5, 0), (1, .5), (.5, 1), (0, .5), (.5, .5)),  # anchor point is 0 (drag point is 2)
        ((1, 0), (0, 0), (0, 1), (1, 1), (.5, 0), (0, .5), (.5, 1), (1, .5), (.5, .5)),  # anchor point is 1 (drag is 3)
        ((1, 1), (0, 1), (0, 0), (1, 0), (.5, 1), (0, .5), (.5, 0), (1, .5), (.5, .5)),  # anchor point is 2, etc.
        ((0, 1), (1, 1), (1, 0), (0, 0), (.5, 1), (1, .5), (.5, 0), (0, .5), (.5, .5)),
        ((0, 0), (0, 0), (0, 1), (0, 1), (0, 0), (0, .5), (0, 1), (0, .5), (0, .5)),  # edges start here
        ((1, 0), (0, 0), (0, 0), (1, 0), (.5, 0), (0, 0), (.5, 0), (1, 0), (.5, 0)),
        ((0, 1), (0, 1), (0, 0), (0, 0), (0, 1), (0, .5), (0, 0), (0, .5), (0, .5)),
        ((0, 0), (1, 0), (1, 0), (0, 0), (.5, 0), (1, 0), (.5, 0), (0, 0), (.5, 0)),
        ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),  # center point acts as rigid move
    ), dtype=np.float32)

    def compute_constrained_control_points(self, cp):
        x1 = cp[0, 0]
        x2 = cp[1, 0]
        xm = (x1 + x2) * .5
        y1 = cp[0, 1]
        y2 = cp[2, 1]
        ym = (y1 + y2) * .5
        cp[4] = (xm, y1)
        cp[5] = (x2, ym)
        cp[6] = (xm, y2)
        cp[7] = (x1, ym)


class RectangleVectorObject(RectangleMixin, FillableVectorObject):
    name = "Rectangle"

    type = "rectangle_obj"

    layer_info_panel = ["Line style", "Line width", "Line color", "Fill style", "Fill color"]

    selection_info_panel = ["Anchor latitude", "Anchor longitude", "Width", "Height", "Area"]

    def get_width_height(self):
        if self.empty():
            wkm, hkm = 0.0, 0.0
        else:
            p = self.points
            dlon = p[7].x - p[5].x
            wkm = haversine_at_const_lat(dlon, p[7].y)
            hkm = haversine(p[4].x, p[4].y, p[6].x, p[6].y)
        return wkm, hkm

    def get_info_panel_text(self, prop):
        if prop == "Width":
            wkm, hkm = self.get_width_height()
            return "%s, %s" % (km_to_rounded_string(wkm), mi_to_rounded_string(wkm * .621371))
        elif prop == "Height":
            wkm, hkm = self.get_width_height()
            return "%s, %s" % (km_to_rounded_string(hkm), mi_to_rounded_string(hkm * .621371))
        elif prop == "Area":
            wkm, hkm = self.get_width_height()
            km2 = wkm * hkm
            return "%s, %s" % (km2_to_rounded_string(km2), mi2_to_rounded_string(km2 * .621371 * .621371))
        return FillableVectorObject.get_info_panel_text(self, prop)

    def get_marker_points(self):
        return []

    def has_boundaries(self):
        return True

    def get_all_boundaries(self):
        b = Boundary(self.points, [0, 1, 2, 3, 0], 0.0)
        return [b]

    def get_points_lines(self):
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:4]
        lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return points, lines


class EllipseVectorObject(RectangleVectorObject):
    """Rectangle uses 4 control points in the self.points array, and nothing in
    the polygon points array.  All corner points can be used as control points.

    """
    name = "Ellipse"

    type = "ellipse_obj"

    def get_info_panel_text(self, prop):
        if prop == "Area":
            wkm, hkm = self.get_width_height()
            km = wkm * hkm * math.pi / 4.0
            return "%s, %s" % (km2_to_rounded_string(km), mi2_to_rounded_string(km * .621371 * .621371))
        return RectangleVectorObject.get_info_panel_text(self, prop)

    def get_semimajor_axes(self, p):
        width = p[1][0] - p[0][0]
        height = p[2][1] - p[1][1]
        sx = width / 2
        sy = height / 2
        return sx, sy

    def update_bounds_after_rotation(self):
        # FIXME: Experimental routine to adjust bounding box points after a
        # rotation; doesn't work yet because it needs a way to calculate the
        # semi-major axes from the rotated square, not the axis-aligned square.
        if True:
            return
        # rotated ellipse bbox from http://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse
        log.debug((self.initial_rotation, self.rotation))
        phi = self.rotation * np.pi / 180.0
        sx, sy = self.get_semimajor_axes(self.points)
        ux = sx * math.cos(phi)
        uy = sx * math.sin(phi)
        vx = sy * math.cos(phi + np.pi / 2)
        vy = sy * math.sin(phi + np.pi / 2)
        bbox_halfwidth = math.sqrt(ux * ux + vx * vx)
        bbox_halfheight = math.sqrt(uy * uy + vy * vy)
        dx = bbox_halfwidth - sx
        dy = bbox_halfheight - sy
        self.move_bounding_box_point(2, 0, dx, dy, about_center=True)

    def calc_ellipse(self, cx, cy, sx, sy, num_segments=128):
        xy = np.zeros((num_segments, 2), dtype=np.float32)

        dtheta = 2 * 3.1415926 / num_segments
        theta = 0.0
        i = 0
        while i < num_segments:
            xy[i] = (cx + sx * math.cos(theta), cy + sy * math.sin(theta))
            theta += dtheta
            i += 1

        return xy

    def rasterize(self, renderer, projected_point_data, z, cp_color, line_color):
        self.rasterize_points(renderer, projected_point_data, z, cp_color)
        p = projected_point_data

        # FIXME: this only supports axis aligned ellipses
        sx, sy = self.get_semimajor_axes(p)
        cx = p[self.center_point_index][0]
        cy = p[self.center_point_index][1]

        num_segments = 128
        xy = self.calc_ellipse(cx, cy, sx, sy, num_segments)

        # create line segment list from one point to the next
        i1 = np.arange(num_segments, dtype=np.uint32)
        i2 = np.arange(1, num_segments + 1, dtype=np.uint32)
        i2[-1] = 0
        lsi = np.vstack((i1, i2)).T  # zip arrays to get line segment indexes

        # set_lines expects a color list for each point, not a single color
        colors = np.empty(num_segments, dtype=np.uint32)
        colors.fill(line_color)

        points = self.rotate_points(xy, p[self.center_point_index])
        renderer.set_lines(points, lsi, colors)

    def get_all_boundaries(self):
        cx = self.points[self.center_point_index].x
        cy = self.points[self.center_point_index].y
        sx, sy = self.get_semimajor_axes(self.points)
        xy = self.calc_ellipse(cx, cy, sx, sy)
        indexes = np.arange(len(xy) + 1, dtype=np.int32)
        indexes[-1] = 0
        points = data_types.make_points_from_xy(xy)
        b = Boundary(points, indexes, 0.0)
        return [b]


class CircleVectorObject(EllipseVectorObject):
    """Special case of the ellipse where the object is constrained to be a
    circle on resizing.

    """
    name = "Circle"

    type = "circle_obj"

    selection_info_panel = ["Anchor latitude", "Anchor longitude", "Radius", "Circumference", "Area"]

    def get_radius(self):
        p = self.points
        dlon = p[7].x - p[5].x
        wkm = haversine_at_const_lat(dlon, p[7].y)
        hkm = haversine(p[4].x, p[4].y, p[6].x, p[6].y)
        km = min(wkm, hkm) / 2.0
        return km

    def get_info_panel_text(self, prop):
        if prop == "Radius":
            km = self.get_radius()
            return "%s, %s" % (km_to_rounded_string(km), mi_to_rounded_string(km * .621371))
        elif prop == "Circumference":
            km = self.get_radius() * math.pi * 2.0
            return "%s, %s" % (km_to_rounded_string(km), mi_to_rounded_string(km * .621371))
        elif prop == "Area":
            r = self.get_radius()
            km = r * r * math.pi
            return "%s, %s" % (km2_to_rounded_string(km), mi2_to_rounded_string(km * .621371 * .621371))
        return EllipseVectorObject.get_info_panel_text(self, prop)

    def set_center_and_radius(self, p1, p2):
        lon1, lat1 = p1
        lon2, lat2 = p2
        rkm = haversine(lon1, lat1, lon2, lat2)
        # bearing = math.atan2(math.sin(lon2 - lon1) * math.cos(lat2), math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
        _, lat2 = distance_bearing(lon1, lat1, 0.0, rkm)
        lon2, _ = distance_bearing(lon1, lat1, 90.0, rkm)
        rx = lon2 - lon1
        ry = lat2 - lat1

        c = np.empty((4, 2), dtype=np.float32)
        c[0] = (lon1 - rx, lat1 - ry)
        c[1] = (lon1 + rx, lat1 - ry)
        c[2] = (lon1 + rx, lat1 + ry)
        c[3] = (lon1 - rx, lat1 + ry)
        cp = self.get_control_points_from_corners(c)
        self.set_data(cp, 0.0, self.lines)

    def get_semimajor_axes(self, p):
        width = p[1][0] - p[0][0]
        height = p[2][1] - p[1][1]
        sx = sy = min(width, height) / 2.0
        return sx, sy


class ScaledImageObject(RectangleVectorObject):
    """Texture mapped image object that scales to the lat/lon view

    Image uses 4 control points like the rectangle, but uses a texture
    object as the foreground.  The background color will show through in
    trasparent pixels.

    """
    name = "Image"

    type = "scaled_image_obj"

    layer_info_panel = ["Transparency"]

    def __init__(self, manager):
        super().__init__(manager)
        self.image_data = None

    def get_image_array(self):
        from maproom.library.numpy_images import get_square
        return get_square(100)

    def move_control_point(self, drag, anchor, dx, dy, about_center=False, ax=0.0, ay=0.0):
        self.move_bounding_box_point(self, drag, anchor, dx, dy, about_center, ax, ay)
        if self.image_data is not None:
            c = self.manager.project.layer_canvas
            renderer = c.get_renderer(self)
            projection = c.projection
            self.image_data.set_control_points(self.points, projection)
            renderer.set_image_projection(self.image_data, projection)

    def rebuild_image(self, renderer):
        """Update renderer

        """
        projection = renderer.canvas.projection
        if self.image_data is None:
            raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[1], raw.shape[0])
            self.image_data.load_numpy_array(self.points, raw, projection)
        renderer.set_image_projection(self.image_data, projection)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        """Renders the outline of the vector object.

        If the vector object subclass is fillable, subclass from
        FillableVectorObject instead of this base class.
        """
        log.log(5, "Rendering vector object %s!!! pick=%s" % (self.name, picker))
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        alpha = alpha_from_int(self.style.line_color)
        renderer.draw_image(self, picker, alpha)


class OverlayMixin(object):
    """An overlay is an object that is fixed in size relative to the screen, so
    zooming in and out doesn't change its size.
    """

    @property
    def is_overlay(self):
        return True

    def calc_control_points_from_screen(self, canvas):
        pass

    def update_world_control_points(self, renderer):
        self.calc_control_points_from_screen(renderer.canvas)
        projected_point_data = self.compute_projected_point_data(renderer.canvas.projection)
        renderer.set_points(projected_point_data, None, None)
        renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], None)
        self.update_bounds(True)

    def update_overlay_bounds(self):
        self.bounds = self.compute_bounding_rect()

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility):
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        self.update_world_control_points(renderer)

    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer

        """
        self.update_bounds(False)
        self.update_world_control_points(renderer)
        self.update_bounds(False)
        VectorObjectLayer.rebuild_renderer(self, renderer, in_place)

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        """Marker rendering occurs in screen coordinates

        It doesn't scale with the image, it scales with the line size on screen
        """
        log.log(5, "Rendering overlay image %s!!! pick=%s" % (self.name, picker))
        self.set_overlay_position(renderer)
        self.render_overlay(renderer, w_r, p_r, s_r, layer_visibility, picker)

    def set_overlay_position(self, renderer):
        pass

    def render_overlay(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        """Draw the overlay on screen.

        Must be implemented by subclass
        """
        raise NotImplementedError

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        # without this, the superclass method from VectorObjectLayer will get
        # called too
        pass


class OverlayLineObject(OverlayMixin, LineVectorObject):
    """OverlayLine uses the first control point as the fixed point in world
    coordinate space, and the 2nd point as the offset in screen space so that
    the resulting line is always the same length regardless of zoom

    """
    name = "OverlayLine"

    type = "overlay_line_obj"

    def __init__(self, manager):
        LineVectorObject.__init__(self, manager)

        self.screen_dx = -1
        self.screen_dy = -1

    def get_undo_info(self):
        return (self.copy_points(), self.copy_bounds(), self.screen_dx, self.screen_dy)

    def restore_undo_info(self, info):
        self.points = info[0]
        self.bounds = info[1]
        self.screen_dx = info[2]
        self.screen_dy = info[3]

    def screen_dx_to_json(self):
        return self.screen_dx

    def screen_dx_from_json(self, json_data):
        self.screen_dx = json_data['screen_dx']

    def screen_dy_to_json(self):
        return self.screen_dy

    def screen_dy_from_json(self, json_data):
        self.screen_dy = json_data['screen_dy']

    def set_opposite_corners(self, p1, p2, update_bounds=True):
        LineVectorObject.set_opposite_corners(self, p1, p2, update_bounds)
        c = self.manager.project.layer_canvas
        s1 = c.get_numpy_screen_point_from_world_point(p1)
        s2 = c.get_numpy_screen_point_from_world_point(p2)
        s = s2 - s1
        self.screen_dx, self.screen_dy = s[0], s[1]

    def move_bounding_box_point(self, drag, anchor, dx, dy, about_center=False, ax=0.0, ay=0.0):
        log.debug("OverlayLine: move_bounding_box_point: delta=%s" % str((dx, dy)))
        c = self.manager.project.layer_canvas
        if drag == 1:  # moving screen-space point
            self.points[1].x += dx
            self.points[1].y += dy
            s1 = c.get_numpy_screen_point_from_world_point((self.points[0].x, self.points[0].y))
            s2 = c.get_numpy_screen_point_from_world_point((self.points[1].x, self.points[1].y))
            s = s2 - s1
            log.debug("OverlayLine: Screen point: %s -> %s" % (str(s1), str(s2)))
            self.screen_dx, self.screen_dy = s[0], s[1]
        else:  # moving world-space point
            # move world-space anchor point normally
            self.points[0].x += dx
            self.points[0].y += dy

            # adjust the screen-space point to conform to the screen deltas
            s1 = c.get_numpy_screen_point_from_world_point((self.points[0].x, self.points[0].y))
            s = (s1[0] + self.screen_dx, s1[1] + self.screen_dy)
            w = c.get_numpy_world_point_from_screen_point(s)
            self.points[1].x = w[0]
            self.points[1].y = w[1]


    def fit_to_bounding_box(self, current_bounds, new_bounds):
        # Recalculate screen size based on new bounds. The points array will
        # have already been resized so it's just a matter of adjusting the
        # screen dimensions to match the resizing.
        scale, old_origin, new_origin = rect.get_transform(current_bounds, new_bounds)
        self.screen_dx *= scale[0][0]
        self.screen_dy *= scale[1][1]

    def calc_control_points_from_screen(self, canvas):
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        anchor = canvas.get_numpy_screen_point_from_world_point(p[0]['xy'])
        xs = anchor[0] + self.screen_dx
        ys = anchor[1] + self.screen_dy
        w = canvas.get_numpy_world_point_from_screen_point((xs, ys))
        log.debug("OverlayLine: calc_control: anchor=%s, screen=%s" % (anchor, (xs, ys)))
        # print "world point for anchor %d" % i, w
        # p[i]['xy'] = w  # Doesn't work!
        self.points.x[1] = w[0]
        self.points.y[1] = w[1]
        self.points.x[2] = (self.points[0].x + self.points[1].x) / 2
        self.points.y[2] = (self.points[0].y + self.points[1].y) / 2

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        OverlayMixin.render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker)
        LineVectorObject.render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker)

    def render_overlay(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        pass

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        LineVectorObject.render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker)


class OverlayImageObject(OverlayMixin, RectangleVectorObject):
    """Texture mapped image object that is fixed in size relative to the screen

    Image uses the same control points as the rectangle, but uses a texture
    object as the foreground.  The background color will show through in
    trasparent pixels.

    """
    name = "Overlay Image"

    type = "overlay_image_obj"

    layer_info_panel = ["Transparency"]

    # Screen y coords are backwards from world y coords (screen y increases
    # downward)
    screen_offset_from_center = np.asarray(
        ((-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (0, 0.5), (0.5, 0), (0, -0.5), (-0.5, 0), (0, 0)),
        dtype=np.float32)

    def __init__(self, manager, show_flagged_anchor_point=True):
        RectangleVectorObject.__init__(self, manager)

        self.image_data = None
        self.anchor_point_index = 8  # Defaults to center point as the anchor
        self.show_flagged_anchor_point = show_flagged_anchor_point
        self.init_overlay_attributes()

    def init_overlay_attributes(self):
        return

    def anchor_point_index_to_json(self):
        return self.anchor_point_index

    def anchor_point_index_from_json(self, json_data):
        self.anchor_point_index = json_data['anchor_point_index']

    def show_flagged_anchor_point_to_json(self):
        return self.show_flagged_anchor_point

    def show_flagged_anchor_point_from_json(self, json_data):
        self.show_flagged_anchor_point = json_data.get('show_flagged_anchor_point', False)

    def can_anchor_point_move(self):
        return True

    def get_image_array(self):
        from maproom.library.numpy_images import get_numpy_from_marplot_icon
        return get_numpy_from_marplot_icon('marplot_drum.png')

    def set_location(self, p1):
        p1 = np.asarray(p1)
        p = np.concatenate((p1, p1), 0)  # flatten to 1D
        c = p[self.corners_from_flat].reshape(-1, 2)
        cp = self.get_control_points_from_corners(c)
        self.set_data(cp, 0.0, self.lines)

    def set_anchor_index(self, index):
        if index == self.anchor_point_index:
            return
        self.anchor_point_index = index

    def copy_control_point_from(self, cp, other_layer, other_cp):
        log.debug("copy control point from %s %s to %s %s" % (other_layer.name, other_cp, self.name, cp))
        x = self.points.x[cp]
        y = self.points.y[cp]
        x1 = other_layer.points.x[other_cp]
        y1 = other_layer.points.y[other_cp]
        dx = x1 - x
        dy = y1 - y
        offset = self.center_point_index + 1
        self.points[0:offset].x += dx
        self.points[0:offset].y += dy
        if self.image_data is not None:
            c = self.manager.project.layer_canvas
            renderer = c.get_renderer(self)
            projection = c.projection
            self.image_data.set_control_points(self.points, projection)
            renderer.set_image_projection(self.image_data, projection)

    def move_control_point(self, drag, anchor, dx, dy, about_center=False, ax=0.0, ay=0.0):
        self.move_bounding_box_point(drag, anchor, dx, dy, about_center, ax, ay)
        if self.image_data is not None:
            c = self.manager.project.layer_canvas
            renderer = c.get_renderer(self)
            projection = c.projection
            self.image_data.set_control_points(self.points, projection)
            renderer.set_image_projection(self.image_data, projection)

    def rebuild_image(self, renderer):
        """Update renderer

        """
        if self.rebuild_needed:
            renderer.release_textures()
            self.image_data = None
        if self.image_data is None:
            raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[1], raw.shape[0])
            self.image_data.load_numpy_array(self.points, raw)
        renderer.set_image_screen(self.image_data)

    def set_overlay_position(self, renderer):
        c = renderer.canvas
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        center = c.get_numpy_screen_point_from_world_point(p[self.center_point_index]['xy'])
        renderer.set_image_center_at_screen_point(self.image_data, center, c.screen_rect, 1.0)

    def render_overlay(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        alpha = alpha_from_int(self.style.line_color)
        renderer.draw_image(self, picker, alpha)

    def render_control_points_only(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if self.show_flagged_anchor_point and self.anchor_point_index != self.center_point_index:
            flagged = [self.anchor_point_index]
        else:
            flagged = []
        renderer.draw_points(self, picker, self.point_size, flagged_point_indexes=flagged)


class OverlayScalableImageObject(OverlayImageObject):
    """Texture mapped image object that is fixed in size relative to the screen

    Image uses 4 control points like the rectangle, but uses a texture
    object as the foreground.  The background color will show through in
    trasparent pixels.

    """
    name = "Scalable Image"

    type = "overlay_scalable_image_obj"

    def init_overlay_attributes(self):
        self.text_width = -1
        self.text_height = -1
        self.border_width = 0

    def get_undo_info(self):
        return (self.copy_points(), self.copy_bounds(), self.text_width, self.text_height, self.border_width)

    def restore_undo_info(self, info):
        self.points = info[0]
        self.bounds = info[1]
        self.text_width = info[2]
        self.text_height = info[3]
        self.border_width = info[4]

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

    def set_location_and_size(self, p1, w, h):
        p1 = np.asarray(p1)
        self.text_width = w
        self.text_height = h
        p = np.concatenate((p1, p1), 0)  # flatten to 1D
        c = p[self.corners_from_flat].reshape(-1, 2)
        cp = self.get_control_points_from_corners(c)
        self.set_data(cp, 0.0, self.lines)

    def calc_control_points_from_screen(self, canvas):
        h, w = self.text_height + (2 * self.border_width), self.text_width + (2 * self.border_width)  # array indexes of numpy images are reversed
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        anchor = canvas.get_numpy_screen_point_from_world_point(p[self.anchor_point_index]['xy'])
        # print "anchor (center):", anchor, "text w,h", self.text_width, self.text_height
        anchor_to_center = self.screen_offset_from_center[self.anchor_point_index]

        scale = self.screen_offset_from_center.T
        xoffset = (scale[0] - anchor_to_center[0]) * w + anchor[0]
        yoffset = (scale[1] - anchor_to_center[1]) * h + anchor[1]

        for i in range(self.center_point_index + 1):
            w = canvas.get_numpy_world_point_from_screen_point((xoffset[i], yoffset[i]))
            # print "world point for anchor %d" % i, w
            # p[i]['xy'] = w  # Doesn't work!
            self.points.x[i] = w[0]
            self.points.y[i] = w[1]

    def fit_to_bounding_box(self, current_bounds, new_bounds):
        # Recalculate screen size based on new bounds. The points array will
        # have already been resized so it's just a matter of adjusting the
        # screen dimensions to match the resizing.
        scale, old_origin, new_origin = rect.get_transform(current_bounds, new_bounds)
        self.text_width *= scale[0][0]
        self.text_height *= scale[1][1]
        self.rebuild_needed = True  # Force rebuild to re-flow text


class OverlayTextObject(OverlayScalableImageObject):
    """Texture mapped image object that is fixed in size relative to the screen

    Image uses 4 control points like the rectangle, but uses a texture
    object as the foreground.  The background color will show through in
    trasparent pixels.

    """
    name = "Text"

    type = "overlay_text_obj"

    layer_info_panel = ["Text color", "Font", "Font size", "Border width", "Line style", "Line width", "Line color", "Fill style", "Fill color"]

    selection_info_panel = ["Text", "Text format", "Anchor point"]

    def init_overlay_attributes(self):
        self.user_text = "<b>New Label</b>"
        self.border_width = 4

    def user_text_to_json(self):
        return self.user_text

    def user_text_from_json(self, json_data):
        self.user_text = json_data['user_text']

    def get_text_box(self):
        return self

    def get_image_array(self):
        from maproom.library.numpy_images import OffScreenHTML
        bg = int_to_color_uint8(self.style.fill_color)
        h = OffScreenHTML(self.text_width, self.text_height, bg)
        c = int_to_html_color_string(self.style.text_color)
        arr = h.get_numpy(self.user_text, c, self.style.font, self.style.font_size, self.style.text_format)
        return arr

    def render_overlay(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        renderer.prepare_to_render_projected_objects()
        renderer.fill_object(self, picker, self.style)
        renderer.outline_object(self, picker, self.style)
        if not picker.is_active:
            # Only render text when we're not drawing the picker framebuffer
            renderer.prepare_to_render_screen_objects()
            alpha = alpha_from_int(self.style.text_color)
            renderer.draw_image(self, picker, alpha)


class OverlayIconObject(OverlayScalableImageObject):
    """Texture mapped Marplot icon object that is fixed in size relative to the screen

    Uses the Marplot category icons.
    """
    name = "Icon"

    type = "overlay_icon_obj"

    layer_info_panel = ["Marplot icon", "Icon size", "Color"]

    def init_overlay_attributes(self):
        self.anchor_point_index = 8  # Defaults to center point as the anchor
        self.text_width = 32
        self.text_height = 32
        self.border_width = 5
        self.min_size = 10

    def set_style(self, style):
        OverlayScalableImageObject.set_style(self, style)
        self.text_width = self.style.icon_pixel_size
        self.text_height = self.style.icon_pixel_size
        self.rebuild_needed = True  # Force rebuild to change icon size

    def fit_to_bounding_box(self, current_bounds, new_bounds):
        # Don't do anything... The icon shouldn't scale with the parent scaling
        pass

    def get_image_array(self):
        return self.style.get_numpy_image_from_icon()

    def set_overlay_position(self, renderer):
        c = renderer.canvas
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        center = c.get_numpy_screen_point_from_world_point(p[self.center_point_index]['xy'])
        if self.text_width < self.text_height:
            w1 = self.text_width
            w2 = self.image_data.x
        else:
            w1 = self.text_height
            w2 = self.image_data.y
        w1 = max(w1, self.min_size)
        scale = w1 * 1.0 / w2
        renderer.set_image_center_at_screen_point(self.image_data, center, c.screen_rect, scale)


class PolylineMixin(object):
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

    def set_points(self, points):
        points = np.asarray(points)

        # initialize boundary box control points with zeros; will be filled in
        # with call to recalc_bounding_box below
        cp = np.zeros((self.center_point_index + 1, 2), dtype=np.float32)

        p = np.concatenate((cp, points), 0)  # flatten to 1D
        lines = self.get_polylines(np.alen(points))
        self.set_data(p, 0.0, np.asarray(lines, dtype=np.uint32))
        self.recalc_bounding_box()

    def get_polylines(self, num_points):
        offset = self.center_point_index + 1
        lines = list(zip(list(range(offset, offset + num_points - 1)), list(range(offset + 1, offset + num_points))))
        return lines

    def move_polyline_point(self, anchor, dx, dy):
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy
        points[anchor] += (dx, dy)
        self.recalc_bounding_box()

    def recalc_bounding_box(self):
        offset = self.center_point_index + 1
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy
        r = rect.get_rect_of_points(points[offset:])
        corners = np.empty((4, 2), dtype=np.float32)
        corners[0] = r[0]
        corners[1] = (r[1][0], r[0][1])
        corners[2] = r[1]
        corners[3] = (r[0][0], r[1][1])
        cp = self.get_control_points_from_corners(corners)
        points[0:offset] = cp
        self.update_bounds()

    def fit_to_bounding_box(self, current_bounds, new_bounds):
        # adjust polyline points; control points have already been moved
        offset = self.center_point_index + 1
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
        scale, old_origin, new_origin = rect.get_transform(current_bounds, new_bounds)
        points = (p.xy[offset:] - old_origin).dot(scale) + new_origin
        p.xy[offset:] = points

    def rasterize(self, renderer, projected_point_data, z, cp_color, line_color):
        self.rasterize_points(renderer, projected_point_data, z, cp_color)
        colors = np.empty(np.alen(self.line_segment_indexes), dtype=np.uint32)
        colors.fill(line_color)
        renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], colors)


class PolylineObject(PolylineMixin, RectangleMixin, LineVectorObject):
    name = "Polyline"

    type = "polyline_obj"

    display_center_control_point = True

    def calculate_distances(self):
        return haversine_list(self.points[self.center_point_index + 1:])

    def get_marker_points(self):
        # Markers are only used on the first and last segments of the line
        indexes = self.line_segment_indexes
        if len(indexes) > 0:
            return (
                (indexes.point1[0], indexes.point2[0], self.style.line_start_marker),
                (indexes.point2[-1], indexes.point1[-1], self.style.line_end_marker))

    def get_all_boundaries(self):
        indexes = list(range(self.center_point_index + 1, np.alen(self.points)))
        b = Boundary(self.points, indexes, 0.0)
        return [b]


class PolygonObject(PolylineMixin, RectangleMixin, FillableVectorObject):
    name = "Polygon"

    type = "polygon_obj"

    layer_info_panel = ["Line style", "Line width", "Line color", "Fill style", "Fill color"]

    selection_info_panel = ["Anchor latitude", "Anchor longitude", "Area"]

    def get_area(self):
        """Adapted from http://stackoverflow.com/questions/4681737

        Assumes spherical earth so there will be an inaccuracy, but it's good
        enough for this purpose for now.
        """
        r = 6371.
        lat_dist = math.pi * r / 180.0
        area = 0.0
        indexes = list(range(self.center_point_index + 1, np.alen(self.points)))
        indexes.append(self.center_point_index + 1)
        x = []
        y = []
        for i in indexes:
            lon = self.points[i].x
            lat = self.points[i].y
            y.append(lat * lat_dist)
            x.append(lon * lat_dist * math.cos(math.radians(lat)))
        for i in range(-1, len(indexes) - 1):
            area += x[i] * (y[i + 1] - y[i - 1])
        return abs(area) / 2.0

    def get_info_panel_text(self, prop):
        if prop == "Area":
            km = self.get_area()
            return "%s, %s" % (km2_to_rounded_string(km), mi2_to_rounded_string(km * .621371 * .621371))
        return FillableVectorObject.get_info_panel_text(self, prop)

    def get_polylines(self, num_points):
        offset = self.center_point_index + 1
        lines = list(zip(list(range(offset, offset + num_points - 1)), list(range(offset + 1, offset + num_points))))
        lines.append((offset + num_points - 1, offset))
        return lines

    def get_marker_points(self):
        # Polygon is closed, so endpoint markers don't make sense
        return []

    def has_boundaries(self):
        return True

    def get_all_boundaries(self):
        indexes = list(range(self.center_point_index + 1, np.alen(self.points)))
        indexes.append(self.center_point_index + 1)
        b = Boundary(self.points, indexes, 0.0)
        return [b]

    def get_points_lines(self):
        start = self.center_point_index + 1
        count = np.alen(self.points) - start
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[start:start + count]
        lines = np.empty((count, 2), dtype=np.uint32)
        lines[:,0] = np.arange(0, count, dtype=np.uint32)
        lines[:,1] = np.arange(1, count + 1, dtype=np.uint32)
        lines[count - 1, 1] = 0
        return points, lines

    def rasterize(self, renderer, projected_point_data, z, cp_color, line_color):
        self.rasterize_points(renderer, projected_point_data, z, cp_color)
        colors = np.empty(np.alen(self.line_segment_indexes), dtype=np.uint32)
        colors.fill(line_color)
        start = self.center_point_index + 1
        last = np.alen(self.points)
        count = last - start
        rings = data_types.make_polygons(1)
        rings.start[0] = self.center_point_index + 1
        rings.count[0] = count
        rings.group[0] = 0
        rings.color[0] = self.style.fill_color
        adjacency = data_types.make_point_adjacency_array(np.alen(self.points))
        adjacency.ring_index[0:start] = 99999
        adjacency.ring_index[start:last] = 0
        adjacency.next[start:last] = np.arange(start + 1, last + 1)
        adjacency.next[last - 1] = start
        renderer.set_polygons(rings, adjacency)
        self.rasterized_rings = rings

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering vector object %s!!! pick=%s" % (self.name, picker))
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        renderer.draw_polygons(self, picker,
                               self.rasterized_rings.color,
                               self.style.line_color,
                               1, self.style)


class AnnotationLayer(BoundedFolder, RectangleVectorObject):
    """Layer for vector annotation image

    """
    name = "Annotation"

    type = "annotation"

    mouse_mode_toolbar = "AnnotationLayerToolBar"

    layer_info_panel = ["Text color", "Font", "Font size", "Border width", "Line style", "Line width", "Line color", "Fill style", "Fill color"]

    selection_info_panel = ["Anchor latitude", "Anchor longitude", "Width", "Height", "Area"]

    @property
    def contains_overlays(self):
        return True

    def has_groupable_objects(self):
        return True

    def can_reparent_to(self, potential_parent_layer):
        # The annotation folder itself can be a child of anything
        return True

    @property
    def can_contain_annotations(self):
        return True

    def can_contain_vector_object(self, potential_vector_object_layer):
        return True

    def can_output_feature_list(self):
        return True

    def calc_output_feature_list(self):
        """Create list of geometry primitives"""
        output = []
        children = self.manager.get_layer_children(self)
        for layer in children:
            output.extend(layer.calc_output_feature_list())
        return output

    def set_border_width(self, width):
        self.border_width = width
        children = self.manager.get_layer_children(self)
        for layer in children:
            layer.set_border_width(width)

    def get_renderer_colors(self):
        """Hook to allow subclasses to override style colors
        """
        style = styles.default_styles_read_only("ui")
        line_color = style.line_color
        r, g, b, a = int_to_color_floats(line_color)
        point_color = color_floats_to_int(r, g, b, 1.0)
        return point_color, line_color

    def set_data_from_bounds(self, bounds):
        log.debug("SETTING BOUNDARY BOX!!! %s %s" % (self, bounds))
        if bounds[0][0] is None:
            self.points = data_types.make_points(0)

        else:
            points = np.asarray(bounds, dtype=np.float32)
            self.set_opposite_corners(points[0], points[1], update_bounds=False)

    def children_affected_by_move(self):
        affected = []
        for layer in self.manager.get_layer_children(self):
            affected.extend(layer.children_affected_by_move())
        affected.append(self)
        return affected

    def fit_to_bounding_box(self, current_bounds, new_bounds):
        layers = self.manager.get_layer_children(self)
        # print(f"FITTING SUB-OBJECTS of {self} = {layers} TO NEW BOUNDING BOX!!!")
        # print(f"  old bounds: {str(current_bounds)}")
        # print(f"  new bounds: {str(new_bounds)}")
        scale, old_origin, new_origin = rect.get_transform(current_bounds, new_bounds)
        # print(" scale: %s" % str(scale))
        # print(" old origin: %s" % str(old_origin))
        # print(" new origin: %s" % str(new_origin))
        for layer in layers:
            # print(f"    fitting layer: {layer}, bounds={layer.bounds}")
            # fit the child layer in the new bounding box by a linear transform
            # of the child bounding box in the proportion of the parent
            # bounding box change
            offset = layer.center_point_index + 1
            current_sublayer_bounds = layer.copy_bounds()
            p = layer.points.view(data_types.POINT_XY_VIEW_DTYPE)
            # print(f"    before: {str(p.xy)}")
            relative = (p.xy[:offset] - old_origin).dot(scale)
            # print(f"    relative: {str(relative)}")
            points = relative + new_origin
            p.xy[:offset] = points
            # print(f"    after: {str(p.xy)}")

            # set new bounding rect
            layer.bounds = self.compute_bounding_rect_from_points(layer.points[0:offset])

            # recursively fit children
            layer.fit_to_bounding_box(current_sublayer_bounds, layer.bounds)
            # offset = self.center_point_index + 1
            # p = self.points.view(data_types.POINT_XY_VIEW_DTYPE)
            # points = ((p.xy[offset:] - old_origin) * scale) + new_origin
            # p.xy[offset:] = points

    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer

        """
        self.update_overlay_bounds()
        if self.bounds[0][0] is not None:
            RectangleVectorObject.rebuild_renderer(self, renderer, in_place)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering annotation layer group %s!!! pick=%s" % (self.name, picker))
        # set new bounding rect every time
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        if renderer.canvas.project.current_layer == self:
            style = styles.default_styles_read_only("ui")
            renderer.outline_object(self, picker, style)

    def render_control_points_only(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering vector object control points %s!!!" % (self.name))
        if renderer.canvas.project.current_layer == self:
            renderer.draw_points(self, picker, self.point_size)


class ArrowTextBoxLayer(AnnotationLayer):
    """Folder layer for to hold a group of two annotation objects: a text box and an
    arrow pointing to lat/lon.

    The layer and associations among child layers is created in DrawArrowTextBoxCommand
    """
    name = "Arrow Text Box"

    type = "arrowtextbox"

    layer_info_panel = ["Text color", "Font", "Font size", "Border width", "Line style", "Line width", "Line color", "Fill style", "Fill color"]

    selection_info_panel = ["Text", "Text format", "Anchor point", "Anchor latitude", "Anchor longitude", "Width", "Height", "Area"]

    def can_reparent_to(self, potential_parent_layer):
        return potential_parent_layer.can_contain_annotations

    def can_contain_vector_object(self, potential_vector_object_layer):
        return False

    def get_layer_of_anchor(self):
        return self.get_text_box()

    def get_text_box(self):
        children = self.manager.get_layer_children(self)
        for layer in children:
            if hasattr(layer, "user_text"):
                return layer
        return None

    @property
    def anchor_point_index(self):
        textbox = self.get_text_box()
        if textbox is not None:
            return textbox.anchor_point_index
        return 0

    def set_anchor_index(self, anchor):
        textbox = self.get_text_box()
        if textbox is not None:
            textbox.set_anchor_index(anchor)

    @property
    def user_text(self):
        textbox = self.get_text_box()
        if textbox is not None:
            return textbox.user_text
        return ""

    @user_text.setter
    def user_text(self, value):
        textbox = self.get_text_box()
        if textbox is not None:
            textbox.user_text = value



class ArrowTextIconLayer(ArrowTextBoxLayer):
    """Folder layer for to hold a group of three annotation objects: a text box, an
    arrow pointing to lat/lon, and an icon at that lat/lon location.
 
    The layer and associations among child layers is created in DrawArrowTextIconCommand
    """
    name = "Arrow Text Icon"

    type = "arrowtexticon"

    layer_info_panel = ["Text color", "Font", "Font size", "Border width", "Line style", "Line width", "Line color", "Fill style", "Fill color", "Marplot icon", "Icon size"]

    def use_layer_for_bounding_rect(self, layer):
        # Defaults to using all layers in boundary rect calculation
        return not isinstance(layer, OverlayIconObject)
