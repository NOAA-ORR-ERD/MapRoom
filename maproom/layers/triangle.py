import numpy as np

from pytriangle import triangulate_simple

# Enthought library imports.
from traits.api import Any
from traits.api import Str

from ..library.Boundary import Boundaries
from ..renderer import color_floats_to_int, data_types

from .point import PointLayer
from . import state

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class TriangleLayer(PointLayer):
    """Layer for triangles.

    """
    type = "triangle"

    mouse_mode_toolbar = Str("BaseLayerToolBar")

    triangles = Any

    visibility_items = ["points", "triangles", "labels"]

    use_color_cycling = True

    layer_info_panel = ["Triangle count", "Show depth shading"]

    def __str__(self):
        try:
            triangles = len(self.triangles)
        except TypeError:
            triangles = 0
        return PointLayer.__str__(self) + ", %d triangles" % triangles

    def get_info_panel_text(self, prop):
        if prop == "Triangle count":
            if self.triangles is not None:
                return str(len(self.triangles))
            return "0"
        return PointLayer.get_info_panel_text(self, prop)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)
        no_triangles = (self.triangles is None or len(self.triangles) == 0)

        return no_points and no_triangles

    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in ["points", "labels"]:
            return self.points is not None
        if label == "triangles":
            return self.triangles is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))

    def set_data(self, f_points, f_depths, f_triangles):
        n = np.alen(f_points)
        self.points = data_types.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[
                0: n
            ] = f_points
            self.points.z[
                0: n
            ] = f_depths
            self.points.color = self.style.line_color
            self.points.state = 0

            n = len(f_triangles)
            if n > 0:
                self.triangles = self.make_triangles(n)
                self.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes = f_triangles

        self.update_bounds()

    def can_save(self):
        return True

    def triangles_to_json(self):
        return self.triangles.tolist()

    def triangles_from_json(self, json_data):
        self.triangles = np.array([tuple(i) for i in json_data['triangles']], data_types.TRIANGLE_DTYPE).view(np.recarray)

    def update_after_insert_point_at_index(self, point_index):
        # update point indexes in the triangles to account for the inserted point
        if (self.triangles is not None):
            offsets = np.zeros(np.alen(self.triangles))
            offsets += np.where(self.triangles.point1 >= point_index, 1, 0).astype(np.uint32)
            self.triangles.point1 += offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point2 >= point_index, 1, 0).astype(np.uint32)
            self.triangles.point2 += offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point3 >= point_index, 1, 0).astype(np.uint32)
            self.triangles.point3 += offsets

    def insert_triangle(self, point_index_1, point_index_2, point_index_3):
        return self.insert_triangle_at_index(len(self.triangles), (point_index_1, point_index_2, point_index_3, self.style.line_color, state.CLEAR))

    def insert_triangle_at_index(self, index, params):
        entry = np.array([params],
                         dtype=data_types.TRIANGLE_DTYPE).view(np.recarray)
        self.triangles = np.insert(self.triangles, index, entry).view(np.recarray)

        return index

    def delete_triangle(self, index):
        # t = self.triangles[index]
        # params = (t.point1, t.point2, t.color, t.state)
        # FIXME: add undo info
        self.triangles = np.delete(self.triangles, index, 0)

    def delete_all_selected_objects(self):
        point_indexes = self.get_selected_point_indexes()
        if point_indexes is not None and len(point_indexes) > 0:
            self.delete_points_and_triangles(point_indexes, True)
        self.increment_change_count()

    def update_after_delete_point(self, point_index):
        # update point indexes in the triangles to account for the inserted point
        if (self.triangles is not None):
            offsets = np.zeros(np.alen(self.triangles))
            offsets += np.where(self.triangles.point1 >= point_index, 1, 0).astype(np.uint32)
            self.triangles.point1 -= offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point2 >= point_index, 1, 0).astype(np.uint32)
            self.triangles.point2 -= offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point3 >= point_index, 1, 0).astype(np.uint32)
            self.triangles.point3 -= offsets

    def delete_points_and_triangles(self, point_indexes):
        triangle_indexes_to_be_deleted = None
        if (self.triangles is not None):
            # (1) delete any triangles whose points are going away
            triangle_indexes_to_be_deleted = np.where(np.in1d(self.triangles.point1, point_indexes))
            triangle_indexes_to_be_deleted = np.append(triangle_indexes_to_be_deleted, np.where(np.in1d(self.triangles.point2, point_indexes)))
            triangle_indexes_to_be_deleted = np.append(triangle_indexes_to_be_deleted, np.where(np.in1d(self.triangles.point3, point_indexes)))
            triangle_indexes_to_be_deleted = np.unique(triangle_indexes_to_be_deleted)

            # FIXME: change to Command class for undo
#                # add everything to the undo stack in an order such that if it was undone from last to first it would all work
#                l = list(triangle_indexes_to_be_deleted)
#                l.reverse()
#                for i in l:
#                    params = (self.triangles.point1[i], self.triangles.point2[i], self.triangles.point3[i], self.triangles.color[i], self.triangles.state[i])
#                    self.manager.add_undo_operation_to_operation_batch(OP_DELETE_TRIANGLE, self, i, params)

            # adjust the point indexes of the remaining triangles
            offsets = np.zeros(np.alen(self.triangles))
            for index in point_indexes:
                offsets += np.where(self.triangles.point1 > index, 1, 0).astype(np.uint32)
            self.triangles.point1 -= offsets
            offsets[: np.alen(offsets)] = 0
            for index in point_indexes:
                offsets += np.where(self.triangles.point2 > index, 1, 0).astype(np.uint32)
            self.triangles.point2 -= offsets
            offsets[: np.alen(offsets)] = 0
            for index in point_indexes:
                offsets += np.where(self.triangles.point3 > index, 1, 0).astype(np.uint32)
            self.triangles.point3 -= offsets

        # FIXME: change to Command class for undo
#            # add everything to the undo stack in an order such that if it was undone from last to first it would all work
#            l = list(point_indexes)
#            l.reverse()
#            for i in l:
#                params = ((self.points.x[i], self.points.y[i]), self.points.z[i], self.points.color[i], self.points.state[i])
#                self.manager.add_undo_operation_to_operation_batch(OP_DELETE_POINT, self, i, params)

        # delete them from the layer
        self.points = np.delete(self.points, point_indexes, 0)
        if (triangle_indexes_to_be_deleted is not None):
            # then delete the line segments
            self.triangles = np.delete(self.triangles, triangle_indexes_to_be_deleted, 0)

        # delete them from the point_and_line_set_renderer (by simply rebuilding it)

        """
        # delete them from the label_set_renderer
        if ( self.label_set_renderer is not None ):
            self.label_set_renderer.delete_points( point_indexes )
            self.label_set_renderer.reproject( self.points.view( data_types.POINT_XY_VIEW_DTYPE ).xy,
                                               self.manager.project.layer_canvas.projection,
                                               self.manager.project.layer_canvas.projection_is_identity )
        """

        # when points are deleted from a layer the indexes of the points in the existing merge dialog box
        # become invalid; so force the user to re-find duplicates in order to create a valid list again
        self.manager.layer_contents_deleted_event(self)

    def make_triangles(self, count):
        return np.repeat(
            np.array([(0, 0, 0, 0, 0)], dtype=data_types.TRIANGLE_DTYPE),
            count,
        ).view(np.recarray)

    def get_triangulated_points(self, layer, q, a):
        # determine the boundaries in the parent layer
        boundaries = Boundaries(layer, allow_branches=True, allow_self_crossing=False)
        # boundaries.check_errors(True)

        progress_log.info("Triangulating...")

        # calculate a hole point for each boundary
        hole_points_xy = np.empty(
            (len(boundaries), 2), np.float64,
        )

        for (boundary_index, boundary) in enumerate(boundaries):
            if (len(boundary.point_indexes) < 3):
                continue

            # the "hole" point for the outer boundary (first in the list) should be outside of it
            if boundary_index == 0:
                hole_points_xy[boundary_index] = boundary.generate_outside_hole_point()
            else:
                hole_points_xy[boundary_index] = boundary.generate_inside_hole_point()

        params = "V"
        if (q is not None):
            params = params + "q" + str(q)
        if (a is not None):
            params = params + "a" + str(a)
        if (len(boundaries) == 0):
            params = params + "cO"  # allows triangulation without explicit boundary

        # we need to use projected points for the triangulation
        projected_points = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[: len(layer.points)].view(np.float64).copy()
        projected_points[:,0], projected_points[:,1] = self.manager.project.layer_canvas.projection(layer.points.x, layer.points.y)
        hole_points_xy[:,0], hole_points_xy[:,1] = self.manager.project.layer_canvas.projection(hole_points_xy[:,0], hole_points_xy[:,1])
#        print "params: " + params
#        print "hole points:"
#        print hole_points_xy
        (triangle_points_xy,
         triangle_points_z,
         triangle_line_segment_indexes,  # not needed
         triangles) = triangulate_simple(
            params,
            projected_points,
            layer.points.z[: len(layer.points)].copy(),
            layer.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points[: len(layer.line_segment_indexes)].view(np.uint32).copy(),
            hole_points_xy)
        return (triangle_points_xy,
                triangle_points_z,
                triangles)

    def unproject_triangle_points(self, points):
        points.x, points.y = self.manager.project.layer_canvas.projection(points.x, points.y, inverse=True)

    def triangulate_from_data(self, points, depths, triangles):
        self.set_data(points, depths, triangles)
        self.unproject_triangle_points(self.points)
        self.manager.layer_contents_changed_event(self)
        self.manager.layer_metadata_changed_event(self)

    def triangulate_from_layer(self, parent_layer, q, a):
        points, depths, triangles = self.get_triangulated_points(parent_layer, q, a)
        self.triangulate_from_data(points, depths, triangles)

    def color_interp(self, z, colormap, alpha):
        c0 = colormap[0]
        if z < c0[0]:
            return color_floats_to_int(c0[1] / 255., c0[2] / 255., c0[3] / 255., alpha)
        for c in colormap[1:]:
            if z >= c0[0] and z <= c[0]:
                perc = (z - c0[0]) / float(c[0] - c0[0])
                return color_floats_to_int((c0[1] + (c[1] - c0[1]) * perc) / 255.,
                                           (c0[2] + (c[2] - c0[2]) * perc) / 255.,
                                           (c0[3] + (c[3] - c0[3]) * perc) / 255.,
                                           alpha)
            c0 = c
        return color_floats_to_int(c[1] / 255., c[2] / 255., c[3] / 255., alpha)

    def get_triangle_point_colors(self, alpha=.9):
        colors = np.zeros(len(self.points), dtype=np.uint32)
        if self.points is not None:

            # Lots of points in the colormap doesn't help because the shading
            # is only applied linearly based on the depth of the endpoints.
            # So if there are colors at depths 10, 25, 50, and 100, but the
            # triangle has points of depth 0 and 100, the naive GL shader
            # skips the colors at 25 and 50.  If it were written as a GLSL
            # shader, maybe a complicated palette could be implemented.  But I
            # know nothing of GLSL at this moment.
            colormap = (
                (-10, 0xf0, 0xeb, 0xc3),
                (-0.01, 0xf0, 0xeb, 0xc3),
                (0, 0xd6, 0xea, 0xeb),
                #                (10, 0x9b, 0xd3, 0xe0),
                #                (20, 0x54, 0xc0, 0xdc),
                #                (30, 0x00, 0xa0, 0xcc),
                #                (40, 0x00, 0x6a, 0xa4),
                #                (50, 0x1f, 0x48, 0x8a),
                (100, 0x00, 0x04, 0x69),
            )

            for i in range(len(colors)):
                d = self.points.z[i]
                colors[i] = self.color_interp(d, colormap, alpha)
        return colors

    def rebuild_renderer(self, renderer, in_place=False):
        """Update display canvas data with the data in this layer

        """
        projected_point_data = self.compute_projected_point_data(renderer.canvas.projection)
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        triangles = self.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes
        tri_points_color = self.get_triangle_point_colors()
        renderer.set_triangles(triangles, tri_points_color)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering line layer!!! pick=%s" % (picker))
        if picker.is_active:
            return

        renderer.draw_triangles(self.style.line_width, layer_visibility["triangles"])

        if layer_visibility["labels"]:
            renderer.draw_labels_at_points(self.points.z, s_r, p_r)
