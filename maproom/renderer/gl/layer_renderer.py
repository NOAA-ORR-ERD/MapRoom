import os
import time
import sys
import numpy as np
import Point_and_line_set_renderer
import Triangle_set_renderer
import Polygon_set_renderer
import Label_set_renderer
import Image_set_renderer
import data_types


"""
    point: x, y, z (depth), color, state
    state = selected | flagged | deleted | edited | added | land_polygon | water_polygon | other_polygon
"""

# for the picker, we only have a total of 255 "layers" that it tracks;
# so we give each Layer 10 slots, 5 for the point-and-line renderer and
# 5 for the polygon renderer; within each renderer, these 5 are further
# divided to distinguish between points, lines, and fills; so the assumption
# here is that we won't have more than 25 actively "pickable" Layers, and
# each layer won't have more than 10 subcategories of pickable item types
# among its renderers

class LayerRenderer(object):
    """Data class to store layer rendering needed on a per-view basis.
    
    Because each LayerManager can be in multiple views, we can't store layer
    renderer data in either the layer or the layer manager.  As currently
    designed, the layer renderers cache data for each view (e.g.  OpenGL
    VBOs), and it might lead to some unintended drawing errors.  Not sure at
    this point, but to ease the refactoring to self-rendering layers, this
    class provides that each view has its own renderer for each layer.

    Maybe as I understand the code better I might realize that all the data
    could be shared in which case the renderers could simply become attributes
    of the layer itself and this storage class would go away.
    """
    POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET = 0
    POLYGONS_SUB_LAYER_PICKER_OFFSET = 5

    MAX_LABEL_CHARACTERS = 1000 * 5

    def __init__(self, canvas):
        self.canvas = canvas

        self.point_and_line_set_renderer = None
        self.triangle_set_renderer = None
        self.label_set_renderer = None
        self.polygon_set_renderer = None
        self.image_set_renderer = None

    def __repr__(self):
        return self.name

    def rebuild_image_set_renderer(self, layer):
        if layer.image_data:
            if not layer.image_data.image_textures:
                layer.image_data.image_textures = Image_set_renderer.ImageTextures(
                    self.canvas.opengl_renderer,
                    layer.image_data.images,
                    layer.image_data.image_sizes,
                    layer.image_data.image_world_rects)
                layer.image_data.release_images()
            if not self.image_set_renderer:
                self.image_set_renderer = Image_set_renderer.Image_set_renderer(
                    self.canvas.opengl_renderer,
                    layer.image_data.image_textures,
                    self.canvas.projection,
                    self.canvas.projection_is_identity)

    def set_up_labels(self, layer):
        if (layer.points != None and self.label_set_renderer == None):
            self.label_set_renderer = Label_set_renderer.Label_set_renderer(self.canvas.opengl_renderer, self.MAX_LABEL_CHARACTERS)

    def rebuild_triangle_set_renderer(self, layer):
        if self.triangle_set_renderer:
            self.triangle_set_renderer.destroy()

        self.triangle_set_renderer = Triangle_set_renderer.Triangle_set_renderer(
            self.canvas.opengl_renderer,
            layer.triangle_points.view(data_types.POINT_XY_VIEW_DTYPE).xy,
            layer.triangle_points.color.copy().view(dtype=np.uint8),
            layer.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes,
            self.canvas.projection,
            self.canvas.projection_is_identity)

    def rebuild_point_and_line_set_renderer(self, layer, create=False):
        if self.point_and_line_set_renderer:
            create = True
            self.point_and_line_set_renderer.destroy()

        t0 = time.clock()
        if create:
            self.point_and_line_set_renderer = Point_and_line_set_renderer.Point_and_line_set_renderer(
                self.canvas.opengl_renderer,
                layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy,
                layer.points.color.copy().view(dtype=np.uint8),
                layer.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"],
                layer.line_segment_indexes.color,
                self.canvas.projection,
                self.canvas.projection_is_identity)

        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        print "rebuilt point and line set renderer in {0} seconds".format(t)

    def rebuild_polygon_set_renderer(self, layer):
        if self.polygon_set_renderer:
            self.polygon_set_renderer.destroy()

        self.polygon_set_renderer = Polygon_set_renderer.Polygon_set_renderer(
            self.canvas.opengl_renderer,
            layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[: len(layer.points)].copy(),
            layer.polygon_adjacency_array,
            layer.polygons,
            self.canvas.projection,
            self.canvas.projection_is_identity)

    def reproject(self, projection, projection_is_identity):
        if (self.point_and_line_set_renderer != None):
            self.point_and_line_set_renderer.reproject(self.layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy,
                                                       projection,
                                                       projection_is_identity)
        if (self.polygon_set_renderer != None):
            self.polygon_set_renderer.reproject(projection, projection_is_identity)
        """
        if ( self.label_set_renderer != None ):
            self.label_set_renderer.reproject( self.layer.points.view( self.POINT_XY_VIEW_DTYPE ).xy,
                                               projection,
                                               projection_is_identity )
        """
        if (self.image_set_renderer != None):
            self.image_set_renderer.reproject(projection, projection_is_identity)

    def __del__(self):
        if (self.point_and_line_set_renderer != None):
            self.point_and_line_set_renderer.destroy()
            self.point_and_line_set_renderer = None
        if (self.polygon_set_renderer != None):
            self.polygon_set_renderer.destroy()
            self.polygon_set_renderer = None
        if (self.label_set_renderer != None):
            self.label_set_renderer.destroy()
            self.label_set_renderer = None
        if (self.image_set_renderer != None):
            self.image_set_renderer.destroy()
            self.image_set_renderer = None
