import os
import sys
import wx
import numpy as np

from peppy2 import get_image_path

import maproom.library.rect as rect
from .. import data_types, int_to_color_floats, color_floats_to_int
from .. import BaseRenderer


class ReportLabRenderer(BaseRenderer):
    def __init__(self, canvas, layer):
        self.canvas = canvas
        self.layer = layer
        self.point_xys = None
        self.point_colors = None
        self.line_xys = None
        self.line_colors = None
        self.image_textures = None
        self.image_projected_rects = []
        
        self.debug_text_bounding_box = False

    def prepare_to_render_projected_objects(self):
        self.canvas.set_projected_viewport()

    def prepare_to_render_screen_objects(self):
        self.canvas.set_screen_viewport()
    
    def convert_colors(self, color, count):
        if color is None:
            black = ((0.0, 0.0, 0.0), 1.0)
            converted = [black for i in range(count)]
        else:
            c = color.view(dtype=np.uint32)
            converted = [((r, g, b), a) for r, g, b, a in [int_to_color_floats(color) for color in c]]
            print "colors", converted
        return converted
    
    def set_points(self, xy, depths, color=None, num_points=-1):
        if num_points == -1:
            num_points = np.alen(xy)
        self.point_xys = np.empty((num_points, 2), dtype=np.float32)
        self.point_xys[:num_points] = xy[:num_points]
        self.point_colors = self.convert_colors(color, num_points)
    
    def set_lines(self, xy, indexes, color):
        self.line_xys = xy[indexes.reshape(-1)].astype(np.float32).reshape(-1, 2)  # .view( self.SIMPLE_POINT_DTYPE ).copy()
        self.line_colors = self.convert_colors(color, np.alen(self.line_xys) / 2)
    
    def draw_lines(self,
                   layer_index_base,
                   picker,
                   style,
                   selected_line_segment_indexes=[],
                   flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        c = self.canvas
        points = self.line_xys.reshape(-1, 4)
        for (x1, y1, x2, y2), (rgb, alpha) in zip(points, self.line_colors):
            print "%f,%f -> %f,%f" % (x1, y1, x2, y2)
            c.pdf.setStrokeColor(rgb, alpha)
            c.pdf.line(x1, y1, x2, y2)

    def draw_selected_lines(self, style, selected_line_segment_indexes=[]):
        pass

    def draw_points(self,
                    layer_index_base,
                    picker,
                    point_size,
                    selected_point_indexes=[],
                    flagged_point_indexes=[]):  # flagged_line_segment_indexes not yet used
        c = self.canvas
        r = point_size * self.canvas.projected_units_per_pixel / 2
        for (x, y), (rgb, alpha) in zip(self.point_xys, self.point_colors):
            print "point %f,%f, r=%f" % (x, y, r)
            c.pdf.setFillColor(rgb, alpha)
            c.pdf.circle(x, y, r, fill=1, stroke=0)

    def draw_selected_points(self, point_size, selected_point_indexes=[]):
        pass

    def draw_labels_at_points(self, values, screen_rect, projected_rect):
        c = self.canvas
        n, labels, relevant_points = c.get_visible_labels(values, self.point_xys, projected_rect)
        if n == 0:
            return
        
        c.pdf.setFillColor((0.0, 0.0, 0.0), 1.0)
        for index, s in enumerate(labels):
            x, y = relevant_points[index]
            w, h = c.get_font_metrics(s)
            x -= w / 2
            y -= h
            c.pdf.drawString(x, y, s)
            print "label %f,%f: %s" % (x, y, s)

    def set_triangles(self, triangle_point_indexes, triangle_point_colors):
        pass

    def draw_triangles(self, line_width):
        pass

    def set_image_projection(self, image_data, projection):
        pass

    def set_image_screen(self, image_data):
        pass
    
    def set_image_center_at_screen_point(self, image_data, center, screen_rect):
        pass
    
    def release_textures(self):
        pass

    def draw_image(self, layer_index_base, picker, alpha=1.0):
        pass

    def set_invalid_polygons(self, polygons, polygon_count):
        pass

    def set_polygons(self, polygons, point_adjacency_array):
        pass

    def draw_polygons(self, layer_index_base, picker,
                      polygon_colors, line_color, line_width,
                      broken_polygon_index=None):
        pass

    def draw_screen_line(self, point_a, point_b, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF, xor=False):
        print "line", point_a[0], point_a[1], point_b[0], point_b[1]
        c = self.canvas
        h = rect.height(c.screen_rect)
        c.pdf.setStrokeColor((red, green, blue), alpha)
        c.pdf.setLineWidth(width / c.viewport_scale)
        c.pdf.line(point_a[0], h - point_a[1], point_b[0], h - point_b[1])

    def draw_screen_lines(self, points, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF, xor=False):
        c = self.canvas
        h = rect.height(c.screen_rect)
        c.pdf.setStrokeColor((red, green, blue), alpha)
        c.pdf.setLineWidth(width / c.viewport_scale)
        for x1, y1, x2, y2 in points:
            print "%f,%f -> %f,%f" % (x1, y1, x2, y2)
            c.pdf.line(x1, h - y1, x2, h - y2)

    def draw_screen_markers(self, markers, style):
        pass

    def draw_screen_box(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0, width=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        pass

    def draw_screen_rect(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0):
        pass

    def get_drawn_string_dimensions(self, text):
        return self.canvas.get_font_metrics(text)

    def draw_screen_string(self, point, text):
        c = self.canvas
        h = rect.height(c.screen_rect) - c.font_scale
        print "text %f,%f -> '%s'" % (point[0], h - point[1], text.encode("utf-8"))
        c.pdf.setFillColor((0.0, 0.0, 0.0), 1.0)
        c.pdf.drawString(point[0], h - point[1], text)
        if self.debug_text_bounding_box:
            dims = self.get_drawn_string_dimensions(text)
            c.pdf.rect(point[0], h - point[1], dims[0], dims[1], fill=0, stroke=1)

    # Vector object drawing routines

    def convert_color(self, color):
        r, g, b, a = int_to_color_floats(color)
        print "rgb, a:", (r, g, b), a
        return (r, g, b), a
    
    def set_stroke_style(self, style):
        d = self.canvas.pdf
        rgb, a = self.convert_color(style.line_color)
        d.setStrokeColor(rgb, a)
        w = style.line_width / self.canvas.viewport_scale
        d.setLineWidth(w)
        return style.line_stipple > 0
    
    def set_fill_style(self, style):
        rgb, a = self.convert_color(style.fill_color)
        self.canvas.pdf.setFillColor(rgb, a)
        return style.fill_style > 0

    def fill_object(self, layer_index_base, picker, style):
        d = self.canvas.pdf
        if self.set_fill_style(style):
            p = d.beginPath()
            x, y = self.line_xys[0]
            p.moveTo(x, y)
            for x, y in self.line_xys[1:]:
                print "%f -> %f" % (x, y)
                p.lineTo(x, y)
            d.drawPath(p, fill=1, stroke=0)

    def outline_object(self, layer_index_base, picker, style):
        d = self.canvas.pdf
        if self.set_stroke_style(style):
            p = d.beginPath()
            x, y = self.line_xys[0]
            p.moveTo(x, y)
            for x, y in self.line_xys[1:]:
                print "%f -> %f" % (x, y)
                p.lineTo(x, y)
            d.drawPath(p, fill=0, stroke=1)
