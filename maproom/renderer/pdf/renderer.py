import numpy as np

from PIL import Image

from reportlab.lib.utils import ImageReader


import maproom.library.rect as rect
from .. import int_to_color_floats
from .. import BaseRenderer

import logging
log = logging.getLogger(__name__)


class PDFImage(object):
    static_renderer = True

    def __init__(self, image_data):
        self.images = []
        self.xywh = []
        self.load(image_data)

    def foreach(self):
        for image, xywh in zip(self.images, self.xywh):
            yield image, xywh[0], xywh[1], xywh[2], xywh[3]

    def load(self, image_data):
        n = 0
        for image in image_data:
            try:
                if image.z != image_data.zoom_level:
                    # Skip tiles on background zoom levels
                    log.debug("skipping image with z=%d when image data has z=%d" % (image.z, image_data.zoom_level))
                    continue
            except AttributeError:
                pass
            converted = Image.fromarray(image.data, mode='RGBA')
            log.debug("PIL image: %s" % converted)
            self.images.append(converted)

    def set_projection(self, image_data, projection):
        self.xywh = []
        for image in image_data:
            print image.world_rect
            lb, lt, rt, rb = image.world_rect
            x1, y1 = projection(lb[0], lb[1])
            x2, y2 = projection(rt[0], rt[1])
            self.xywh.append((x1, y1, x2 - x1, y2 - y1))

    def use_screen_rect(self, image_data, r, scale=1.0):
        self.xywh = []
        for image in image_data:
            x = (image.origin[0] * scale) + r[0][0]
            y = (image.origin[1] * scale) + r[0][1]
            w = image.size[0] * scale
            h = image.size[1] * scale
            self.xywh.append((x, y, w, h))

    def center_at_screen_point(self, image_data, point, screen_height, scale=1.0):
        left = int(point[0] - (image_data.x / 2) * scale)
        bottom = int(point[1] + (image_data.y / 2) * scale)
        right = left + (image_data.x * scale)
        top = bottom + (image_data.y * scale)
        # flip y to treat rect as normal opengl coordinates
        r = ((left, screen_height - bottom),
             (right, screen_height - top))
        self.use_screen_rect(image_data, r, scale)

    def reorder_tiles(self, image_data):
        # not needed for PDF rendering; background zoom levels are thrown out
        # in load()
        pass


class ReportLabRenderer(BaseRenderer):
    def __init__(self, canvas, layer):
        self.canvas = canvas
        self.layer = layer
        self.point_xys = None
        self.point_colors = None
        self.line_xys = None
        self.line_colors = None
        self.images = None

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
                   layer,
                   picker,
                   style,
                   selected_line_segment_indexes=[],
                   flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        c = self.canvas
        points = self.line_xys.reshape(-1, 4)
        for (x1, y1, x2, y2), (rgb, alpha) in zip(points, self.line_colors):
            log.debug("draw_lines: %f,%f -> %f,%f" % (x1, y1, x2, y2))
            c.pdf.setStrokeColor(rgb, alpha)
            c.pdf.line(x1, y1, x2, y2)

    def draw_selected_lines(self, style, selected_line_segment_indexes=[]):
        pass

    def draw_points(self,
                    layer,
                    picker,
                    point_size,
                    selected_point_indexes=[],
                    flagged_point_indexes=[]):  # flagged_line_segment_indexes not yet used
        c = self.canvas
        r = point_size * self.canvas.projected_units_per_pixel / 2
        for (x, y), (rgb, alpha) in zip(self.point_xys, self.point_colors):
            log.debug("point %f,%f, r=%f" % (x, y, r))
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
            log.debug("label %f,%f: %s" % (x, y, s))

    def set_triangles(self, triangle_point_indexes, triangle_point_colors):
        pass

    def draw_triangles(self, line_width):
        pass

    def set_image_projection(self, image_data, projection):
        if self.images is None:
            self.images = PDFImage(image_data)
        self.images.set_projection(image_data, projection)

    def set_image_screen(self, image_data):
        if self.images is None:
            self.images = PDFImage(image_data)

    def set_image_center_at_screen_point(self, image_data, center, screen_rect, scale=1.0):
        height = rect.height(screen_rect)
        self.images.center_at_screen_point(image_data, center, height, scale)

    def release_textures(self):
        pass

    def draw_image(self, layer, picker, alpha=1.0):
        d = self.canvas.pdf
        for image, x, y, w, h in self.images.foreach():
            if self.canvas.is_onscreen(x, y, w, h):
                log.debug("draw_image: %s @ %f,%f" % (image, x, y))
                d.drawImage(ImageReader(image), x, y, w, h, mask="auto")
            else:
                log.debug("  skipping draw_image (fully clipped): %s @ %f,%f" % (image, x, y))

    def set_tiles(self, image_data):
        if self.images is None:
            self.images = PDFImage(image_data)
            self.images.set_projection(image_data, image_data.projection)
        self.image_tiles = self.images
        self.canvas.zoom_level = image_data.zoom_level

    def release_tiles(self):
        pass

    draw_tiles = draw_image

    def set_invalid_polygons(self, polygons, polygon_count):
        pass

    def set_polygons(self, polygons, point_adjacency_array):
        self.point_adjacency_array = point_adjacency_array.copy()
        self.polygons = polygons.copy()

    def draw_polygons(self, layer, picker,
                      polygon_colors, line_color, line_width,
                      broken_polygon_index=None):
        d = self.canvas.pdf
        rgb, a = self.convert_color(line_color)
        self.set_stroke(rgb, a, line_width)

        for polygon in self.polygons:
            rgb, a = self.convert_color(polygon['color'])
            d.setFillColor(rgb, a)
            p = d.beginPath()
            current = polygon['start']
            x, y = self.point_xys[current]
            p.moveTo(x, y)
            count = polygon['count']
            while count > 0:
                next, polygon_id = self.point_adjacency_array[current]
                x, y = self.point_xys[next]
                p.lineTo(x, y)
                count -= 1
                current = next
            d.drawPath(p, fill=1, stroke=1)

    def draw_screen_line(self, point_a, point_b, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF, xor=False):
        log.debug("screen_line %f,%f -> %f,%f" % (point_a[0], point_a[1], point_b[0], point_b[1]))
        c = self.canvas
        h = rect.height(c.screen_rect)
        self.set_stroke((red, green, blue), alpha, width)
        c.pdf.line(point_a[0], h - point_a[1], point_b[0], h - point_b[1])

    def draw_screen_lines(self, points, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF, xor=False):
        c = self.canvas
        h = rect.height(c.screen_rect)
        self.set_stroke((red, green, blue), alpha, width)
        if len(points) > 1:
            d = c.pdf
            p = d.beginPath()
            x, y = points[0]
            p.moveTo(x, h - y)
            for x, y in points[1:]:
                p.lineTo(x, h - y)
            d.drawPath(p, fill=0, stroke=1)

    def draw_screen_markers(self, markers, style):
        c = self.canvas
        self.set_stroke_style(style)

        # Markers use the same the fill color as the line color
        rgb, a = self.convert_color(style.line_color)
        c.pdf.setFillColor(rgb, a)

        for p1, p2, symbol in markers:
            marker_points, filled = style.get_marker_data(symbol)
            if marker_points is None:
                continue
            # Compute the angles in screen coordinates, because using world
            # coordinates for the angles results in the projection being applied,
            # which shows distortion as it moves away from the equator
            point = c.get_numpy_screen_point_from_world_point(p1)
            d = point - c.get_numpy_screen_point_from_world_point(p2)
            mag = np.linalg.norm(d)
            if mag > 0.0:
                d = d / np.linalg.norm(d)
            else:
                d[:] = (1, 0)
            r = np.array(((d[0], d[1]), (d[1], -d[0])), dtype=np.float32)
            points = (np.dot(marker_points, r) * style.line_width) + point
            #self.renderer.draw_screen_lines(a, self.style.line_width, smooth=True, color4b=self.style.line_color)
            h = rect.height(c.screen_rect)
            p = c.pdf.beginPath()
            x, y = points[0]
            p.moveTo(x, h - y)
            for x, y in points[1:]:
                log.debug("%f -> %f" % (x, y))
                p.lineTo(x, h - y)
            p.close()
            c.pdf.drawPath(p, fill=filled, stroke=1)

    def draw_screen_box(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0, width=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        pass

    def draw_screen_rect(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0):
        pass

    def get_drawn_string_dimensions(self, text):
        return self.canvas.get_font_metrics(text)

    def draw_screen_string(self, point, text):
        c = self.canvas
        h = rect.height(c.screen_rect) - c.font_scale
        log.debug("text %f,%f -> '%s'" % (point[0], h - point[1], text.encode("utf-8")))
        c.pdf.setFillColor((0.0, 0.0, 0.0), 1.0)
        c.pdf.drawString(point[0], h - point[1], text)
        if self.debug_text_bounding_box:
            dims = self.get_drawn_string_dimensions(text)
            c.pdf.rect(point[0], h - point[1], dims[0], dims[1], fill=0, stroke=1)

    # Vector object drawing routines

    def convert_color(self, color):
        r, g, b, a = int_to_color_floats(color)
        return (r, g, b), a

    def set_stroke_style(self, style):
        rgb, a = self.convert_color(style.line_color)
        self.set_stroke(rgb, a, style.line_width)
        return style.line_stipple > 0

    def set_stroke(self, rgb, a, width):
        d = self.canvas.pdf
        d.setStrokeColor(rgb, a)
        w = width / self.canvas.viewport_scale * self.canvas.linewidth_factor
        d.setLineWidth(w)
        d.setLineJoin(1)  # rounded
        d.setLineCap(1)  # rounded

    def set_fill_style(self, style):
        rgb, a = self.convert_color(style.fill_color)
        self.canvas.pdf.setFillColor(rgb, a)
        return style.fill_style > 0

    def fill_object(self, layer, picker, style):
        d = self.canvas.pdf
        if self.set_fill_style(style):
            p = d.beginPath()
            x, y = self.line_xys[0]
            p.moveTo(x, y)
            for x, y in self.line_xys[1:]:
                log.debug("fill_object: %f -> %f" % (x, y))
                p.lineTo(x, y)
            d.drawPath(p, fill=1, stroke=0)

    def outline_object(self, layer, picker, style):
        d = self.canvas.pdf
        if self.set_stroke_style(style):
            p = d.beginPath()
            x, y = self.line_xys[0]
            p.moveTo(x, y)
            for x, y in self.line_xys[1:]:
                log.debug("outline_object: %f -> %f" % (x, y))
                p.lineTo(x, y)
            d.drawPath(p, fill=0, stroke=1)
