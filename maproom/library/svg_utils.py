import xml.etree.cElementTree as ET

import OpenGL.GL as gl

import glsvg
from glsvg.glutils import CurrentTransform, DisplayListGenerator


class SVGOverlay(glsvg.SVGDoc):
    """Subclass that renders SVG images without clearing the screen
    """

    def __init__(self, xmltext):
        root_element = ET.fromstring(xmltext)
        glsvg.SVGDoc.__init__(self, root_element)

    def _generate_disp_list(self):
        # NOTE: the prerendering seems to place additional small copies of the
        # patterns on the screen, and removing the cause of this (the two calls
        # to prerender_* below) doesn't seem to affect the compass rose sample
        # image, so I'm taking them out.

        # prepare all the patterns
        #self.prerender_patterns()

        # prepare all the predefined paths
        #self.prerender_defs()

        with DisplayListGenerator() as display_list:
            self.disp_list = display_list
            self.render()

    def flipped_anchor(self):
        if self._anchor_y == 'bottom':
            flip_y = self.height
        elif self._anchor_y == 'center':
            flip_y = self.height * .5
        elif self._anchor_y == 'top':
            flip_y = 0
        else:
            flip_y = self.height - self.anchor_y
        return flip_y

    def draw(self, x, y, z=0, angle=0, scale=1):
        """Version of draw without clearing the screen
        """
        with CurrentTransform():
            gl.glTranslatef(x, y, z)
            if angle:
                gl.glRotatef(angle, 0, 0, 1)
            if scale != 1:
                try:
                    gl.glScalef(scale[0], scale[1], 1)
                except TypeError:
                    gl.glScalef(scale, scale, 1)
            if self._a_x or self._a_y:
                gl.glTranslatef(-self._a_x, -self._a_y, 0)

            # Flip the Y axis for difference in OpenGL and SVG coord sys
            gl.glScalef(1, -1, 1)
            gl.glTranslatef(0, -self.flipped_anchor(), 0)

            self.disp_list()
