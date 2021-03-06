# -*- coding: utf-8 -*-
# vispy: testskip
# Copyright (c) 2014, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
This is a very minimal example that opens a window and makes the background
color to change from black to white to black ...

The wx backend is used to embed the canvas in a simple wx Frame with
a menubar.
"""

import wx
import math
import numpy as np

from vispy import app, gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import create_cube

vertex = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec4 u_color;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

varying vec4 v_color;
void main()
{
    v_color = u_color * color;
    gl_Position = u_projection * u_view * u_model * vec4(position,1.0);
}
"""

fragment = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""


class Canvas(app.Canvas):
    def __init__(self, *args, **kwargs):
        app.Canvas.__init__(self, *args, **kwargs)
        self.timer = app.Timer('auto', self.on_timer)

    def on_initialize(self, event):
        # Build cube data
        V, I, O = create_cube()
        vertices = VertexBuffer(V)
        self.faces = IndexBuffer(I)
        self.outline = IndexBuffer(O)

        # Build program
        # --------------------------------------
        self.program = Program(vertex, fragment)
        self.program.bind(vertices)

        # Build view, model, projection & normal
        # --------------------------------------
        view = np.eye(4, dtype=np.float32)
        model = np.eye(4, dtype=np.float32)
        translate(view, 0, 0, -5)
        self.program['u_model'] = model
        self.program['u_view'] = view
        self.phi, self.theta = 0, 0

        # OpenGL initalization
        # --------------------------------------
        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True,
                       polygon_offset=(1, 1), line_width=0.75,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.timer.start()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)

        # Filled cube
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program['u_color'] = 1, 1, 1, 1
        self.program.draw('triangles', self.faces)

        # Outlined cube
        gloo.set_state(blend=True, depth_mask=False, polygon_offset_fill=False)
        self.program['u_color'] = 0, 0, 0, 1
        self.program.draw('lines', self.outline)
        gloo.set_state(depth_mask=True)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)
        projection = perspective(45.0, event.size[0] / float(event.size[1]),
                                 2.0, 10.0)
        self.program['u_projection'] = projection

    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        model = np.eye(4, dtype=np.float32)
        rotate(model, self.theta, 0, 0, 1)
        rotate(model, self.phi, 0, 1, 0)
        self.program['u_model'] = model
        self.update()


class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, "Vispy Test",
                          wx.DefaultPosition, size=(500, 500))

        MenuBar = wx.MenuBar()
        file_menu = wx.Menu()
        file_menu.Append(-1, "New Window")
        self.Bind(wx.EVT_MENU, self.on_new_window)
        file_menu.Append(wx.ID_EXIT, "&Quit")
        self.Bind(wx.EVT_MENU, self.on_quit, id=wx.ID_EXIT)
        MenuBar.Append(file_menu, "&File")
        self.SetMenuBar(MenuBar)

        self.canvas = Canvas(app="wx", parent=self)
        self.canvas.native.Show()

    def on_new_window(self, event):
        frame = TestFrame()
        frame.Show(True)

    def on_quit(self, event):
        self.Close(True)

if __name__ == '__main__':
    myapp = wx.App(0)
    frame = TestFrame()
    frame.Show(True)
    myapp.MainLoop()
