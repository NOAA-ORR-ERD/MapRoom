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
// texcoord removed
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

    def on_initialize(self, event):
        # Build cube data
        V, I, O = create_cube()
        # Each item in the vertex data V is a tuple of lists, e.g.:
        #
        # ([1.0, 1.0, 1.0], [0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0])
        #
        # where each list corresponds to one of the attributes in the vertex
        # shader.  (See vispy.geometry.create_cube).  To experiment with this,
        # I've removed the 2nd tuple (the texcoord) and changed the vertex
        # shader from vispy5.  We'll see if this works. UPDATE: yes, it does!
        v2type = [('position', np.float32, 3),
                  ('normal', np.float32, 3),
                  ('color',    np.float32, 4)]
        v2 = np.zeros(V.shape[0], v2type)
        v2['position'] = V['position']
        v2['normal'] = V['normal']
        v2['color'] = V['color']
        print V[0]
        print v2[0]
        vertices = VertexBuffer(v2)
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
 
        # OpenGL initalization
        # --------------------------------------
        gloo.set_state(clear_color=(1.0, 1.0, 1.0, 1.00), depth_test=True,
                       polygon_offset=(1, 1), line_width=0.75,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

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
        native = self.canvas.native
        native.Show()
        
        native.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
#        native.Bind(wx.EVT_MOTION, self.on_mouse_motion)
#        native.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
#        native.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll)
#        native.Bind(wx.EVT_ENTER_WINDOW, self.on_mouse_enter)
#        native.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
#        native.Bind(wx.EVT_CHAR, self.on_key_char)
#        native.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
#        native.Bind(wx.EVT_KEY_DOWN, self.on_key_up)

    def on_mouse_down(self, event):
        # self.SetFocus() # why would it not be focused?
        print("in on_mouse_down: event=%s" % event)
        event.Skip()

    def on_new_window(self, event):
        frame = TestFrame()
        frame.Show(True)

    def on_quit(self, event):
        app.quit()

if __name__ == '__main__':
    myapp = wx.App(0)
    frame = TestFrame()
    frame.Show(True)
    app.run()
