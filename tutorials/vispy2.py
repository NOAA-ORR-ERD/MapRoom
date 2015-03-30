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
from vispy import app, gloo
from vispy.gloo import Program

vertex = """
    attribute vec4 color;
    attribute vec2 position;
    varying vec4 v_color;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_color = color;
    } """

fragment = """
    varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    } """


class Canvas(app.Canvas):
    def __init__(self, *args, **kwargs):
        app.Canvas.__init__(self, *args, **kwargs)
        
        # Build program & data
        self.program = Program(vertex, fragment, count=4)
        self.program['color'] = [(1, 0, 0, 1), (0, 1, 0, 1),
                                 (0, 0, 1, 1), (1, 1, 0, 1)]
        self.program['position'] = [(-1, -1), (-1, +1),
                                    (+1, -1), (+1, +1)]

    def on_draw(self, event):
        gloo.clear(color='white')
        self.program.draw('triangle_strip')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)


class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, "Vispy Test",
                          wx.DefaultPosition, size=(500, 500))

        MenuBar = wx.MenuBar()
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_EXIT, "&Quit")
        self.Bind(wx.EVT_MENU, self.on_quit, id=wx.ID_EXIT)
        MenuBar.Append(file_menu, "&File")
        self.SetMenuBar(MenuBar)

        self.canvas = Canvas(app="wx", parent=self)
        self.canvas.native.Show()

    def on_quit(self, event):
        self.Close(True)

if __name__ == '__main__':
    myapp = wx.App(0)
    frame = TestFrame()
    frame.Show(True)
    myapp.MainLoop()
