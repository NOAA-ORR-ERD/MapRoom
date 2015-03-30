"""Simple wx/opengl demo

Before vispy/gloo attempt
"""


import sys
import ctypes

import numpy as np

import wx
import wx.glcanvas as glcanvas

import OpenGL.GL as gl
from OpenGL.GL import shaders
import OpenGL.GLUT as glut
from OpenGL.arrays import vbo



class MyCanvasBase(glcanvas.GLCanvas):
    def __init__(self, parent):
        wx.glcanvas.GLCanvas.__init__(self, parent, -1)
        self.init = False
        self.context = glcanvas.GLContext(self)
        
        # initial mouse position
        self.lastx = self.x = 30
        self.lasty = self.y = 30
        self.size = None
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        event.Skip()

    def DoSetViewport(self):
        size = self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        gl.glViewport(0, 0, size.width, size.height)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()

    def OnMouseDown(self, evt):
        self.CaptureMouse()
        self.x, self.y = self.lastx, self.lasty = evt.GetPosition()

    def OnMouseUp(self, evt):
        self.ReleaseMouse()

    def OnMouseMotion(self, evt):
        if evt.Dragging() and evt.LeftIsDown():
            self.lastx, self.lasty = self.x, self.y
            self.x, self.y = evt.GetPosition()
            self.Refresh(False)

    def OnTimer(self, event):
        pass

    def start_timer(self):
        self.timer.Start(10)


vertex_code = """
    uniform float scale;
    attribute vec4 color;
    attribute vec2 position;
    varying vec4 v_color;
    void main()
    {
        gl_Position = vec4(scale*position, 0.0, 1.0);
        v_color = color;
    } """

fragment_code = """
    varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    } """

class CubeCanvas(MyCanvasBase):
    def InitGL(self):
        data = np.zeros(4, [("position", np.float32, 2),
                            ("color",    np.float32, 4)])
        data['color']    = [ (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,0,1) ]
        data['position'] = [ (-1,-1),   (-1,+1),   (+1,-1),   (+1,+1)   ]

        # Build & activate program
        # --------------------------------------

        # Request a program and shader slots from GPU
        self.program  = gl.glCreateProgram()
        vertex   = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        # Set shaders source
        gl.glShaderSource(vertex, vertex_code)
        gl.glShaderSource(fragment, fragment_code)

        # Compile shaders
        gl.glCompileShader(vertex)
        gl.glCompileShader(fragment)

        # Attach shader objects to the program
        gl.glAttachShader(self.program, vertex)
        gl.glAttachShader(self.program, fragment)

        # Build program
        gl.glLinkProgram(self.program)

        # Get rid of shaders (no more needed)
        gl.glDetachShader(self.program, vertex)
        gl.glDetachShader(self.program, fragment)

        # Make program the default program
        gl.glUseProgram(self.program)


        # Build buffer
        # --------------------------------------

        # Request a buffer slot from GPU
        buffer = gl.glGenBuffers(1)

        # Make this buffer the default one
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

        # Upload data
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


        # Bind attributes
        # --------------------------------------
        stride = data.strides[0]
        offset = ctypes.c_void_p(0)
        loc = gl.glGetAttribLocation(self.program, "position")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

        offset = ctypes.c_void_p(data.dtype["position"].itemsize)
        loc = gl.glGetAttribLocation(self.program, "color")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

        # Bind uniforms
        # --------------------------------------
        self.ticks = 0
        self.scale = 1.0
        loc = gl.glGetUniformLocation(self.program, "scale")
        gl.glUniform1f(loc, self.scale)

    def OnDraw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        self.SwapBuffers()

    def OnTimer(self, event):
        self.ticks += 0.005 * 1000.0/60.0
        self.scale = (1+np.cos(self.ticks))/2.0
        print "Timer!", self.scale
        loc = gl.glGetUniformLocation(self.program, "scale")
        gl.glUniform1f(loc, self.scale)
        self.OnPaint(event)
        


if __name__ == '__main__':
    app = wx.App(0)
    frame = wx.Frame(None, -1, "Cube #1", size=(400,400))
    canvas = CubeCanvas(frame)
    frame.Show(True)
    canvas.start_timer()
    app.MainLoop()
