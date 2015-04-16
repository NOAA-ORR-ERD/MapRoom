import math

import wx
import numpy as np

import OpenGL.GL as gl


class GLDC(object):
    def __init__(self, renderer):
        self.renderer = renderer
    
    def BeginDrawing(self):
        gl.glDisable(gl.GL_TEXTURE_2D) # don't let texture colors override line color
        gl.glDisable(gl.GL_LINE_SMOOTH)
    
    def EndDrawing(self):
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glDisable(gl.GL_LINE_STIPPLE)
    
    def Blit(self, xdest, ydest, width, height, source, xsrc, ysrc, rop=wx.COPY, useMask=False, xsrcMask=-1, ysrcMask=-1):
        pass
    
    def Clear(self):
        pass
    
    def DrawArc(self, x1, y1, x2, y2, xc, yc):
        pass
    
    def DrawArcPoint(self, pt1, pt2, center):
        pass
    
    def DrawBitmap(self, bmp, x, y, useMask=False):
        pass
    
    def DrawBitmapPoint(self, bmp, pt, useMask=False):
        pass
    
    def DrawCircle(self, cx, cy, r):
        print "DRAWCIRCLE!", cx, cy, r
        # Simple rasterization from public domain algorithm
        # http://slabode.exofire.net/circle_draw.shtml
        num_segments = 128
        theta = 2 * 3.1415926 / num_segments
        c = math.cos(theta) # precalculate the sine and cosine
        s = math.sin(theta)
        
        x = r # we start at angle = 0 
        y = 0
        
        gl.glBegin(gl.GL_LINE_LOOP);
        while num_segments > 0:
            gl.glVertex2f(x + cx, y + cy)
            t = x
            x = c * x - s * y
            y = s * t + c * y
            num_segments -= 1
        gl.glEnd()
    
    def DrawCirclePoint(self, pt, radius):
        self.DrawCircle(pt[0], pt[1], radius)
    
    def DrawEllipse(self, x, y, width, height):
        pass
    
    def DrawEllipseList(self, ellipses, pens=None, brushes=None):
        pass
    
    def DrawEllipsePointSize(self, pt, sz):
        pass
    
    def DrawEllipseRect(self, rect):
        pass
    
    def DrawEllipticArc(self, x, y, w, h, start, end):
        pass
    
    def DrawEllipticArcPointSize(self, pt, sz, start, end):
        pass
    
    def DrawLabel(self, text, rect, alignment=wx.ALIGN_LEFT|wx.ALIGN_TOP, indexAccel=-1):
        pass
    
    def DrawLine(self, x1, y1, x2, y2):
        print "DRAWLINE!", x1, y1, x2, y2
        gl.glBegin(gl.GL_LINE_STRIP)
        gl.glVertex(x1, y1, 0)
        gl.glVertex(x2, y2, 0)
        gl.glEnd()
    
    def DrawLineList(self, lines, pens=None):
        print "DRAWLINELIST!", lines[0][0], lines[0][1], lines[1][0], lines[1][1], "..."
        gl.glBegin(gl.GL_LINES)
        for line in lines:
            gl.glVertex(line[0], line[1], 0)
            gl.glVertex(line[2], line[3], 0)
        gl.glEnd()
    
    def DrawLinePoint(self, pt1, pt2):
        gl.glBegin(gl.GL_LINE_STRIP)
        gl.glVertex(pt1.x, pt1.y, 0)
        gl.glVertex(pt2.x, pt2.y, 0)
        gl.glEnd()
    
    def DrawLines(self, points, xoffset=0, yoffset=0):
        print "DRAWLINES!", points[0][0], points[0][1], points[1][0], points[1][1], "..."
        gl.glBegin(gl.GL_LINE_STRIP)
        for point in points:
            gl.glVertex(point[0] + xoffset, point[1] + yoffset, 0)
        gl.glEnd()
    
    def DrawPoint(self, x, y):
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex(x, y, 0)
        gl.glEnd()
    
    def DrawPointList(self, points, pens=None):
        gl.glBegin(gl.GL_POINTS)
        for p in points:
            gl.glVertex(p[0], p[1], 0)
        gl.glEnd()
    
    def DrawPointPoint(self, pt):
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex(pt.x, pt.y, 0)
        gl.glEnd()
    
    def DrawPolygon(self, points, xoffset=0, yoffset=0, fillStyle=wx.ODDEVEN_RULE):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBegin(gl.GL_LINE_LOOP)
        for point in points:
            gl.glVertex(point[0] + xoffset, point[1] + yoffset, 0)
        gl.glEnd()
    
    def DrawPolygonList(self, polygons, pens=None, brushes=None):
        pass
    
    def DrawRectangle(self, x, y, width, height):
        print "DRAWRECTANGLE!", x, y, width, height
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex(x, y, 0)
        gl.glVertex(x + width, y, 0)
        gl.glVertex(x + width, y + height, 0)
        gl.glVertex(x, y + height, 0)
        gl.glEnd()
    
    def DrawRectangleList(self, rectangles, pens=None, brushes=None):
        pass
    
    def DrawRectanglePointSize(self, pt, sz):
        print "DRAWRECTANGLEPOINTSIZE!", pt, sz
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex(pt[0], pt[1], 0)
        gl.glVertex(pt[0] + sz[0], pt[1], 0)
        gl.glVertex(pt[0] + sz[0], pt[1] + sz[1], 0)
        gl.glVertex(pt[0], pt[1] + sz[1], 0)
        gl.glEnd()
        pass
    
    def DrawRectangleRect(self, rect):
        pass
    
    def DrawRotatedText(self, text, x, y, angle):
        pass
    
    def DrawRotatedTextPoint(self, text, pt, angle):
        pass
    
    def DrawRoundedRectangle(self, x, y, width, height, radius):
        pass
    
    def DrawRoundedRectanglePointSize(self, pt, sz, radius):
        pass
    
    def DrawRoundedRectangleRect(self, r, radius):
        pass
    
    def DrawSpline(self, points):
        pass
    
    def DrawText(self, text, x, y):
        pass
    
    def DrawTextList(self, textList, coords, foregrounds=None, backgrounds=None):
        pass
    
    def DrawTextPoint(self, text, pt):
        pass
    
    def SetBackground(self, brush):
        pass
    
    def SetBackgroundMode(self, mode):
        pass
    
    def SetBrush(self, brush):
        pass
    
    def SetClippingRect(self, rect):
        pass
    
    def SetClippingRegion(self, x, y, width, height):
        pass
    
    def SetPen(self, pen):
        color = pen.GetColour()
        r, g, b, a = color.Get(True)
        gl.glColor(r/255., g/255., b/255., a/255.)
        gl.glLineWidth(pen.GetWidth())
        gl.glPointSize(pen.GetWidth())
        #gl.glLineStipple(stipple_factor, stipple_pattern)
