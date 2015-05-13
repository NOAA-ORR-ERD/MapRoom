#!/usr/bin/env python
import wx
import wx.html

import numpy as np


def get_square(size):
    size = 100
    arr = np.empty((size, size, 4), np.uint8)

    # just some indexes to keep track of which byte is which
    R, G, B, A = range(4)

    red, green, blue, alpha = (35, 142,  35, 128)
    # initialize all pixel values to the values passed in
    arr[:,:,R] = red
    arr[:,:,G] = green
    arr[:,:,B] = blue
    arr[:,:,A] = alpha

    # Set the alpha for the border pixels to be fully opaque
    arr[0,      0:size, A] = wx.ALPHA_OPAQUE  # first row
    arr[size-1, 0:size, A] = wx.ALPHA_OPAQUE  # last row
    arr[0:size, 0,      A] = wx.ALPHA_OPAQUE  # first col
    arr[0:size, size-1, A] = wx.ALPHA_OPAQUE  # last col

    return arr


class OffScreenHTML(object):
    """
    test of rendering HTML to an off-screen bitmap

    This version uses a wx.GCDC, so you can have an alpha background.

    Works on OS-X, may need an explicite alpha bitmap on other platforms
    """
    
    def __init__(self, width):
        self.width = width
        self.height = 1
        self.bitmap = wx.EmptyBitmap(self.width, self.height)
        
        self.hr = wx.html.HtmlDCRenderer()
        
        # a bunch of defaults...
        self.bg = (255, 255, 255)
        self.padding = 10
    
    def draw(self, text):
        DC = wx.MemoryDC()
        DC.SelectObject(self.bitmap)
        DC = wx.GCDC(DC)
        DC.SetBackground(wx.Brush(self.bg))
        DC.Clear()
        
        self.hr.SetDC(DC, 1.0)
        self.hr.SetSize(self.width-2*self.padding, self.height)
        self.hr.SetFonts("Deja Vu Serif", "Deja Vu Sans Mono")
        
        self.hr.SetHtmlText(text)
        
        return DC
       
    def render(self, source):
        """
        Render the html source to the bitmap
        """
        dc = self.draw(source)
        needed = self.hr.GetTotalHeight() + 2*self.padding
        print "needed", needed
        if needed > self.height:
            self.height = needed
            self.bitmap = wx.EmptyBitmap(self.width, self.height)
            dc = self.draw(source)
            
        self.hr.Render(self.padding, self.padding, [])
        self.rendered_size = (self.width, self.hr.GetTotalHeight()+2*self.padding)
        print self.rendered_size
        print self.bitmap.GetSize()

    def get_numpy(self, text):
        self.render(text)
        w, h = self.rendered_size
        sub = self.bitmap.GetSubBitmap(wx.Rect(0, 0, w, h))
        arr = np.empty((h, w, 4), np.uint8)
        sub.CopyToBuffer(arr, format=wx.BitmapBufferFormat_RGBA)
        return arr

    def get_png(self, text, filename):
        self.render(text)
        sub = self.bitmap.GetSubBitmap(wx.Rect(0, 0, *self.rendered_size) )
        sub.SaveFile(filename, wx.BITMAP_TYPE_PNG)


if __name__ == "__main__":
    HTML = """<h1> A Title </h1>
    
    <p>This is a simple test of rendering a little text with an html renderer</p>
    
    <p>
    Now another paragraph, with a bunch more text in it. This is a test
    that will show if it can take care of the basic rendering for us, and
    auto-wrap, and all sorts of nifty stuff like that
    </p>
    <p>
    and here is some <b> Bold Text </b>
    <p> It does seem to work OK </p>
    """
    
    App = wx.App(False)
    OSR = OffScreenHTML(200)
    OSR.get_png(HTML, 'junk.png')
    print OSR.get_numpy(HTML)
