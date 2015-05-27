#!/usr/bin/env python
import wx
import wx.html

import numpy as np

from pyface.api import ImageResource

def get_numpy_from_marplot_icon(icon_path, r=0, g=128, b=128):
    image = ImageResource(icon_path)
    bitmap = image.create_bitmap()
    arr = np.empty((bitmap.Height, bitmap.Width, 4), np.uint8)
    bitmap.CopyToBuffer(arr, format=wx.BitmapBufferFormat_RGBA)
    # Marplot icons have white foreground which is not ideal for
    # us as we'll usually be printing on white backgrounds, so the
    # optional color values passed in will replace white.  Thanks to
    # http://stackoverflow.com/questions/6483489/
    red, green, blue = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mask = (red == 255) & (green == 255) & (blue == 255)
    arr[:,:,:3][mask] = [r, g, b]
    return arr


def get_square(size):
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
    
    def __init__(self):
        width, height = wx.GetDisplaySize()
        self.width = width
        self.height = 100
        
        self.hr = wx.html.HtmlDCRenderer()
        
        # White background will be transformed into transparent in get_numpy
        self.bg = (255, 255, 255)
    
    def draw(self, text, bitmap, face, size):
        DC = wx.MemoryDC()
        DC.SelectObject(bitmap)
        DC = wx.GCDC(DC)
        DC.SetBackground(wx.Brush(self.bg))
        DC.Clear()
        
        self.hr.SetDC(DC, 1.0)
        self.hr.SetSize(self.width, self.height)
        self.hr.SetStandardFonts(size, face, "Deja Vu Sans Mono")
        
        self.hr.SetHtmlText(text)
        
        return DC
       
    def render(self, source, face, size):
        """
        Render the html source to the bitmap
        """
        bitmap = wx.EmptyBitmap(self.width, self.height)
        dc = self.draw(source, bitmap, face, size)
        needed = self.hr.GetTotalHeight()
        if needed > self.height:
            bitmap = wx.EmptyBitmap(self.width, needed)
            dc = self.draw(source, bitmap, face, size)
            
        self.hr.Render(0, 0, [])
        # NOTE: no built-in way to get the bounding width from wx; i.e.  no
        # analogue to GetTotalHeight
        self.rendered_size = (self.width, self.hr.GetTotalHeight())
        return bitmap

    def get_numpy(self, text, c=None, face="", size=12):
        if c is not None:
            text = "<font color='%s'>%s</font>" % (c, text)
        bitmap = self.render(text, face, size)
        w, h = self.rendered_size
        if h > 0:
            sub = bitmap.GetSubBitmap(wx.Rect(0, 0, w, h))
            arr = np.empty((h, w, 4), np.uint8)
            sub.CopyToBuffer(arr, format=wx.BitmapBufferFormat_RGBA)
            # Turn background transparent
            red, green, blue = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            mask = (red == 255) & (green == 255) & (blue == 255)
            arr[:,:,3][mask] = 0
            
            # Compute bounding box of text by looking at the mask.  The mask
            # contains all those pixels that are only background color,
            # so the bounding box can be computed by using the idea from
            # http://stackoverflow.com/questions/4808221
            fg = np.argwhere(np.logical_not(mask))
            (ystart, xstart), (ystop, xstop) = fg.min(0), fg.max(0) + 1
            bb = arr[ystart:ystop, xstart:xstop]
        else:
            # he HTML renderer doesn't render anything when the input is empty
            # or only whitespace, so need to return a fake (blank) image
            bb = np.asarray([255, 255, 255, 0], dtype=np.uint8).reshape((1, 1, 4))

        return bb

    def get_png(self, text, filename):
        bitmap = self.render(text)
        sub = bitmap.GetSubBitmap(wx.Rect(0, 0, *self.rendered_size) )
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
