#!/usr/bin/env python
import io

# FIXME: really weird error on macos when using py.test. The import of
# docutils.core raises an exception about an invalid locale. Somehow,
# the locale get reset to just plain "UTF-8", even though the
# environmental variable for LC_CTYPE, LC_ALL, LC_TIME, LANG are all
# set to "en_US.UTF-8".
import _locale
if _locale.setlocale(_locale.LC_CTYPE) == "UTF-8":
    _locale.setlocale(_locale.LC_CTYPE, "en_US.UTF-8")
from docutils.core import publish_parts

import wx
import wx.html

import numpy as np
from PIL import Image
import markdown

import cgi

# from pyface.api import ImageResource

import logging
log = logging.getLogger(__name__)


def get_numpy_from_marplot_icon(icon_path, r=0, g=128, b=128):
    # FIXME: this is based on old pyface code. Is it still used any more?
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


def get_numpy_from_data(data):
    image = Image.open(io.BytesIO(data))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    arr = np.array(image)
    return arr


def get_bitmap_from_numpy(array):
    image = wx.Image(array.shape[1], array.shape[0])
    image.SetData(array.tostring())
    bmp = wx.Bitmap(image)
    return bmp


def get_rect(w, h):
    arr = np.empty((h, w, 4), np.uint8)

    # just some indexes to keep track of which byte is which
    R, G, B, A = list(range(4))

    red, green, blue, alpha = (35, 142, 35, 128)
    # initialize all pixel values to the values passed in
    arr[:,:,R] = red
    arr[:,:,G] = green
    arr[:,:,B] = blue
    arr[:,:,A] = alpha

    # Set the alpha for the border pixels to be fully opaque
    arr[0,0:w,A] = wx.ALPHA_OPAQUE  # first row
    arr[h-1,0:w,A] = wx.ALPHA_OPAQUE  # last row
    arr[0:h,0,A] = wx.ALPHA_OPAQUE  # first col
    arr[0:h,w-1,A] = wx.ALPHA_OPAQUE  # last col

    return arr


def get_checkerboard(w, h, sq=16):
    # Algorithm from http://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    color1 = (64, 64, 64, 32)
    color2 = (255, 255, 255, 32)
    coords = np.ogrid[0:h, 0:w]
    idx = (coords[0] // sq + coords[1] // sq) & 1
    vals = np.array([color1, color2], dtype=np.uint8)
    arr = vals[idx]
    return arr


def get_square(size):
    return get_rect(size, size)


def simple_text_formatter(text):
    text = cgi.escape(text)
    # double returns are a new paragraph
    text = text.split("\n\n")
    text = [p.strip().replace("\n", "<br>") for p in text]
    text = "<p>\n" + "\n</p>\n<p>\n".join(text) + "\n</p>"

    return text


def rst_text_formatter(text):
    # rst creates a document class for the first section heading, so force a
    # document header which puts all other section headings into section divs.
    # But of course that means removing the fake header and div from the
    # output.
    wrap = "======\rREMOVE\n======\n\nREMOVETHIS\n\n" + text
    text = publish_parts(wrap, writer_name='html')['html_body']
    _, wrapped = text.split("REMOVETHIS</p>", 1)
    if "</div>" in wrapped:
        wrapped, _ = wrapped.rsplit("</div>", 1)
    return wrapped


def markdown_text_formatter(text):
    text = markdown.markdown(text)
    return text


class OffScreenHTML(object):
    """
    test of rendering HTML to an off-screen bitmap

    This version uses a wx.GCDC, so you can have an alpha background.

    Works on OS-X, may need an explicite alpha bitmap on other platforms

    Specifying a height of -1 will return an image that contains the entire
    height of the text
    """

    def __init__(self, width=200, height=100, bg=(255, 255, 255)):
        self.width = int(width)
        self.height = int(height)

        self.hr = wx.html.HtmlDCRenderer()

        # background will be transformed into transparent in get_numpy
        self.bg = tuple(bg[0:3])  # throw away alpha value, if any

    def setup(self, text, bitmap, face, size):
        DC = wx.MemoryDC()
        DC.SelectObject(bitmap)
        DC = wx.GCDC(DC)
        DC.SetBackground(wx.Brush(self.bg))
        DC.Clear()

        self.hr.SetDC(DC, 1.0)
        self.hr.SetSize(bitmap.Width, self.height)
        self.hr.SetStandardFonts(size, face, "Deja Vu Sans Mono")

        self.hr.SetHtmlText(text)

        return DC

    def render(self, source, face, size):
        """
        Render the html source to the bitmap
        """
        bitmap = wx.Bitmap(self.width, 100)
        dc = self.setup(source, bitmap, face, size)

        # Calculate the height of the final rendered text
        y = ylast = 0
        while True:
            y = self.hr.Render(0, 0, [], y, True, y + self.height)
            if y == ylast or y < 0:
                break
            ylast = y
        if ylast < self.height:
            ylast = self.height
        bitmap = wx.Bitmap(self.width, ylast)
        dc = self.setup(source, bitmap, face, size)
        self.hr.Render(0, 0, [])

        if self.height < 0:
            h = self.hr.GetTotalHeight()
        else:
            # NOTE: no built-in way to get the bounding width from wx; i.e.  no
            # analogue to GetTotalHeight
            h = self.height
        return self.width, h, bitmap

    def get_numpy(self, text, c=None, face="", size=12, text_format=0):
        if self.width < 1 or self.height < 1:
            return self.get_blank()

        if text_format == 0:
            text = simple_text_formatter(text)
        elif text_format == 2:
            text = rst_text_formatter(text)
        elif text_format == 3:
            text = markdown_text_formatter(text)
        if c is not None:
            text = "<font color='%s'>%s</font>" % (c, text)
        w, h, bitmap = self.render(text, face, size)
        log.debug("bitmap: %s" % str([w, h, self.width, self.height, bitmap.Width, bitmap.Height]))
        if self.width < 1 or self.height < 1:
            return self.get_blank()
        if h > 0 and w > 0:
            sub = bitmap.GetSubBitmap(wx.Rect(0, 0, w, h))
            arr = np.empty((h, w, 4), np.uint8)
            sub.CopyToBuffer(arr, format=wx.BitmapBufferFormat_RGBA)
            # Turn background transparent
            red, green, blue = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            bg = self.bg
            mask = (red == bg[0]) & (green == bg[1]) & (blue == bg[2])
            arr[:,:,3][mask] = 0

            # Compute bounding box of text by looking at the mask.  The mask
            # contains all those pixels that are only background color,
            # so the bounding box can be computed by using the idea from
            # http://stackoverflow.com/questions/4808221
            fg = np.argwhere(np.logical_not(mask))
            if np.alen(fg) > 0:
                (ystart, xstart), (ystop, xstop) = fg.min(0), fg.max(0) + 1
                bb = arr[ystart:ystop, xstart:xstop]
            else:
                # background and text color must have been the same because it
                # didn't find any bounding box
                bb = self.get_blank()
        else:
            # he HTML renderer doesn't render anything when the input is empty
            # or only whitespace, so need to return a fake (blank) image
            bb = self.get_blank()

        return bb

    def get_blank(self):
        return np.asarray([255, 255, 255, 0], dtype=np.uint8).reshape((1, 1, 4))

    def get_png(self, text, filename):
        w, h, bitmap = self.render(text)
        sub = bitmap.GetSubBitmap(wx.Rect(0, 0, w, h))
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
    print(OSR.get_numpy(HTML))
