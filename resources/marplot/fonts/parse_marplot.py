#!/usr/bin/env python

import os
import sys
import re

import wx
import numpy as np
from pyface.api import ImageResource


def write_png(arr):
    # from http://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
    import zlib, struct

    buf = arr.tostring()
    width = arr.shape[1]
    height = arr.shape[0]
    width_byte_4 = width * 4
#    for span in range(0, (height - 1) * width * 4, width_byte_4):
#        print span, span + width_byte_4
#    for span in range((height - 1) * width * 4, -1, - width_byte_4):
#        print span, span + width_byte_4, repr(b'\x00' + buf[span:span + width_byte_4])
#    # reverse the vertical line order and add null bytes at the start
#    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
#                        for span in range((height - 1) * width * 4, -1, - width_byte_4))
    # Why was that reversed? Seems to print the image upside down.  Maybe that
    # was for a non-numpy image
    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                        for span in range(0, (height) * width * 4, width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

def get_numpy_from_marplot(icon_path, r=0, g=128, b=128):
    image = ImageResource(icon_path)
    bitmap = image.create_bitmap()
    arr = np.empty((bitmap.Height, bitmap.Width, 4), np.uint8)
    bitmap.CopyToBuffer(arr, format=wx.BitmapBufferFormat_RGBA)
#    # Marplot icons have white foreground which is not ideal for
#    # us as we'll usually be printing on white backgrounds, so the
#    # optional color values passed in will replace white.  Thanks to
#    # http://stackoverflow.com/questions/6483489/
#    red, green, blue = arr[:,:,0], arr[:,:,1], arr[:,:,2]
#    mask = (red == 255) & (green == 255) & (blue == 255)
#    arr[:,:,:3][mask] = [r, g, b]
    return arr

page = {
    1: get_numpy_from_marplot("marplot_font1_0.png"),
    2: get_numpy_from_marplot("marplot_font2_0.png"),
}

class Icon(object):
    def __init__(self, items):
        for param in items:
            k, v = param.split("=")
            if k == "name":
                setattr(self, k, v)
            else:
                setattr(self, k, int(v))
        self.data = page[self.page][self.y:self.y+self.height,self.x:self.x+self.width,:]
    
    def __cmp__(self, other):
        if self.name < other.name:
            return -1
        elif self.name > other.name:
            return 1
        return 0
    
    def __str__(self):
        return "%s: %dx%d@%d,%d on p=%d" % (self.name, self.width, self.height, self.x, self.y, self.page)
    
    def __repr__(self):
        return "%s: %dx%d@%d,%d on p=%d" % (self.name, self.width, self.height, self.x, self.y, self.page)
    
    def get_py(self):
        return self.name, self.width, self.height, self.xoffset, self.yoffset, self.data.tostring()
    
    def get_png(self):
        png = write_png(self.data)
        #arr = np.fromstring(self.data.tostring(), dtype=np.uint8).reshape((self.height, self.width, 4))
        return png

class AddrParser(object):
    def __init__(self, filename):
        self.filename = filename
        self.fileroot, _ = os.path.splitext(os.path.basename(self.filename))
        self.max_width = -1
        self.max_height = -1
        self.parse()
    
    def parse(self):
        self.icon_list = {}
        category = ""
        icons = []
        with open(self.filename) as fh:
            text = fh.read()
            for line in text.splitlines():
                items = line.split(None, 10)
                if len(items) == 0:
                    continue
                if len(items) < 9:
                    if icons:
                        self.icon_list[category] = icons
                        icons = []
                    category = items[0]
                else:
                    if category == "[Unused]":
                        continue
                    icon = Icon(items)
                    icons.append(icon)
                    self.max_width = max(self.max_width, icon.width)
                    self.max_height = max(self.max_height, icon.height)
        
        if icons:
            self.icon_list[category] = icons

if __name__ == "__main__":
    mm = AddrParser("marplot_font.txt")
#    print mm.icon_list
#    print mm.icon_list.keys()
#    print mm.icon_list['Logos']
    icon = mm.icon_list['Alphabet'][0]
#    print icon
#    print icon.data
    png = icon.get_png()
    with open("test.png", 'wb') as fh:
        fh.write(png)
#    print repr(png)
    
    icon_count = 0
    png_data = []
    
    lines = [
        "marplot_icons = [",
        ]

    for category in sorted(mm.icon_list.keys()):
        lines.append("  ('%s', [" % category)
        for icon in sorted(mm.icon_list[category]):
            lines.append("    ('%s', %d)," % (icon.name, icon_count))
            icon_count += 1
            png_data.append(repr(icon.get_png()))
        lines.append("  ]),")
    lines.append("]")
    lines.append("marplot_icon_data = [")
    lines.append(",\n".join(png_data))
    lines.append("]")
    
    header = """# Automatically generated file, DO NOT EDIT!",
# Generated from resources/marplot/fonts/parse_marplot.py",

def get_wx_bitmap(icon_num):
    import wx
    import cStringIO
    
    data = marplot_icon_data[icon_num]
    image = wx.ImageFromStream(cStringIO.StringIO(data))
    bitmap = wx.BitmapFromImage(image)
    return bitmap

def get_numpy_bitmap(icon_num):
    from PIL import Image
    import numpy as np
    import cStringIO
    
    data = marplot_icon_data[icon_num]
    image = Image.open(cStringIO.StringIO(data))
    return np.array(image)

marplot_icon_max_size = (%d, %d)

""" % (mm.max_width, mm.max_height)
    with open("../../../maproom/library/marplot_icons.py", 'w') as fh:
        fh.write(header)
        fh.write("\n".join(lines) + "\n")
    
#    print page[1]
#    mm.replace()
#    with open("test.html", "w") as fh:
#        fh.write(mm.header)
#        fh.write(mm.new_body)
#        fh.write(mm.footer)
