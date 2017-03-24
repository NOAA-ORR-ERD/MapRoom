import wx

import numpy as np

from omnivore import get_image_path

from font_extents import FONT_EXTENTS


def load_font_texture_with_alpha():
    font_path = get_image_path("font.png", __name__, up_one_level=True)
    image = wx.Image(font_path, wx.BITMAP_TYPE_PNG)
    width = image.GetWidth()
    height = image.GetHeight()
    buffer = np.frombuffer(image.GetDataBuffer(), np.uint8).reshape(
        (width, height, 3),
    )

    # Make an alpha channel that is opaque where the pixels are black
    # and semi-transparent where the pixels are white.
    buffer_with_alpha = np.empty((width, height, 4), np.uint8)
    buffer_with_alpha[:,:,0: 3] = buffer
    buffer_with_alpha[:,:,3] = (
        255 - buffer[:,:,0: 3].sum(axis=2) / 3
    ).clip(180, 255)

    return buffer_with_alpha, FONT_EXTENTS
