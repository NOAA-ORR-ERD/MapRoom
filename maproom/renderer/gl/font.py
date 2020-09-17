import wx

import numpy as np
import OpenGL.GL as gl

from maproom.app_framework.filesystem import get_image_path

from .font_extents import FONT_EXTENTS


def load_font_texture_with_alpha():
    font_path = get_image_path("font.png", file=__name__)
    image = wx.Image(font_path, wx.BITMAP_TYPE_PNG)
    width = image.GetWidth()
    height = image.GetHeight()
    data = image.GetData()
    buffer = np.frombuffer(data, np.uint8).reshape(
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


def load_font_texture():
    buffer_with_alpha, extents = load_font_texture_with_alpha()
    width = buffer_with_alpha.shape[0]
    height = buffer_with_alpha.shape[1]

    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGBA8,
        width,
        height,
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        buffer_with_alpha.tostring(),
    )
    gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    # gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )

    return (texture, (width, height), extents)
