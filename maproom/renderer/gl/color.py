import numpy as np


def color_floats_to_int(red, green, blue, alpha):
    """
    Given individual channel values for a color from 0.0 to 1.0, return that
    color as a single integer RGBA value.

    :param red: red channel value
    :type red: float
    :param green: green channel value
    :type green: float
    :param blue: blue channel value
    :type blue: float
    :param alpha: alpha channel value
    :type alpha: float
    :return: combined RGBA color value
    :rtype: uint32
    """
    CHANNEL_MAX = 255.0

    return np.array(
        (
            int(red * CHANNEL_MAX),
            int(green * CHANNEL_MAX),
            int(blue * CHANNEL_MAX),
            int(alpha * CHANNEL_MAX),
        ),
        dtype=np.uint8,
    ).view(np.uint32)[0]

def color_ints_to_int(red, green, blue, alpha=255):
    """
    Given individual channel values for a color from 0 to 255, return that
    color as a single integer RGBA value.

    :param red: red channel value
    :type red: float
    :param green: green channel value
    :type green: float
    :param blue: blue channel value
    :type blue: float
    :param alpha: alpha channel value
    :type alpha: float
    :return: combined RGBA color value
    :rtype: uint32
    """
    CHANNEL_MAX = 255

    return np.array(
        (
            int(min(red, CHANNEL_MAX)),
            int(min(green, CHANNEL_MAX)),
            int(min(blue, CHANNEL_MAX)),
            int(min(alpha, CHANNEL_MAX)),
        ),
        dtype=np.uint8,
    ).view(np.uint32)[0]


def alpha_from_int(color):
    c = np.uint32(color)  # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    return ints[3] / 255.0


def int_to_color_floats(color):
    c = np.uint32(color)  # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    floats = tuple([i / 255.0 for i in ints])
    return floats


def int_to_color_uint8(color):
    c = np.uint32(color)  # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    return tuple(ints)


def int_to_color_ints(color):
    c = np.uint32(color)  # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    return tuple(int(c) for c in ints)


def int_to_wx_colour(color):
    import wx

    rgba = int_to_color_ints(color)
    return wx.Colour(*rgba)

def int_to_html_color_string(color):
    c = np.uint32(color)  # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    cstr = "#%02x%02x%02x" % (ints[0], ints[1], ints[2])
    return cstr


# colormap is an array of 4 elements: (x coord, r, g, b)

def color_interp(value, colormap, alpha):
    c0 = colormap[0]
    if value < c0[0]:
        return color_floats_to_int(c0[1] / 255., c0[2] / 255., c0[3] / 255., alpha)
    for c in colormap[1:]:
        if value >= c0[0] and value <= c[0]:
            perc = (value - c0[0]) / float(c[0] - c0[0])
            return color_floats_to_int((c0[1] + (c[1] - c0[1]) * perc) / 255.,
                                       (c0[2] + (c[2] - c0[2]) * perc) / 255.,
                                       (c0[3] + (c[3] - c0[3]) * perc) / 255.,
                                       alpha)
        c0 = c
    return color_floats_to_int(c[1] / 255., c[2] / 255., c[3] / 255., alpha)

def linear_contour(values, colormap, smooth=True, alpha=1.0):
    colors = np.zeros(len(values), dtype=np.uint32)

    for i, value in enumerate(values):
        colors[i] = color_interp(value, colormap, alpha)
        #colors[i] = color_floats_to_int(1.0, 1.0, 1.0, 1.0)
    return colors


if __name__ == "__main__":
    rgba = (.5, .5, .5, 1)
    i = color_floats_to_int(*rgba)
    print "%x" % i
    print type(i)
    color = int_to_color_floats(i)
    rgba2 = list(color)
    print rgba2
