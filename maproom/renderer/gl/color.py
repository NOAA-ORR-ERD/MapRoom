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

def int_to_color_floats(color):
    c = np.uint32(color) # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    floats = tuple([i/255.0 for i in ints])
    return floats

def int_to_color_uint8(color):
    c = np.uint32(color) # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    return tuple(ints)

def int_to_html_color_string(color):
    c = np.uint32(color) # handle plain python integer being passed in
    ints = np.frombuffer(c.tostring(), dtype=np.uint8)
    cstr = "#%02x%02x%02x" % (ints[0], ints[1], ints[2])
    return cstr

if __name__ == "__main__":
    rgba = (.5, .5, .5, 1)
    i = color_floats_to_int(*rgba)
    print "%x" % i
    print type(i)
    color = int_to_color_floats(i)
    rgba2 = list(color)
    print rgba2
