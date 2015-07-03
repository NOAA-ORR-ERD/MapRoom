driver="gl_immediate"

if driver == "gl_immediate":
    from .gl.color import color_to_int, int_to_color, int_to_color_uint8, int_to_html_color_string
    import gl.data_types as data_types
    from .gl.picker import NullPicker
    from .gl.textures import ImageData, ImageTextures, SubImageLoader
    from .gl_immediate.base_canvas import BaseCanvas
else:
    from .vispy.parser import *
    from .gl.color import color_to_int, int_to_color, int_to_color_uint8, int_to_html_color_string
    import gl.data_types as data_types
    from .vispy.base_canvas import BaseCanvas
