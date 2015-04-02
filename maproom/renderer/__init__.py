driver="gl_immediate"

if driver == "gl_immediate":
    from .gl.color import color_to_int, int_to_color
    import gl.data_types as data_types
    from .gl.textures import ImageData, ImageTextures, SubImageLoader
    from .gl_immediate.base_canvas import BaseCanvas
else:
    from .vispy.parser import *
    from .vispy.color import color_to_int, int_to_color
    import gl.data_types as data_types
    from .vispy.base_canvas import BaseCanvas
