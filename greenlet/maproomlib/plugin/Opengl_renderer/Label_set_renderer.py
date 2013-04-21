import os
import wx
import logging
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.ui as ui
import maproomlib.utility as utility


class Label_set_renderer:
    """
    A sub-renderer that the :class:`Opengl_renderer` delegates to for
    rendering text labels.
    """
    FONT_PATH = "maproomlib/ui/images/font.png"
    RENDER_LENGTH_TOLERANCE = 0.001
    RENDER_THRESHOLD_FACTOR = 17000
    EXTRA_CAPACITY_FACTOR = 2.0   # extra capacity to reserve
    DEFAULT_MAX_CHAR_COUNT = 25    # per label
    VERTEX_DTYPE = np.dtype(
        [ ( "x", np.float32 ), ( "y", np.float32 ) ],
    )
    EXTENT_DTYPE = np.dtype(
        [ ( "offset_x", np.float32 ), ( "offset_y", np.float32 ),
          ( "width", np.float32 ), ( "height", np.float32 ) ],
    )
    LABEL_INFO_DTYPE = np.dtype(
        [ ( "anchor_x", np.float32 ), ( "anchor_y", np.float32 ),
          ( "pixel_width", np.float32 ), ( "pixel_height", np.float32 ) ],
    )

    def __init__( self, root_layer, layer, viewport, opengl_renderer,
                  transformer, picker,
                  max_char_count = DEFAULT_MAX_CHAR_COUNT ):
        self.root_layer = root_layer
        self.layer = layer
        self.viewport = viewport
        self.inbox = ui.Wx_inbox()
        self.outbox = utility.Outbox()
        self.opengl_renderer = opengl_renderer
        self.transformer = transformer
        self.picker = picker
        self.max_char_count = max_char_count
        ( self.font_texture, self.font_texture_size ) = self.load_font_texture()
        self.font_extents = ui.FONT_EXTENTS
        self.char_extents = None  # per-character metadata
        self.label_info = None    # per-label metadata
        self.label_count = 0
        self.label_capacity = 0
        self.vertex_buffer = None
        self.texcoord_buffer = None
        self.render_threshold = None
        self.hide_when_zoomed = False
        self.reference_render_length = None
        self.logger = logging.getLogger( __name__ )

        gl.glEnable( gl.GL_BLEND )
        gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )

    def load_font_texture( self ):
        font_path = self.FONT_PATH
        if os.path.basename( os.getcwd() ) == "maproom":
            font_path = os.path.join( "..", font_path )

        texture = gl.glGenTextures( 1 )
        gl.glBindTexture( gl.GL_TEXTURE_2D, texture )
        image = wx.Image( font_path, wx.BITMAP_TYPE_PNG )

        width = image.GetWidth()
        height = image.GetHeight()
        buffer = np.frombuffer( image.GetDataBuffer(), np.uint8 ).reshape(
            ( width, height, 3 ),
        )

        # Make an alpha channel that is opaque where the pixels are black
        # and semi-transparent where the pixels are white.
        buffer_with_alpha = np.empty( ( width, height, 4 ), np.uint8 )
        buffer_with_alpha[ :, :, 0:3 ] = buffer
        buffer_with_alpha[ :, :, 3 ] = (
            255 - buffer[ :, :, 0:3 ].sum( axis = 2 ) / 3
        ).clip( 230, 255 )

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
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_NEAREST )
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_NEAREST )
        gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
                            gl.GL_CLAMP_TO_EDGE )
        gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
                            gl.GL_CLAMP_TO_EDGE )

        gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )
        return ( texture, ( width, height ) )

    def run( self, scheduler ):
        if self.transformer:
            self.layer.outbox.subscribe(
                self.transformer( self.inbox ),
                request = (
                    "points_updated",
                    "depths_updated",
                    "points_deleted",
                    "points_undeleted",
                    "labels_added",
                    "labels_updated",
                    "size_changed",
                ),
            )

        self.fetch_labels()

        while True:
            message = self.inbox.receive(
                request = (
                    "update",
                    "labels",
                    "points_updated",
                    "depths_updated",
                    "points_deleted",
                    "points_undeleted",
                    "labels_added",
                    "labels_updated",
                    "projection_changed",
                    "size_changed",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "update" and message.get( "force" ):
                self.inbox.discard( request = "update" )
                self.position_labels()
                self.opengl_renderer.Refresh( False )
            elif request == "labels":
                unique_id = "render labels %s" % self.layer.name
                self.outbox.send(
                    request = "start_progress",
                    id = unique_id,
                    message = "Initializing labels"
                )
                try:
                    self.init_labels( scheduler, **message )
                    self.opengl_renderer.Refresh( False )
                finally:
                    self.outbox.send(
                        request = "end_progress",
                        id = unique_id,
                    )
            elif request in ( "points_updated", "points_undeleted" ):
                self.points_updated( **message )
            elif request == "depths_updated":
                # Note: This assumes that all labels need to be updated.
                self.fetch_labels()
            elif request == "points_deleted":
                self.points_deleted( **message )
            elif request == "labels_added":
                self.labels_added( **message )
            elif request == "labels_updated":
                self.labels_updated( **message )
            elif request == "projection_changed":
                if not self.transformer:
                    continue

                self.layer.inbox.send(
                    request = "get_points",
                    origin = self.viewport.geo_origin( self.layer.projection ),
                    size = self.viewport.geo_size( self.layer.projection ),
                    response_box = self.transformer( self.inbox ),
                )

                message = self.inbox.receive( request = "points" )
                points = message.get( "points" )

                self.points_updated(
                    layer = self.layer,
                    points = points,
                    projection = message.get( "projection" ),
                    indices = None,
                )
            elif request == "size_changed":
                size = message.get( "size" )

                if size is not None and self.hide_when_zoomed:
                    self.render_threshold = max( size ) * \
                        self.RENDER_THRESHOLD_FACTOR
            elif request == "close":
                self.delete()
                return

    def fetch_labels( self ):
        self.layer.inbox.send(
            request = "get_labels",
            origin = self.viewport.geo_origin( self.layer.projection ),
            size = self.viewport.geo_size( self.layer.projection ),
            response_box = self.inbox,
        )

    def init_labels( self, scheduler, labels, label_count, projection,
                     origin, size, hide_when_zoomed = False ):
        # Make the threshold at which this layer is rendered based on its
        # geographic size, but record that threshold in arbitrary reference
        # render units.
        self.hide_when_zoomed = hide_when_zoomed

        if size is not None and self.hide_when_zoomed:
            self.render_threshold = max( size ) * \
                self.RENDER_THRESHOLD_FACTOR

        MIN_TOTAL_LABEL_CAPACITY = 100

        self.label_count = label_count
        self.label_capacity = max( len( labels ), MIN_TOTAL_LABEL_CAPACITY )
        total_char_count = self.label_capacity * self.max_char_count

        vertex_data = np.zeros(
            ( int( 4 * total_char_count ), ),
            dtype = self.VERTEX_DTYPE,
        ).view( np.recarray )

        texcoord_data = np.zeros(
            ( len( vertex_data ), ),
            dtype = self.VERTEX_DTYPE,
        ).view( np.recarray )

        self.char_extents = np.zeros(
            ( int( total_char_count ), ),
            dtype = self.EXTENT_DTYPE,
        ).view( np.recarray )

        self.label_info = np.zeros(
            ( int( self.label_capacity ), ),
            dtype = self.LABEL_INFO_DTYPE,
        ).view( np.recarray )

	# Only recreate the VBOs if absolutely necessary to prevent a rendering
	# issue on Mac OS X.
        if self.vertex_buffer and \
           len( vertex_data ) <= len( self.vertex_buffer ):
            self.vertex_buffer[ : len( vertex_data ) ] = vertex_data
        else:
            self.vertex_buffer = gl_vbo.VBO( vertex_data )

        if self.texcoord_buffer and \
           len( texcoord_data ) <= len( self.texcoord_buffer ):
            self.texcoord_buffer[ : len( texcoord_data ) ] = texcoord_data
        else:
            self.texcoord_buffer = gl_vbo.VBO( texcoord_data )

        for ( label_index, ( text, anchor_position ) ) in \
            enumerate( labels ):

            if label_index >= label_count:
                break
            if text is None:
                continue

            self.add_label(
                text,
                anchor_position,
                projection,
                label_index,
                # self.char_extents is initialized to zeros above
                zero_extents = False,
            )

            if scheduler and label_index % 100 == 0:
                scheduler.switch()

        self.position_labels( set_texcoords = True )

    def add_label( self, text, anchor_position, projection, label_index,
                   zero_extents = True ):
        if len( text ) > self.max_char_count:
            text = text[ : self.max_char_count - 2 ] + ".."

        if self.transformer:
            anchor_position = self.transformer.transform(
                anchor_position,
                projection,
            )

        self.label_info.anchor_x[ label_index ] = anchor_position[ 0 ]
        self.label_info.anchor_y[ label_index ] = anchor_position[ 1 ]

        pixel_size = [ 0, 0 ]

        # Determine the pixel width and height of the entire label.
        index = label_index
        for character in text:
            if character not in self.font_extents:
                character = "?"

            self.char_extents[ index ] = \
                tuple( self.font_extents[ character ] )

            pixel_size[ 0 ] += self.char_extents.width[ index ]
            pixel_size[ 1 ] = max(
                pixel_size[ 1 ],
                self.char_extents.height[ index ],
            )

            # Characters within a label are interleaved. So all the first
            # characters from all the labels come first in the array, and
            # then all the second characters, and so on.
            index += self.label_capacity

        # Set zeros for the rest of the available character slots.
        if zero_extents is True:
            self.char_extents[
                index : len( self.char_extents ) : self.label_capacity
            ].view( "4f4" ).fill( 0 )

        pixel_size = tuple( pixel_size )
        self.label_info.pixel_width[ label_index ] = pixel_size[ 0 ]
        self.label_info.pixel_height[ label_index ] = pixel_size[ 1 ]

    def position_labels( self, label_start_index = 0, label_count = None,
                         set_texcoords = False ):
        if self.label_info is None:
            return

        if label_count is None:
            label_count = self.label_count - label_start_index

        if self.transformer:
            ( label_render_widths, _ ) = self.viewport.pixel_sizes_to_render_sizes(
                self.label_info.pixel_width,
            )

            ( char_render_widths, char_render_heights ) = \
                self.viewport.pixel_sizes_to_render_sizes(
                    self.char_extents.width, self.char_extents.height,
                )
        else:
            label_render_widths = self.label_info.pixel_width
            char_render_widths = self.char_extents.width
            char_render_heights = self.char_extents.height

        vertex_data = self.vertex_buffer.data
        texcoord_data = self.texcoord_buffer.data
        ( texture_width, texture_height ) = self.font_texture_size

        # Calculate the positions of all the 0th characters in each label,
        # then all of the 1st characters in each label, then all of the 2nd
        # characters, and so on.
        for char_index in range( 0, self.max_char_count ):
            index = label_start_index + char_index * self.label_capacity
            end_index = index + label_count
            char_slice = slice( index, end_index )
            label_slice = slice( label_start_index, label_start_index + label_count )
            buffer_index = index * 4
            buffer_end_index = end_index * 4

            if char_index == 0:
                # Center each label horizontally underneath its anchor. This
                # calculates the position of the 0th character's left side for
                # each label.
                left_xs = self.label_info.anchor_x[ label_slice ] - \
                          label_render_widths[ label_slice ] // 2
            else:
                # This bases the position of the nth character's left side for
                # each label on the right side of the previous character.
                left_xs = prev_right_xs

            top_ys = self.label_info.anchor_y[ label_slice ]
            right_xs = left_xs + char_render_widths[ char_slice ]

            if self.transformer:
                bottom_ys = top_ys - char_render_heights[ char_slice ]
            else:
                bottom_ys = top_ys + char_render_heights[ char_slice ]

            if set_texcoords:
                left_texcoords = \
                    self.char_extents.offset_x[ char_slice ] \
                    / texture_width
                bottom_texcoords = \
                    ( self.char_extents.offset_y[ char_slice ] + \
                      self.char_extents.height[ char_slice ] ) \
                    / texture_height
                right_texcoords = \
                    ( self.char_extents.offset_x[ char_slice ] + \
                      self.char_extents.width[ char_slice ] ) \
                    / texture_width
                top_texcoords = \
                    self.char_extents.offset_y[ char_slice ] \
                    / texture_height

            # Bottom left corners.
            buffer_slice = slice( buffer_index, buffer_end_index, 4 )
            vertex_data.x[ buffer_slice ] = left_xs
            vertex_data.y[ buffer_slice ] = bottom_ys
            ( texture_width, texture_height ) = self.font_texture_size

            if set_texcoords:
                texcoord_data.x[ buffer_slice ] = left_texcoords
                texcoord_data.y[ buffer_slice ] = bottom_texcoords

            # Upper left corners.
            buffer_slice = slice( buffer_index + 1, buffer_end_index + 1, 4 )
            vertex_data.x[ buffer_slice ] = left_xs
            vertex_data.y[ buffer_slice ] = top_ys

            if set_texcoords:
                texcoord_data.x[ buffer_slice ] = left_texcoords
                texcoord_data.y[ buffer_slice ] = top_texcoords

            # Upper right corners.
            buffer_slice = slice( buffer_index + 2, buffer_end_index + 2, 4 )
            vertex_data.x[ buffer_slice ] = right_xs
            vertex_data.y[ buffer_slice ] = top_ys

            if set_texcoords:
                texcoord_data.x[ buffer_slice ] = right_texcoords
                texcoord_data.y[ buffer_slice ] = top_texcoords

            # Bottom right corners.
            buffer_slice = slice( buffer_index + 3, buffer_end_index + 3, 4 )
            vertex_data.x[ buffer_slice ] = right_xs
            vertex_data.y[ buffer_slice ] = bottom_ys

            if set_texcoords:
                texcoord_data.x[ buffer_slice ] = right_texcoords
                texcoord_data.y[ buffer_slice ] = bottom_texcoords

            prev_right_xs = right_xs

        buffer_slice = slice( 0, self.label_count * self.max_char_count * 4 )
        self.vertex_buffer[ buffer_slice ] = \
            self.vertex_buffer.data[ buffer_slice ]

        if set_texcoords:
            self.texcoord_buffer[ buffer_slice ] = \
                self.texcoord_buffer.data[ buffer_slice ]

        self.reference_render_length = self.viewport.reference_render_length()

    def points_updated( self, layer, points, projection, indices = None,
                        undo_recorded = None ):
        if self.label_info is None or len( points ) == 0:
            return

        if indices is None:
            self.label_info.anchor_x[ : ] = points.x
            self.label_info.anchor_y[ : ] = points.y
        else:            
            for index in indices:
                self.label_info.anchor_x[ index ] = points.x[ index ]
                self.label_info.anchor_y[ index ] = points.y[ index ]

        self.position_labels()

    def points_deleted( self, layer, points, projection, indices,
                        undo_recorded = None ):
        for index in indices:
            self.label_info.anchor_x[ index : index + 1 ] = np.nan
            self.label_info.anchor_y[ index : index + 1 ] = np.nan

        self.position_labels()

    def grow_arrays( self ):
        # Grow label info array.
        old_label_capacity = self.label_capacity
        self.label_capacity = \
            int( self.label_capacity * self.EXTRA_CAPACITY_FACTOR )
        added_label_capacity = self.label_capacity - old_label_capacity
        self.logger.debug(
            "Growing label info array from %d labels capacity to %d." % (
                old_label_capacity, self.label_capacity,
            ),
        )
        shape = list( self.label_info.shape )
        shape[ 0 ] = self.label_capacity
        self.label_info = np.resize(
            self.label_info, tuple( shape ),
        ).view( np.recarray )

        # Grow per-character extents array. This is a little tricky, because
        # this actually requires inserting extra space at several places in
        # the array due to the way all the 0th, 1st, 2nd, etc characters are
        # grouped together. To accomplish this, make a new, larger extents
        # array and then copy select regions from the old array.
        new_extent_capacity = \
            int( len( self.char_extents ) * self.EXTRA_CAPACITY_FACTOR )
        self.logger.debug(
            "Growing label extents array from %d characters capacity to %d." % (
                len( self.char_extents ), new_extent_capacity,
            ),
        )
        new_extents = np.zeros(
            ( self.label_capacity * self.max_char_count, ),
            dtype = self.EXTENT_DTYPE,
        ).view( np.recarray )

        for char_index in range( 0, self.max_char_count ):
            from_index = old_label_capacity * char_index
            from_slice = slice( from_index, from_index + old_label_capacity )
            to_index = self.label_capacity * char_index
            to_slice = slice( to_index, to_index + old_label_capacity )

            new_extents[ to_slice ] = self.char_extents[ from_slice ]

        self.char_extents = new_extents

        # Grow vertex and texcoord VBOs. This is tricky for the same reason
        # as described above, and relies on the same approach.
        new_buffer_capacity = \
            int( len( self.vertex_buffer ) * self.EXTRA_CAPACITY_FACTOR )
        self.logger.debug(
            "Growing label VBOs from %d points capacity to %d." % (
                len( self.vertex_buffer ), new_buffer_capacity,
            ),
        )
        new_vertex_data = np.zeros(
            ( self.label_capacity * self.max_char_count * 4, ),
            dtype = self.VERTEX_DTYPE,
        ).view( np.recarray )
        new_texcoord_data = np.zeros(
            ( self.label_capacity * self.max_char_count * 4, ),
            dtype = self.VERTEX_DTYPE,
        ).view( np.recarray )

        for char_index in range( 0, self.max_char_count ):
            from_index = old_label_capacity * char_index * 4
            from_slice = slice( from_index, from_index + old_label_capacity * 4 )
            to_index = self.label_capacity * char_index * 4
            to_slice = slice( to_index, to_index + old_label_capacity * 4 )

            new_vertex_data[ to_slice ] = self.vertex_buffer.data[ from_slice ]
            new_texcoord_data[ to_slice ] = self.texcoord_buffer.data[ from_slice ]

        self.vertex_buffer = gl_vbo.VBO( new_vertex_data )
        self.texcoord_buffer = gl_vbo.VBO( new_texcoord_data )

    def labels_added( self, labels, projection, start_index, count ):
        if start_index + count >= self.label_capacity:
            self.grow_arrays()

        for ( label_index, ( text, anchor_position ) ) in \
            enumerate( labels ):

            if text is None:
                continue

            self.add_label(
                text,
                anchor_position,
                projection,
                start_index + label_index,
                zero_extents = False,
            )

        self.label_count = max( self.label_count, start_index + count )

        self.position_labels(
            label_start_index = start_index, label_count = count,
            set_texcoords = True,
        )

        self.opengl_renderer.Refresh( False )

    def labels_updated( self, labels, projection ): 
        if not labels:
            return

        for ( text, anchor_position, label_index ) in labels:
            if text is None: continue

            self.add_label(
                text,
                anchor_position,
                projection,
                label_index,
            )

        self.position_labels( set_texcoords = True )

        self.opengl_renderer.Refresh( False )

    def render( self, pick_mode = False ):
        if pick_mode is True or \
           self.vertex_buffer is None or \
           self.reference_render_length is None:
            return

        reference_render_length = self.viewport.reference_render_length()
        if self.render_threshold is not None and \
           reference_render_length > self.render_threshold:
            return

        if abs( reference_render_length - self.reference_render_length ) > \
           self.RENDER_LENGTH_TOLERANCE:
            return

        gl.glEnable( gl.GL_TEXTURE_2D )
        gl.glBindTexture( gl.GL_TEXTURE_2D, self.font_texture )

        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        self.vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
        gl.glEnableClientState( gl.GL_TEXTURE_COORD_ARRAY )

        self.texcoord_buffer.bind()
        gl.glTexCoordPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated

        vertex_count = self.label_capacity * self.max_char_count * 4
        gl.glDrawArrays( gl.GL_QUADS, 0, vertex_count )

        self.vertex_buffer.unbind()
        self.texcoord_buffer.unbind()
        gl.glDisableClientState( gl.GL_TEXTURE_COORD_ARRAY )
        gl.glDisableClientState( gl.GL_VERTEX_ARRAY )

        gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )
        gl.glDisable( gl.GL_TEXTURE_2D )

    def delete( self ):
        """
        Remove all labels.
        """
        self.layer = None
        if self.font_texture:
            gl.glDeleteTextures(
                np.array( [ self.font_texture ], np.uint32 ),
            )
            self.font_texture = None

        self.vertex_buffer = None
        self.texcoord_buffer = None
