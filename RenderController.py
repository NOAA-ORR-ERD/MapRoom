"""
This class controls rendering.
"""

from wx.lib.pubsub import pub

class RenderController( object ):
    def __init__( self, layer_manager, render_window ):
        self.layer_manager = layer_manager
        self.render_window = render_window
        
        pub.subscribe( self.on_layer_points_lines_changed, ('layer', 'lines', 'changed') )
        pub.subscribe( self.on_layer_points_lines_changed, ('layer', 'points', 'changed') )
        pub.subscribe( self.on_layer_points_lines_changed, ('layer', 'points', 'deleted') )

        pub.subscribe( self.on_projection_changed, ('layer', 'projection', 'changed') )
        pub.subscribe( self.on_layer_loaded, ('layer', 'loaded') )
        pub.subscribe( self.on_layer_updated, ('layer', 'updated') )
        pub.subscribe( self.on_layer_triangulated, ('layer', 'triangulated') )
        
    def on_layer_updated( self, layer ):
        if layer in self.layer_manager.layers:
            self.render_window.update_renderers()
        
    def on_layer_loaded( self, layer ):
        self.zoom_to_layer( layer )
        
    def on_layer_points_lines_changed( self, layer ):
        if layer in self.layer_manager.layers:
            self.render_window.rebuild_points_and_lines_for_layer( layer )
        
    def on_projection_changed( self, layer, projection ):
        if layer in self.layer_manager.layers:
            self.render_window.reproject_all( projection )

    def on_layer_triangulated( self, layer ):
        if layer in self.layer_manager.layers:
            self.render_window.rebuild_triangles_for_layer( layer )

    def zoom_to_selected_layer( self ):
        sel_layer = self.layer_manager.get_selected_layer()
        print "Selected layer = %r" % sel_layer
        if sel_layer is not None:
            self.zoom_to_layer( sel_layer )
    
    def zoom_to_layer( self, layer ):
        self.render_window.zoom_to_world_rect( layer.bounds )
