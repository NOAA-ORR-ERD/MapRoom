#!/usr/bin/env python

# import code; code.interact( local = locals() )

"""
The main Maproom application executable.
"""
import os
import sys
import logging
import wx
import ui as app_ui
import Version as version
from ui.Error_notifier import notify_all_errors
import app_globals

def main( args ):
    app_globals.main_logger = logging.getLogger( "" )
    app_globals.main_logger.setLevel( logging.DEBUG )
    # PyOpenGL is a little too chatty with its logging.
    app_globals.opengl_logger = logging.getLogger( "OpenGL" )
    app_globals.opengl_logger.setLevel( logging.WARNING )
    app_globals.version = version
    
    # If the app is frozen, don't write anything to stdout.
    if hasattr( sys, "frozen" ):
        pyproj.set_datapath( "pyproj_data" )

        class Blackhole:
            softspace = 0
            def write( self, text ):
                pass

        sys.stdout = Blackhole()
        sys.stderr = Blackhole()
    # Otherwise, log to stdout.
    else:
        console = logging.StreamHandler()
        console.setLevel( logging.INFO )
        formatter = logging.Formatter(
            "%(levelname)s: %(message)s at %(filename)s:%(lineno)d",
        )
        console.setFormatter( formatter )
        app_globals.main_logger.addHandler( console )
    
    notify_all_errors( app_globals.main_logger )
    
    """
    root_layer = plugin.Composite_layer(
        command_stack,
        plugin_loader,
        parent = None,
        name = "root",
    )
    """
    
    app = app_ui.Application( args )
    app.MainLoop()

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main( sys.argv[ 1: ] )
