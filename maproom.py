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

import pyproj

# these are here to get py2exe / py2app to pick them up, since the deps
# import them in a way that they miss
import multiprocessing

class Logger(object):
    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

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

        sys.stdout = Logger()
        sys.stderr = Logger()
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
