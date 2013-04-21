#!/usr/bin/env python

"""
The main Maproom application executable.
"""
import os
import sys

# See http://www.py2exe.org/index.cgi/WhereAmI
if hasattr( sys, "frozen" ):
    if sys.frozen == "windows_exe":
        main_dir = os.path.dirname(
            unicode( sys.executable, sys.getfilesystemencoding() )
        )
        sys.path.append( main_dir )
        os.chdir( main_dir )
else:
    sys.path.extend( ( ".", ".." ) )


import logging
import threading
import pyproj
import multiprocessing
import maproomlib.utility as utility
import maproomlib.ui as ui

try:
    import maproom.ui as app_ui
    import maproom.Version as version
except ImportError:
    import ui as app_ui
    import Version as version


def main_thread( root_layer, command_stack, plugin_loader ):
    main_scheduler = utility.Scheduler()
    main_scheduler.add( root_layer.run )
    main_scheduler.add( command_stack.run )
    main_scheduler.add( plugin_loader.run )

    main_scheduler.switch()


def main( args ):
    logger = logging.getLogger( "maproomlib" )
    logger.setLevel( logging.DEBUG )
    root_logger = logging.getLogger( "" )
    root_logger.setLevel( logging.DEBUG )

    # PyOpenGL is a little too chatty with its logging.
    opengl_logger = logging.getLogger( "OpenGL" )
    opengl_logger.setLevel( logging.WARNING )

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
        root_logger.addHandler( console )

    ui.notify_all_errors( logger )

    # Importing the plugin module here (instead of at the top of the file) so
    # that logging is configured before PyOpenGL starts spewing things to the
    # logs.
    import maproomlib.plugin as plugin

    plugin_loader = utility.Plugin_loader( plugin )
    command_stack = utility.Command_stack()

    root_layer = plugin.Composite_layer(
        command_stack,
        plugin_loader,
        parent = None,
        name = "root",
    )

    layer_selection_layer = plugin.Layer_selection_layer(
        command_stack,
        plugin_loader,
        root_layer,
    )

    layer_selection_layer.outbox.subscribe(
        root_layer.inbox,
        request = "selection_updated",
    )

    root_layer.children.append( layer_selection_layer )

    ui_scheduler = utility.Scheduler()

    app = app_ui.Application(
        root_layer,
        command_stack,
        plugin_loader,
        ui_scheduler,
        version,
        args,
    )

    ui_scheduler.add( app.MainLoop )

    # Do the application's main work in a separate thread so as not to bog
    # down the UI's responsiveness.
    thread = threading.Thread(
        target = main_thread,
        args = ( root_layer, command_stack, plugin_loader ),
    )
    thread.setDaemon( True )
    thread.start()

    ui_scheduler.switch()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main( sys.argv[ 1: ] )
