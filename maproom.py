#!/usr/bin/env python

# import code; code.interact( local = locals() )

"""
The main Maproom application executable.
"""
import os
import sys
import logging
import wx

# load this as early as possible so that we can ideally report even startup errors
import library.exception_handler

import ui as app_ui
import Version as version
from ui.Error_notifier import notify_all_errors
import app_globals

import pyproj

# these are here to get py2exe / py2app to pick them up, since the deps
# import them in a way that they miss
import multiprocessing
multiprocessing.freeze_support()


class Logger(object):

    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.log.flush()


def main():
    app_globals.main_logger = logging.getLogger("")
    app_globals.main_logger.setLevel(logging.DEBUG)
    # PyOpenGL is a little too chatty with its logging.
    app_globals.opengl_logger = logging.getLogger("OpenGL")
    app_globals.opengl_logger.setLevel(logging.WARNING)
    app_globals.version = version

    app = app_ui.Application()

    # If the app is frozen, don't write anything to stdout.
    if hasattr(sys, "frozen"):
        log_file = os.path.join(wx.StandardPaths.Get().GetUserDataDir(), "Maproom", "log.txt")
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))

        # Make sure we don't accumulate really long log files
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except IOError, e:
                # this probably means the app has another instance running
                pass
            except WindowsError:
                # on Windows, failure to access generates a WindowsError rather than an IOError
                pass

        sys.stdout = Logger(filename=log_file)
        sys.stderr = Logger(filename=log_file)
    # Otherwise, log to stdout.
    else:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(levelname)s: %(message)s at %(filename)s:%(lineno)d",
        )
        console.setFormatter(formatter)
        app_globals.main_logger.addHandler(console)

    """
    root_layer = plugin.Composite_layer(
        command_stack,
        plugin_loader,
        parent = None,
        name = "root",
    )
    """

    app.MainLoop()

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()
