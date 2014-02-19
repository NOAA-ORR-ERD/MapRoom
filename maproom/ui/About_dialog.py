import wx
import os
import app_globals


class About_dialog:
    LOGO_FILENAME = "ui/images/maproom_large.png"

    def show(self):
        logo_filename = self.LOGO_FILENAME

        info = wx.AboutDialogInfo()

        info.SetIcon(wx.Icon(logo_filename, wx.BITMAP_TYPE_PNG))
        info.SetName("Maproom")

        if not app_globals.version.SOURCE_CONTROL_REVISION:
            info.SetVersion(app_globals.version.VERSION)
        else:
            info.SetVersion("%s (r%s)" % (
                app_globals.version.VERSION, app_globals.version.SOURCE_CONTROL_REVISION,
            ))

        desc = "High-performance 2d mapping developed by NOAA\n\nUsing:\n"
        desc += "  wxPython %s\n" % wx.version()
        try:
            import gdal
            desc += "  GDAL %s\n" % gdal.VersionInfo()
        except:
            pass
        try:
            import numpy
            desc += "  numpy %s\n" % numpy.version.version
        except:
            pass
        try:
            import OpenGL
            desc += "  PyOpenGL %s\n" % OpenGL.__version__
        except:
            pass
        try:
            import pyproj
            desc += "  PyProj %s\n" % pyproj.__version__
        except:
            pass
        info.SetDescription(desc)
        info.SetWebSite("http://www.noaa.gov/")

        dialog = wx.AboutBox(info)
