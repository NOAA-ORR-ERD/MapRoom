import re
import wx
import pyproj


class Coordinate_jumper:
    """
    Prompts the user for a geographical coordinate and then jumps the
    viewport to that location.
    """
    WHITESPACE_AND_COMMAS_PATTERN = re.compile( "[\s,]+" )
    MIN_LATITUDE = -90
    MAX_LATITUDE = 90
    MIN_LONGITUDE = -180
    MAX_LONGITUDE = 180

    def __init__( self, frame, viewport ):
        self.frame = frame
        self.viewport = viewport
        self.projection = pyproj.Proj( "+proj=latlong" )

    def run( self, scheduler ):
        dialog = wx.TextEntryDialog(
            self.frame,
            "Please enter decimal coordinates in the form: latitude longitude",
            "Jump to Coordinates...",
        )

        if dialog.ShowModal() != wx.ID_OK:
            return

        try:
            coordinates = self.validate_and_parse( dialog.GetValue() )
        except ValueError:
            wx.MessageDialog(
                self.frame,
                message = "The entered coordinates are invalid.",
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            return

        if self.viewport.uninitialized():
            wx.MessageDialog(
                self.frame,
                message = "Please load or add a layer first.",
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            return

        self.viewport.jump_geo_center( coordinates, self.projection )

    def validate_and_parse( self, value ):
        pieces = self.WHITESPACE_AND_COMMAS_PATTERN.split( value.strip() )

        if len( pieces ) != 2:
            raise ValueError()

        latitude = float( pieces[ 0 ] )
        longitude = float( pieces[ 1 ] )

        if longitude < self.MIN_LONGITUDE or longitude > self.MAX_LONGITUDE:
            raise ValueError()

        if latitude < self.MIN_LATITUDE or latitude > self.MAX_LATITUDE:
            raise ValueError()

        return ( longitude, latitude )
