# coding=utf8
import os
import wx
import sys


class Help_window( wx.Frame ):
    DEFAULT_SIZE = ( 640, 480 )
    IMAGE_PATH = "ui/images"

    def __init__( self, parent_frame ):
        wx.Frame.__init__(
            self, None, wx.ID_ANY, "Maproom Help", size = self.DEFAULT_SIZE,
        )
        self.SetIcon( parent_frame.GetIcon() )

        image_path = self.IMAGE_PATH
        if os.path.basename( os.getcwd() ) != "maproom":
            image_path = os.path.join( "maproom", image_path )

        if sys.platform == "darwin":
            multiselect_key = u"⌘"
        else:
            multiselect_key = u"Ctrl"

        self.text_area = wx.richtext.RichTextCtrl(
            self, wx.ID_ANY, style = wx.richtext.RE_READONLY | wx.VSCROLL,
        )
        self.text_area.GetCaret().Hide()
        self.sizer = wx.BoxSizer( wx.VERTICAL )
        self.sizer.Add( self.text_area, 1, wx.EXPAND )
        self.SetSizer( self.sizer )

        # Necessary for unicode characters to show up.
        self.text_area.SetFont(
            wx.Font( 10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL,
                     wx.FONTWEIGHT_NORMAL ),
        )

        self.text_area.BeginFontSize( 14 )
        self.text_area.BeginUnderline()
        self.text_area.WriteText( "Overview" )
        self.text_area.EndFontSize()
        self.text_area.EndUnderline()

        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Menu bar: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "At the top of the application window (on Windows) or at the top of the screen (on Mac OS X), there is a menu bar containing most operations." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Tool bar: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "A series of buttons near the top of the window allows quick access to common operations." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Map area: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "The main portion of the window displays your currently opened maps, one on top of another in layers." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Layer tree: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "The upper left of the window contains a hierarchical tree of your opened layers by name with the currently selected layer indicated." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Properties panel: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "The lower left of the window contains a list of properties for the currently selected layer, line, or point." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Status bar: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "At the bottom of the window, there is a status bar containing the current mouse position in lat-long coordinates, along with any current status messages." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginFontSize( 14 )
        self.text_area.BeginUnderline()
        self.text_area.WriteText( "Loading and Saving" )
        self.text_area.EndFontSize()
        self.text_area.EndUnderline()

        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Open a KAP nautical chart file: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "open.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select File → Open... from the menu. Select a KAP file to open. " )
        self.text_area.WriteText( "The KAP will load in the map area and show up in the layer tree as well." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Open a GeoTIFF raster file: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "open.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select File → Open... from the menu. Select a GeoTiff file to open. " )
        self.text_area.WriteText( "The GeoTiff will load in the map area and show up in the layer tree as well." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Open a Verdat bathymetry file: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "open.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select File → Open... from the menu. Select a Verdat file to open. " )
        self.text_area.WriteText( "The Verdat will load in the map area and show up in the layer tree as well." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Open a Maproom bathymetry file: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "open.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select File → Open... from the menu. Select a Maproom file to open. " )
        self.text_area.WriteText( "The data will load in the map area and show up in the layer tree as well." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Open an NGA DNC file: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "open.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select File → Open... from the menu. Select a zipped NGA DNC file to open. " )
        self.text_area.WriteText( "The DNC's bathymetry and shoreline data will load in the map area and show up in the layer tree as well. " )
        self.text_area.WriteText( "Note: This file format is only supported when OGDI support is enabled in the GDAL/OGR library." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Save a Verdat bathymetry file: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Select the Verdat layer you'd like to save in the layer tree on the left. " )
        self.text_area.WriteText( u"Select File → Save... from the menu. Select or enter a filename. Select Verdat as the File Type/Format. " )
        self.text_area.WriteText( "The layer will be saved to the file." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Save a Maproom bathymetry file: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Select the Verdat layer you'd like to save in the layer tree on the left. " )
        self.text_area.WriteText( u"Select File → Save... from the menu. Select or enter a filename. Select Maproom as the File Type/Format. " )
        self.text_area.WriteText( "The layer will be saved to the file. " )
        self.text_area.WriteText( "Note: This file format is useful for saving your work as you go. It supports saving bathymetry data (such as unclosed boundaries) that would be invalid in a Verdat file." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginFontSize( 14 )
        self.text_area.BeginUnderline()
        self.text_area.WriteText( "Navigation" )
        self.text_area.EndFontSize()
        self.text_area.EndUnderline()

        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Zoom in or out: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "zoom_in.png" ) ),
        )
        self.text_area.WriteText( " or " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "zoom_out.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select View → Zoom In or View → Zoom Out from the menu. " )
        self.text_area.WriteText( "Alternately, use your mouse's scroll wheel to zoom in or out." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Zoom to box: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "zoom_box.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select View → Zoom to Box. " )
        self.text_area.WriteText( "Click and drag on an area of the map to select a rectangular region. " )
        self.text_area.WriteText( "A box will indicate the selected area as you drag the mouse. Release the mouse to complete the selection. " )
        self.text_area.WriteText( "The map will move and zoom in to encompass the region selected." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Zoom to fit: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "zoom_fit.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select View → Zoom to Fit. " )
        self.text_area.WriteText( "The map will move and zoom in or out as necessary so that all of the loaded, non-hidden layers are within view." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Pan the map: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Switch to Pan mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "pan.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar, selecting Layer → Pan from the menu, or simply holding down the Alt key. " )
        self.text_area.WriteText( "Click and drag on an area of the map. " )
        self.text_area.WriteText( "The map will move in the direction you drag your mouse. Release the mouse button to stop panning. " )
        self.text_area.WriteText( u"Releasing the Alt key, if pressed, will return to the previous mode." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Jump to Coordinates: " )
        self.text_area.EndBold()
        self.text_area.WriteText( u"Select View → Jump to Coordinates... from the menu, and enter lat-long coordinates as instructed. " )
        self.text_area.WriteText( "The map will jump so that the entered lat-long coordinates are centered." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginFontSize( 14 )
        self.text_area.BeginUnderline()
        self.text_area.WriteText( "Editing a Verdat Layer" )
        self.text_area.EndFontSize()
        self.text_area.EndUnderline()

        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Move a point: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the point you want to move. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click and drag a point on the map. The point will move in the direction you drag your mouse. Release the mouse to stop moving the point. " )
        self.text_area.WriteText( "If the point is part of a line, then the line segments connected to the point will stay connected to it." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Add a point: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer you want to add a point to. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click on an area of the map without any points or lines. A new point will appear there and become selected. " )
        self.text_area.WriteText( "You can change the depth of the selected point by typing a new depth and pressing enter. " )
        self.text_area.WriteText( "You can continue to add points in this manner as long as you remain in Edit Points mode. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Add a line: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer you want to add a line to. " )
        self.text_area.WriteText( "If one is not already selected, select a point on the map by clicking on it. This is where the line will start. " )
        self.text_area.WriteText( "Switch to Edit Lines mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_lines.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Lines from the menu. " )
        self.text_area.WriteText( "Click on an area of the map without any points or lines. A new point will appear there and become selected, and a line segment will connect the previously selected point to the new point. " )
        self.text_area.WriteText( "Alternately, click on an existing point in the same layer. A new line segment will connect the two existing points. " )
        self.text_area.WriteText( "You can change the depth of the selected point by typing a new depth and pressing enter. " )
        self.text_area.WriteText( "You can continue to add lines in this manner as long as you remain in Edit Lines mode. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Split a line: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the line you want to split. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click on the line at the location where you'd like the split to occur. " )
        self.text_area.WriteText( "A new point will appear there and become selected, and the clicked line segment will be split into two line segments at that point. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete a point: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the point you want to delete. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Select a point by clicking on it. Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Edit → Delete Selection from the menu. The point will disappear. " )
        self.text_area.WriteText( "If the point was part of a line, then the line will be kept contiguous by connecting the remaining points on either side of the deleted point." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete several points: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the points you want to delete. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Select a point by clicking on it. " )
        self.text_area.WriteText( u"Select each additional point by holding down the %s key and clicking on a point. " % multiselect_key )
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Edit → Delete Selection from the menu. The selected points will disappear. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete a line: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the line you want to delete. " )
        self.text_area.WriteText( "Switch to Edit Lines mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_lines.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Lines from the menu. " )
        self.text_area.WriteText( "Select a line by clicking on it. Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Edit → Delete Selection from the menu. The line will disappear." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete several lines: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the lines you want to delete. " )
        self.text_area.WriteText( "Switch to Edit Lines mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_lines.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Lines from the menu. " )
        self.text_area.WriteText( "Select a line by clicking on it. " )
        self.text_area.WriteText( u"Select each additional line by holding down the %s key and clicking on a line. " % multiselect_key )
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Edit → Delete Selection from the menu. The selected lines will disappear." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete a span of points on a line (and that section of the line as well): " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the points you want to delete. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Select a start point on a line by clicking on it. " )
        self.text_area.WriteText( "Select an end point by holding down the Shift key and clicking on another point on the same line. " )
        self.text_area.WriteText( "All of the points between the start and end points on the line will become selected as well. " )
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Edit → Delete Selection from the menu. " )
        self.text_area.WriteText( "The selected points (and the lines connecting them) will disappear. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginFontSize( 14 )
        self.text_area.BeginUnderline()
        self.text_area.WriteText( "Editing a BNA Layer" )
        self.text_area.EndFontSize()
        self.text_area.EndUnderline()

        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Move a polygon point: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the point you want to move. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click on the polygon containing the point you want to move. Its points will become shown. " )
        self.text_area.WriteText( "Click and drag a point on the polygon's boundary. The point will move in the direction you drag your mouse. Release the mouse to stop moving the point. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Add a point to a polygon: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the polygon you want to add a point to. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click on the polygon you want to add a point to. Its points will become shown. " )
        self.text_area.WriteText( "Click on an existing point on the polygon's boundary near the spot you want to add a new point. " )
        self.text_area.WriteText( "The boundary of the polygon will open up near the point you have selected, indicating the area of the polygon boundary that any new points will be added to. " )
        self.text_area.WriteText( "Click on the location of the map where you want the new point added. A new point will appear there and become selected. " )
        self.text_area.WriteText( "You can continue to add points in this manner as long as you remain in Edit Points mode and a polygon point remains selected. " )
        self.text_area.WriteText( "When you are done adding points, you can clear the selection to close up the polygon." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Add a new polygon: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer that you want to add a polygon to. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click on the location of the map where you want the new polygon started. A new point will appear there and become selected. " )
        self.text_area.WriteText( "You can continue to add points in this manner as long as you remain in Edit Points mode and a polygon point remains selected. " )
        self.text_area.WriteText( "When you are done adding points, you can clear the selection to close up the polygon." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete a point from a polygon: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the point you want to delete. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click on the polygon you want to delete a point from. Its points will become shown. " )
        self.text_area.WriteText( "Select a point by clicking on it. Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Edit → Delete Selection from the menu. The point will disappear. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete several points from a polygon: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the vector layer containing the points you want to delete. " )
        self.text_area.WriteText( "Switch to Edit Points mode by clicking the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_points.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or selecting Layer → Edit Points from the menu. " )
        self.text_area.WriteText( "Click on the polygon you want to delete points from. Its points will become shown. " )
        self.text_area.WriteText( "Select a point by clicking on it. " )
        self.text_area.WriteText( u"Select each additional point by holding down the %s key and clicking on a point. " % multiselect_key )
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Edit → Delete Selection from the menu. The selected points will disappear. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginFontSize( 14 )
        self.text_area.BeginUnderline()
        self.text_area.WriteText( "Layer Manipulation" )
        self.text_area.EndFontSize()
        self.text_area.EndUnderline()

        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Create a new Verdat layer: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_layer.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Layer → Add Layer from the menu. Then select the Verdat option. " )
        self.text_area.WriteText( "An empty Verdat layer will be created in the map area and show up in the layer tree as well." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Create a new BNA layer: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "add_layer.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Layer → Add Layer from the menu. Then select the BNA option. " )
        self.text_area.WriteText( "An empty BNA layer will be created in the map area and show up in the layer tree as well." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Delete a layer: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the layer to delete. " )
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "delete_layer.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Layer → Delete Layer from the menu. " )
        self.text_area.WriteText( "The selected layer will disappear, including all of its points and lines." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Raise or lower a layer: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "If it's not already selected in the layer tree on the left, select the layer to raise or lower. " )
        self.text_area.WriteText( "Click the " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "raise.png" ) ),
        )
        self.text_area.WriteText( u" or " )
        self.text_area.WriteImage(
            wx.Image( os.path.join( image_path, "lower.png" ) ),
        )
        self.text_area.WriteText( u" button on the toolbar or select Layer → Raise Layer or Layer → Lower Layer from the menu. " )
        self.text_area.WriteText( "The layer will swap places with the layer above or below it in the layer tree, and the ordering of layers within the map area will update accordingly." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Show or hide a layer: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "In the layer tree on the left, click the checkbox next to a layer name to toggle whether the layer is shown or hidden. " )
        self.text_area.WriteText( "When hidden, a layer will not appear in the map area. " )
        self.text_area.WriteText( "Note: All the depth labels for a particular Verdat are grouped together as a child layer of the Verdat. " )
        self.text_area.WriteText( "So, to show the depth labels for a Verdat, check the checkbox next to the depth labels layer under the Verdat." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Merge Verdat bathymetry layers together: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Open or create two or more Verdat layers to merge together. " )
        self.text_area.WriteText( u"Select Layer → Merge Layers... from the menu. " )
        self.text_area.WriteText( "A dialog box will open with a list of layers. Select the layers that you'd like to merge and click the OK button. " )
        self.text_area.WriteText( "A single new layer will be created in the map area and show up in the layer tree, replacing the selected layers. " )
        self.text_area.WriteText( "This new layer will contain all of the points and lines of the selected layers." )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.BeginBold()
        self.text_area.WriteText( "Merge duplicate points together: " )
        self.text_area.EndBold()
        self.text_area.WriteText( "Open or create a Verdat layer and select it. Or, open two or more Verdat layers and merge them into one layer as described above. " )
        self.text_area.WriteText( u"Select Tools → Merge Duplicate Points... from the menu. " )
        self.text_area.WriteText( "A dialog box will open with two tolerance sliders. " )
        self.text_area.WriteText( "Drag the sliders to select the distance and point depth tolerance to use when searching for duplicate points. " )
        self.text_area.WriteText( "Click the Find Duplicates button. " )
        self.text_area.WriteText( "A list of possible duplicate points will be displayed and highlighted on the map. " )
        self.text_area.WriteText( "You can adjust the sliders and click the Find Duplicates button again, or you can remove pairs of points from the list. " )
        self.text_area.WriteText( "When you are satisfied with the list of point pairs to merge, click the Merge button to merge each pair into a single point. " )

        self.text_area.Newline()
        self.text_area.Newline()

        self.text_area.WriteText( "Note that point pairs indicated in red cannot be merged automatically. " )
        self.text_area.WriteText( "Also note that if the selected layer was created by merging two or more Verdat layers together, " )
        self.text_area.WriteText( "then potential duplicate points originally from the same source layer are disregarded and not displayed in the list of point pairs." )

        self.Show()
