import wx
import os.path
import Layer
from Layer_manager import *
import app_globals

"""
Prompts the user for a filename and then tries to load it as a layer,
displaying any resulting error message to the user.
"""

def show():
    file_types = (
        "All Files|*.bna;*.tif;*.png;*.kap;*.maproomv;*.ms1;*.zip;*.shp;*.verdat;*.xml"
        "BNA files (*.bna)|*.bna",
        "GeoTIFF files (*.tif)|*.tif",
        "GeoTIFF files (*.png)|*.png",
        "KAP files (*.kap)|*.kap",
        "Maproom Vector files (*.maproomv)|*.maproomv",
        "MOSS files (*.ms1)|*.ms1",
        "NGA DNC ZIP files (*.zip)|*.zip",
        "Shape files (*.shp)|*.shp",
        "Verdat files (*.verdat)|*.verdat",
        "XML files (*.xml)|*.xml"
    )
    
    dialog = wx.FileDialog(
        app_globals.application.frame,
        style = wx.FD_FILE_MUST_EXIST,
        message = "Select a file to open",
        wildcard = "|".join( file_types )
    )
        
    if dialog.ShowModal() != wx.ID_OK:
        return
        
    file_path = os.path.normcase( os.path.abspath( dialog.GetPath() ) )

    open_file(file_path)

def open_file(file_path):

    # xml files are treated specially
    if ( file_path.endswith( ".xml" ) ):
        layer = Layer.Layer()
        layer.read_from_file( file_path )
        # we don't need to insert anything because the xml reader inserts layers as appropriate
        
        return
    
    insertion_multi_index = app_globals.layer_manager.get_layer_multi_index_from_file_path( file_path )
    if ( insertion_multi_index != None ):
        layer = app_globals.layer_manager.get_layer_by_multi_index( insertion_multi_index )
        dialog = wx.MessageDialog(
            app_globals.application.frame,
            message = 'That file is already loaded as layer "%s". Load the file again, replacing the existing layer?' % layer.name,
            caption = "File already loaded",
            style = wx.OK | wx.CANCEL | wx.ICON_QUESTION,
        )
        
        if ( dialog.ShowModal() != wx.ID_OK ):
            return
        
        app_globals.layer_manager.delete_selected_layer( layer )
    
    layer = Layer.Layer()
    layer.read_from_file( file_path )
    layer.name = os.path.split( file_path )[ 1 ]
    
    if ( layer.load_error_string != "" ):
        wx.MessageDialog(
            app_globals.application.frame,
            message = str( layer.load_error_string ),
            style = wx.OK | wx.ICON_ERROR,
        ).ShowModal()
        
        return None
    
    app_globals.layer_manager.insert_layer( insertion_multi_index, layer )

    return None
