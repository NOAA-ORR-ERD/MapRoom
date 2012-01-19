import wx
import Layer
import app_globals

"""
Prompts the user for a file name and saves the selected layer or the entire layer tree.
"""
    
def show():
    file_types = (
        "Maproom XML files (*.xml)|*.xml",
    )
    file_types_including_verdat = (
        "Maproom XML files (*.xml)|*.xml",
        "Verdat files (*.verdat)|*.verdat"
    )
    
    layer = app_globals.application.layer_tree_control.get_selected_layer()
    if ( layer == None ):
        wx.MessageDialog(
            app_globals.application.frame,
            message = "You must select an item in the tree (either a single layer or a tree of layers) in the tree control before saving.",
            style = wx.OK | wx.ICON_ERROR,
        ).ShowModal()
        
        return
    
    if ( layer.type == "root" ):
        m = "The root node of the layer tree is selected. This will save the entire tree of layers as an xml file."
    elif (layer.type == "folder" ):
        m = "A folder in the layer tree is selected. This will save the entire sub-tree of layers as an xml file."
    else:
        m = "An individual layer in the layer tree is selected. This will save the selected layer, " + layer.name + ", as an xml file."
    
    if ( layer.points != None ):
        m += "\n\nBecause you have selected a layer with points, you also have the option of saving this layer as a .verdat file."
    
    dialog = wx.MessageDialog(
        app_globals.application.frame,
        caption = "Save",
        message = m,
        style = wx.OK | wx.CANCEL,
    )
    if ( dialog.ShowModal() != wx.ID_OK ):
        return
    
    dialog = wx.FileDialog(
        app_globals.application.frame,
        style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        message = "Select save file",
        wildcard = "|".join( file_types_including_verdat if layer.points != None else file_types )
    )
    
    if dialog.ShowModal() != wx.ID_OK:
        return
    
    app_globals.layer_manager.save_layer( layer, dialog.GetPath() )
