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
    if (layer == None):
        wx.MessageDialog(
            app_globals.application.frame,
            message="You must select an item in the tree (either a single layer or a tree of layers) in the tree control before saving.",
            style=wx.OK | wx.ICON_ERROR,
        ).ShowModal()

        return

    if (layer.points != None and len(layer.points) > 0):
        filetype = "XML or Verdat"
    else:
        filetype = "XML"

    if (layer.type == "root"):
        m = "Select %s file to save the entire tree of layers"
    elif (layer.type == "folder"):
        m = "Select %s file to save the sub-tree of layers"
    elif not layer.empty():
        m = "Select %s file to save layer " + layer.name
    else:
        wx.MessageBox("An empty layer cannot be saved. Please add some data to the layer and try again.", "Cannot Save Empty Layer")
        return

    dialog = wx.FileDialog(
        app_globals.application.frame,
        style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        message=m % filetype,
        wildcard="|".join(file_types_including_verdat if layer.points != None else file_types)
    )

    if dialog.ShowModal() != wx.ID_OK:
        return

    app_globals.layer_manager.save_layer(layer, dialog.GetPath())
