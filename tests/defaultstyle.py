#!/usr/bin/env python
import wx

import  wx.lib.buttons  as  buttons

from mock import *

from maproom.ui.dialogs import StyleDialog
from maproom.ui.info_panels import LayerInfoPanel
from maproom.layers import LayerStyle


def hook(self, f):
    layer = self.layer_tree_control.get_selected_layer()
    print(layer.style)
    self.control.info.display_panel_for_layer(self, layer)

#MockProject.process_flags = hook

class MyFrame(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.DefaultSize)
        self.project = MockProject()
        self.project.control = self
        self.layer = self.project.layer_tree_control.get_selected_layer()
        self.layer.layer_info_panel = ["Line Style", "Line Width", "Line Color", "Start Marker", "End Marker", "Line Transparency", "Fill Style", "Fill Color", "Fill Transparency", "Text Color", "Font", "Font Size", "Text Transparency", "Marplot Icon"]
        self.info = LayerInfoPanel(self, self.project)
        self.info.display_panel_for_layer(self.project, self.layer)

class MyApp(wx.App):
    def OnInit(self):
#        frame = MyFrame(None, -1, 'test_column_autosize.py')
#        frame.Show(True)
#        frame.Centre()
        project = MockProject(add_tree_control=True)
        project.control = None
        frame = StyleDialog(project, [])
        import wx.lib.inspection
#        wx.lib.inspection.InspectionTool().Show()
        result = frame.ShowModal()
        print(result)
        print(frame.get_style())
        return True

if __name__ == '__main__':
    app = MyApp(0)
    app.MainLoop()
