#!/usr/bin/env python


import sys
import traceback
import wx


class Triangle_dialog(wx.Dialog):
    SPACING = 15
    NAME = "Triangulate Layer"

    def __init__(self, project):
        self.project = project
        wx.Dialog.__init__(self, project.window.control, wx.ID_ANY, self.NAME,
                           style=wx.DEFAULT_DIALOG_STYLE, name=self.NAME
                           )
        self.SetIcon(project.window.control.GetIcon())

        self.outer_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.panel = wx.Panel(self, wx.ID_ANY)

        self.outer_sizer.Add(self.panel, 1, wx.EXPAND)

        # put the various dialog items into sizer, which will become the sizer of panel

        # quality mesh

        box = wx.StaticBox(self.panel, label="Quality Mesh Minimum Angle")
        s = wx.StaticBoxSizer(box, wx.VERTICAL)
        t = wx.StaticText(self.panel, label="You can specify a minimum triangle angle (or leave blank if you don't want to specify a minimum).", pos=(0, 0))
        s.Add(t, 0, wx.ALIGN_TOP | wx.ALL, 5)
        t = wx.StaticText(self.panel, label="If the minimum angle is 20.7 degrees or smaller, the triangulation is theoretically guaranteed to terminate.", pos=(0, 0))
        s.Add(t, 0, wx.ALIGN_TOP | wx.ALL, 5)
        t = wx.StaticText(self.panel, label="It often succeeds for minimum angles up to 33 degrees. It usually doesn't terminate for angles above 34 degrees.", pos=(0, 0))
        s.Add(t, 0, wx.ALIGN_TOP | wx.ALL, 5)

        s2 = wx.BoxSizer(wx.HORIZONTAL)
        t = wx.StaticText(self.panel, label="Minimum angle:")
        s2.Add(t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.angle_text_box = wx.TextCtrl(self.panel, 0, "", size=(100, -1))
        s2.Add(self.angle_text_box, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        t = wx.StaticText(self.panel, label="degrees")
        s2.Add(t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        s.Add(s2, 0, wx.ALIGN_TOP | wx.ALL, 5)

        self.sizer.Add(s, 0, wx.ALIGN_TOP | wx.ALL, 5)

        # maximum triangle area

        box = wx.StaticBox(self.panel, label="Maximum Triangle Area")
        s = wx.StaticBoxSizer(box, wx.VERTICAL)
        t = wx.StaticText(self.panel, label="You can specify a maximum triangle area (or leave blank if you don't want to specify a maximum).", pos=(0, 0))
        s.Add(t, 0, wx.ALIGN_TOP | wx.ALL, 5)
        t = wx.StaticText(self.panel, label="The units are those of the point coordinates on this layer.", pos=(0, 0))
        s.Add(t, 0, wx.ALIGN_TOP | wx.ALL, 5)

        s2 = wx.BoxSizer(wx.HORIZONTAL)
        t = wx.StaticText(self.panel, label="Maximum area:")
        s2.Add(t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.area_text_box = wx.TextCtrl(self.panel, 0, "", size=(100, -1))
        s2.Add(self.area_text_box, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        t = wx.StaticText(self.panel, label="units")
        s2.Add(t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        s.Add(s2, 0, wx.ALIGN_TOP | wx.ALL, 5)

        self.sizer.Add(s, 0, wx.ALIGN_TOP | wx.ALL, 5)

        # the buttons

        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.triangulate_button = wx.Button(self.panel, label="Triangulate")
        self.triangulate_button.SetDefault()

        self.close_button = wx.Button(self.panel, label="Close")

        # Dialog button ordering, by convention, is different between Mac and Windows.
        # fixme -- is there a standard way to take care of this?
        # i.e. use the regular OK and Cancel IDs?
        button_a = self.close_button
        button_b = self.triangulate_button
        if sys.platform.startswith("win"):
            (button_a, button_b) = (button_b, button_a)
        self.button_sizer.Add(button_a, 0, wx.LEFT, border=self.SPACING)
        self.button_sizer.Add(button_b, 0, wx.LEFT, border=self.SPACING)

        self.sizer.Add(self.button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL,
                       border=self.SPACING
                       )

        self.panel.SetSizer(self.sizer)
        self.SetSizer(self.outer_sizer)

        self.sizer.Layout()
        self.Fit()
        self.Show()

        self.triangulate_button.Bind(wx.EVT_BUTTON, self.triangulate)
        self.close_button.Bind(wx.EVT_BUTTON, self.close)
        self.Bind(wx.EVT_CLOSE, self.close)

    def triangulate(self, event):
        self.triangulate_button.Enable(False)
        self.triangulate_button.SetLabel("Triangulating...")
        self.sizer.Layout()
        self.triangulate_internal()
        self.triangulate_button.Enable(True)
        self.triangulate_button.SetLabel("Triangulate")
        self.sizer.Layout()

    def triangulate_internal(self):
        q = None
        if (self.angle_text_box.GetValue().strip() != ""):
            try:
                q = float(self.angle_text_box.GetValue().strip())
            except:
                wx.MessageBox("The minimum angle you entered is not a valid number.", "Value Error")

                return

        a = None
        if (self.area_text_box.GetValue().strip() != ""):
            try:
                a = float(self.area_text_box.GetValue().strip())
            except:
                wx.MessageBox("The maximum area you entered is not a valid number.", "Value Error")

                return

        layer = self.project.layer_tree_control.get_selected_layer()
        if (layer == None or layer.points == None):
                wx.MessageBox("You must select a layer with points in the tree to triangulate.", "Triangulate")

                return

        t_layer = self.project.layer_manager.find_dependent_layer(layer, "triangles")
        if t_layer is None:
            t_layer = self.project.layer_manager.add_layer("vector", after=layer)
            self.project.layer_manager.set_dependent_layer(layer, "triangles", t_layer)
            t_layer.name = "Triangulated %s" % layer.name
        try:
            t_layer.triangulate_from_layer(layer, q, a)
        except Exception as e:
            print traceback.format_exc(e)
            wx.MessageBox(e.message, "Triangulate Error")

        self.project.refresh()
        self.project.layer_tree_control.select_layer(layer)

    def close(self, event):
        self.Destroy()

if __name__ == "__main__":
    """
    simple test for the dialog
    """
    a = wx.App(False)
    import wx.lib.inspection
    wx.lib.inspection.InspectionTool().Show()
    d = Triangle_dialog()
    d.Show()
    a.MainLoop()
