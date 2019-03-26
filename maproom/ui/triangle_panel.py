#!/usr/bin/env python


import wx


from ..menu_commands import TriangulateLayerCommand

import logging
progress_log = logging.getLogger("progress")


class TrianglePanel(wx.Panel):
    SPACING = 15
    NAME = "Triangulate Layer"

    def __init__(self, parent, editor):
        self.editor = editor
        wx.Panel.__init__(self, parent, wx.ID_ANY, name="Triangulate")

        # Mac/Win needs this, otherwise background color is black
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)

        self.SetHelpText("You can specify a minimum triangle angle (or leave blank if you don't want to specify a minimum). If the minimum angle is 20.7 degrees or smaller, the triangulation is theoretically guaranteed to terminate. It often succeeds for minimum angles up to 33 degrees. It usually doesn't terminate for angles above 34 degrees.")

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # put the various dialog items into sizer, which will become the sizer of panel

        # quality mesh

        box = wx.StaticBox(self, label="Quality Mesh Minimum Angle")
        s = wx.StaticBoxSizer(box, wx.VERTICAL)
        #  # t = wx.StaticText(self, label="You can specify a minimum triangle angle (or leave blank if you don't want to specify a minimum). If the minimum angle is 20.7 degrees or smaller, the triangulation is theoretically guaranteed to terminate. It often succeeds for minimum angles up to 33 degrees. It usually doesn't terminate for angles above 34 degrees.", pos=(0, 0))
        #  # s.Add(t, 0, wx.ALIGN_TOP | wx.ALL, 5)

        # t = wx.StaticText(self, label="Minimum angle:")
        # s.Add(t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        s2 = wx.BoxSizer(wx.HORIZONTAL)
        self.angle_text_box = wx.TextCtrl(self, 0, "", size=(100, -1), name="TrainglePanel.angle_text_box")
        self.angle_text_box.SetHelpText("You can specify a minimum triangle angle (or leave blank if you don't want to specify a minimum). If the minimum angle is 20.7 degrees or smaller, the triangulation is theoretically guaranteed to terminate. It often succeeds for minimum angles up to 33 degrees. It usually doesn't terminate for angles above 34 degrees.")
        self.angle_text_box.SetToolTip("You can specify a minimum triangle angle (or leave blank if you don't want to specify a minimum). If the minimum angle is 20.7 degrees or smaller, the triangulation is theoretically guaranteed to terminate. It often succeeds for minimum angles up to 33 degrees. It usually doesn't terminate for angles above 34 degrees.")
        s2.Add(self.angle_text_box, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        t = wx.StaticText(self, label="degrees")
        s2.Add(t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        s.Add(s2, 0, wx.ALIGN_TOP | wx.ALL, 5)

        self.sizer.Add(s, 0, wx.ALIGN_TOP | wx.ALL, 5)

        # maximum triangle area

        box = wx.StaticBox(self, label="Maximum Triangle Area")
        s = wx.StaticBoxSizer(box, wx.VERTICAL)

        s2 = wx.BoxSizer(wx.HORIZONTAL)
        self.area_text_box = wx.TextCtrl(self, 0, "", size=(100, -1), name="TrainglePanel.area_text_box")
        self.area_text_box.SetToolTip("You can specify a maximum triangle area (or leave blank if you don't want to specify a maximum). The units are those of the point coordinates on this layer.")
        s2.Add(self.area_text_box, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        t = wx.StaticText(self, label="units")
        s2.Add(t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        s.Add(s2, 0, wx.ALIGN_TOP | wx.ALL, 5)

        self.sizer.Add(s, 0, wx.ALIGN_TOP | wx.ALL, 5)

        # the buttons

        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.triangulate_button = wx.Button(self, label="Triangulate")
        self.triangulate_button.SetDefault()
        self.triangulate_button.Bind(wx.EVT_BUTTON, self.triangulate)

        self.button_sizer.Add(self.triangulate_button, 0, wx.LEFT, border=self.SPACING)

        self.sizer.Add(self.button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL,
                       border=self.SPACING
                       )

        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.Fit()

    def triangulate(self, event):
        if not self.IsShown():
            # Hack for OS X: default buttons can still get Return key presses
            # even when the panel is hidden, which is not what we want.
            event.Skip()
            return

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
                self.editor.error("The minimum angle you entered is not a valid number.", "Value Error")

                return

        a = None
        if (self.area_text_box.GetValue().strip() != ""):
            try:
                a = float(self.area_text_box.GetValue().strip())
            except:
                self.editor.error("The maximum area you entered is not a valid number.", "Value Error")

                return

        project = self.editor
        layer = project.layer_tree_control.get_edit_layer()
        if layer.type == "triangle":
            # If attempting to triangulate a triangle layer, check if it is an
            # already-triangulated layer, and if so, use its parent
            parent = layer.manager.find_parent_of_dependent_layer(layer, layer.type)
            if parent is not None:
                layer = parent
        if (layer is None or not layer.has_points()):
                self.editor.error("To triangulate, use the Layers tree to select a layer that contains points.", "Triangulate")
                return

        cmd = TriangulateLayerCommand(layer, q, a)
        project.process_command(cmd)
