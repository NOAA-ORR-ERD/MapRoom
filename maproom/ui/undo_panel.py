#!/usr/bin/env python


import sys
import traceback
import wx

from omnimon.framework.errors import ProgressCancelError

import logging
progress_log = logging.getLogger("progress")


class UndoHistoryPanel(wx.Panel):
    def __init__(self, parent, task):
        self.task = task
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        
        # Mac/Win needs this, otherwise background color is black
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)

        self.SetHelpText("You can specify a minimum triangle angle (or leave blank if you don't want to specify a minimum). If the minimum angle is 20.7 degrees or smaller, the triangulation is theoretically guaranteed to terminate. It often succeeds for minimum angles up to 33 degrees. It usually doesn't terminate for angles above 34 degrees.")

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.history = wx.ListBox(self, size=(100, -1))

        self.sizer.Add(self.history, 1, wx.EXPAND)

        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.Fit()
    
    def set_task(self, task):
        self.task = task

    def update_history(self):
        project = self.task.active_editor
        summary = project.layer_manager.undo_stack.history_list()
        self.history.Set(summary)
        index = project.layer_manager.undo_stack.insert_index
        if index > 0:
            self.history.SetSelection(index - 1)

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
