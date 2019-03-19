import wx


import logging
progress_log = logging.getLogger("progress")


class UndoHistoryPanel(wx.Panel):
    def __init__(self, parent, editor):
        self.editor = editor
        wx.Panel.__init__(self, parent, wx.ID_ANY, name="Undo History")

        # Mac/Win needs this, otherwise background color is black
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.history = wx.ListBox(self, size=(100, -1))

        self.sizer.Add(self.history, 1, wx.EXPAND)

        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.Fit()

    def update_history(self):
        project = self.editor
        summary = project.layer_manager.undo_stack.history_list()
        self.history.Set(summary)
        index = project.layer_manager.undo_stack.insert_index
        if index > 0:
            self.history.SetSelection(index - 1)
