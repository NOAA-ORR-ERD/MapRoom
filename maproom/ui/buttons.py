import wx
import wx.lib.buttons as buttons
import wx.lib.colourselect as csel
import wx.lib.agw.cubecolourdialog as CCD

import logging
log = logging.getLogger(__name__)


EVT_COLORSELECT = csel.EVT_COLOURSELECT
ColorSelect = csel.ColourSelect

GenBitmapToggleButton = buttons.GenBitmapToggleButton


class AlwaysAlphaCCD(CCD.CubeColourDialog):
    def DoLayout(self):
        CCD.CubeColourDialog.DoLayout(self)
        self.mainSizer.Hide(self.showAlpha)


if wx.Platform == "__WXMAC__":
    color_dialog = wx.ColourDialog
else:
    color_dialog = AlwaysAlphaCCD


def prompt_for_rgba(parent, color, custom=None, use_float=False):
    data = wx.ColourData()
    data.SetChooseFull(True)
    if use_float:
        color = [a * 255 for a in color]
    data.SetColour(color)
    if custom is not None:
        for idx, clr in enumerate(custom.Colours):
            if clr is not None:
                data.SetCustomColour(idx, clr)

    dlg = color_dialog(wx.GetTopLevelParent(parent), data)
    changed = dlg.ShowModal() == wx.ID_OK

    if changed:
        data = dlg.GetColourData()
        color = data.GetColour()
        if use_float:
            color = [float(c / 255.0) for c in color]
        if custom is not None:
            custom.Colours = \
                [data.GetCustomColour(idx) for idx in range(0, 16)]
    else:
        color = None

    dlg.Destroy()
    return color



class ColorSelectButton(ColorSelect):
    SetColor = ColorSelect.SetColour

    def MakeBitmap(self):
        """ Creates a bitmap representation of the current selected colour. """

        bdr = 8
        width, height = self.GetSize()
        #print("button:", width, height)

        # yes, this is weird, but it appears to work around a bug in wxMac
        if "wxMac" in wx.PlatformInfo and width == height:
            height -= 1

        w = max(width - bdr, 1)
        h = max(height - bdr, 1)
        bmp = wx.Bitmap(w, h)
        dc = wx.MemoryDC()
        dc.SelectObject(bmp)
        dc.SetFont(self.GetFont())
        label = self.GetLabel()
        # Just make a little colored bitmap
        fg = self.colour

        # bitmaps aren't able to use alpha, so  fake the alpha color on a white
        # background for the button color
        blend = tuple(wx.Colour.AlphaBlend(c, 255, fg.alpha / 255.0) for c in fg.Get(False))
        dc.SetBackground(wx.Brush(blend))
        dc.Clear()

        if label:
            # Add a label to it
            avg = functools.reduce(lambda a, b: a + b, self.colour.Get()) / 3
            fcolour = avg > 128 and wx.BLACK or wx.WHITE
            dc.SetTextForeground(fcolour)
            dc.DrawLabel(label, (0,0, w, h),
                         wx.ALIGN_CENTER)

        dc.SelectObject(wx.NullBitmap)
        return bmp

    def OnClick(self, event):
        color = prompt_for_rgba(self, self.colour, self.customColours)
        if color is not None:
            self.SetColour(color)
            self.OnChange()
