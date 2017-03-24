import wx
import wx.lib.sized_controls  # for control border calcs


class SliderLabel(wx.Panel):

    def __init__(self, parent, id, value, minValue, maxValue, valueUnit):
        wx.Panel.__init__(self, parent, id)

        self.minValue = minValue
        self.maxValue = maxValue
        self.value = value
        self.valueUnit = valueUnit

        self.SPACING = wx.lib.sized_controls.GetDefaultBorder(self)

        self.Sizer = wx.BoxSizer(wx.VERTICAL)

        self.label_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.label_sizer.Add(
            wx.StaticText(
                self,
                wx.ID_ANY,
                self.FormatValue(minValue),
                style=wx.ALIGN_RIGHT
            ),
            0, wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.value_label = wx.StaticText(
            self,
            wx.ID_ANY,
            self.FormatValue(self.value),
            style=wx.ALIGN_CENTRE
        )

        value_font = self.value_label.GetFont()
        value_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.value_label.SetFont(value_font)

        self.label_sizer.Add(
            self.value_label,
            1, wx.EXPAND
        )

        self.label_sizer.Add(
            wx.StaticText(
                self,
                wx.ID_ANY,
                self.FormatValue(maxValue),
                style=wx.ALIGN_RIGHT
            ),
            0, wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.Sizer.Add(
            self.label_sizer,
            1, wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

    def GetValue(self):
        return self.value_label.Label

    def SetValue(self, value):
        self.value_label.Label = self.FormatValue(value)
        self.Layout()

    def FormatValue(self, value):
        return "%s%s" % (value, self.valueUnit)

    Value = property(GetValue, SetValue)


class FloatSlider(wx.PyPanel):

    """
    A Slider control that accepts float values.
    """

    def __init__(self, parent, id, value, minValue, maxValue, steps=1000, valueUnit='', point=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.SL_HORIZONTAL, validator=wx.DefaultValidator,
                 name="FloatSlider"):

        wx.Panel.__init__(self, parent, id)

        self.Sizer = wx.BoxSizer(wx.VERTICAL)

        BORDER = wx.lib.sized_controls.GetDefaultBorder(self)

        self.sliderLabels = None
        self.steps = steps
        self.stepValue = 0
        self.maxValue = maxValue
        self.minValue = minValue
        self._ConstraintsUpdated()

        self.value = value
        self.stepValue = self.float_to_slider_value(value)
        self.valueUnit = valueUnit

        self.sliderCtrl = wx.Slider(self, -1, self.stepValue, 0, steps, point, size, style & ~wx.SL_LABELS, validator, name)
        self.Sizer.Add(self.sliderCtrl, 0, wx.EXPAND | wx.ALL, BORDER)

        if style & wx.SL_LABELS:
            self.sliderLabels = SliderLabel(self, -1, value, minValue, maxValue, valueUnit)
            self.Sizer.Add(self.sliderLabels, 0, wx.EXPAND | wx.ALL, BORDER)

        self.sliderCtrl.Bind(wx.EVT_SLIDER, self.OnSliderChanged)

    def _ConstraintsUpdated(self):
        self.step_size = (self.maxValue - self.minValue) / self.steps

    def Enable(self, enable=True):
        self.sliderCtrl.Enable(enable)

    def slider_value_to_float(self, slider_value):
        return self.minValue + (slider_value * self.step_size)

    def float_to_slider_value(self, value):
        return int((value - self.minValue) / self.step_size)

    def GetValue(self):
        return self.slider_value_to_float(self.sliderCtrl.Value)

    def SetValue(self, value):
        self.value = value
        self.stepValue = self.float_to_slider_value(value)
        if self.stepValue > self.steps:
            print "Value %s is outside of slider bounds." % self.stepValue
            wx.Bell()
            return
        self.sliderCtrl.Value = self.stepValue
        if self.sliderLabels:
            self.sliderLabels.Value = self.value

    def OnSliderChanged(self, event):
        event.Skip()  # so users of this control can add their own handling

        if self.sliderLabels:
            self.sliderLabels.Value = "%s" % self.GetValue()

    Value = property(GetValue, SetValue)


class TextSlider(wx.PyPanel):

    """
    This control is a composite of a text box and a slider, where changes in one are
    reflected in the other. The API tries its best to mimic wx.Slider so that it can
    pretty much act as a drop-in replacement.
    """

    def __init__(self, parent, id, value, minValue, maxValue, steps=1000, valueUnit='', point=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.SL_HORIZONTAL, validator=wx.DefaultValidator,
                 name="ValueSlider"):
        wx.Panel.__init__(self, parent, id)

        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Set the slider width small to work around layout problem with the
        # initial size of the slider being too wide for the sidebar panel
        # width & pushing the text control off to the right
        self.sliderCtrl = FloatSlider(self, -1, value, minValue, maxValue, steps, valueUnit, point, (50, -1), style, validator, name)
        self.Sizer.Add(self.sliderCtrl, 2, wx.CENTER)
        self.textCtrl = wx.SpinCtrlDouble(self, value='0.00', size=(50, 21),
                                          min=minValue, max=maxValue, inc=(maxValue - minValue) / steps)
        self.textCtrl.SetDigits(0)
        self.Sizer.Add(self.textCtrl, 1, wx.CENTER)

        self.textCtrl.Bind(wx.EVT_TEXT, self.OnTextChanged)
        self.sliderCtrl.Bind(wx.EVT_SLIDER, self.OnSliderChanged)

    def Enable(self, enable=True):
        self.textCtrl.Enable(enable)
        self.sliderCtrl.Enable(enable)

    def GetValue(self):
        return self.sliderCtrl.GetValue()

    def SetValue(self, value):
        self.sliderCtrl.SetValue(value)
        self.textCtrl.Value = value

    def OnTextChanged(self, event):
        event.Skip()
        print event.String
        if event.String.strip() != "":
            try:
                value = origValue = float(event.String)
            except ValueError:
                return
            if value < self.sliderCtrl.minValue:
                value = self.sliderCtrl.minValue
            if value > self.sliderCtrl.maxValue:
                value = self.sliderCtrl.maxValue
            if value != origValue:
                self.textCtrl.Value = value
            self.sliderCtrl.Value = value
        else:
            self.Value = self.sliderCtrl.minValue

    def OnSliderChanged(self, event):
        event.Skip()
        print "slider", self.sliderCtrl.GetValue()
        self.textCtrl.Value = self.sliderCtrl.GetValue()

    Value = property(GetValue, SetValue)


if __name__ == "__main__":
    app = wx.PySimpleApp()

    frame = wx.Frame(None, -1, "TextSlider test")
    TextSlider(frame, -1, 1000.6, 1000.0, 2000.0, steps=10000, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
    frame.Show()

    app.MainLoop()
