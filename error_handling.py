import sys
import traceback

import wx
import wx.lib.sized_controls as sc

inExceptHook = False


def tracebackAsString(type, value, trace):
    list = traceback.format_exception(type, value, trace)
    return "".join(list)


class ErrorDialog(sc.SizedDialog):

    def __init__(self, parent, id):
        sc.SizedDialog.__init__(self, parent, id, "Maproom Error Reporter",
                                style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        # Always use self.GetContentsPane() - this ensures that your dialog
        # automatically adheres to HIG spacing requirements on all platforms.
        # pane here is a sc.SizedPanel with a vertical sizer layout. All children
        # should be added to this pane, NOT to self.
        pane = self.GetContentsPane()

        # second row
        self.lblDetails = wx.StaticText(pane, -1, "Error Details")

        # third row
        self.details = wx.TextCtrl(pane, -1, size=(300, 240), style=wx.TE_MULTILINE)
        self.details.SetSizerProps(expand=True, proportion=1)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())

        self.submitBtn = self.FindWindowById(wx.ID_OK)
        self.submitBtn.Label = "Submit"

        self.FindWindowById(wx.ID_CANCEL).Label = "Ignore and Close"

        self.Fit()
        self.SetMinSize(self.GetSize())


def guiExceptionHook(exctype, value, trace):
    """
    We override the system error handler as soon as possible so that we can present
    this GUI to users for even very early errors. That is why we might need to initialize
    the GUI and start up the event loop.
    """

    if not wx.GetApp():
        app = wx.PySimpleApp()
        app.MainLoop()

    global inExceptHook
    # if we're entering an error handler recursively, just bail out and let the original
    # error get handled
    if not inExceptHook:
        inExceptHook = True

        dialog = ErrorDialog(None, wx.ID_ANY)
        dialog.details.Value = tracebackAsString(exctype, value, trace)
        if dialog.ShowModal() == wx.ID_OK:
            import smtplib

            # Import the email modules we'll need
            from email.mime.text import MIMEText
            try:
                # Open a plain text file for reading.  For this example, assume that
                # the text file contains only ASCII characters.
                #fp = open(textfile, 'rb')
                # Create a text/plain message
                msg = MIMEText(dialog.details.Value)
                # fp.close()

                # me == the sender's email address
                # you == the recipient's email address
                msg['Subject'] = 'Maproom Error Report'
                msg['From'] = 'maproombugreports@gmail.com'
                msg['To'] = 'maproombugreports@gmail.com'

                # Send the message via our own SMTP server, but don't include the
                # envelope header.
                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.starttls()
                s.login("maproombugreports@gmail.com", "bushy206")
                s.sendmail(msg['From'], [msg['To']], msg.as_string())
                s.quit()
            except:
                wx.MessageBox("Unable to send email. Please email the bug report to maproombugreports@gmail.com")

        app = wx.GetApp()
        for tlw in wx.GetTopLevelWindows():
            tlw.Destroy()

        app.ExitMainLoop()

sys.excepthook = guiExceptionHook
