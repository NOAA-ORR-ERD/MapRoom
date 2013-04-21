import wx
import sys
import maproomlib
import maproomlib.ui

class Triangle_maker( wx.Dialog ):
    SPACING = 15
    
    def __init__( self, frame, root_layer, transformer, command_stack ):
        wx.Dialog.__init__(
            self, None, wx.ID_ANY, "Triangulate Layer",
            style = wx.DEFAULT_DIALOG_STYLE | wx.DIALOG_NO_PARENT,
        )
        self.SetIcon( frame.GetIcon() )
        
        self.frame = frame
        self.root_layer = root_layer
        self.transformer = transformer
        self.command_stack = command_stack
        self.inbox = maproomlib.ui.Wx_inbox()
        
        self.outer_sizer = wx.BoxSizer( wx.VERTICAL )
        self.sizer = wx.BoxSizer( wx.VERTICAL )
        
        self.panel = wx.Panel( self, wx.ID_ANY )
        self.outer_sizer.Add( self.panel, 1, wx.EXPAND )
        
        # put the various dialog items into sizer, which will become the sizer of panel
        
        ## quality mesh
        
        box = wx.StaticBox( self.panel, -1, "Quality Mesh Minimum Angle", pos = ( 0, 0 ), size = ( 400, 170 ) )
        s = wx.StaticBoxSizer( box, wx.VERTICAL )
        t = wx.StaticText( self.panel, -1, "You can specify a minimum triangle angle (or leave blank if you don't want to specify a minimum).", pos = ( 0, 0 ) )
        s.Add( t, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        t = wx.StaticText( self.panel, -1, "If the minimum angle is 20.7 degrees or smaller, the triangulation is theoretically guaranteed to terminate.", pos = ( 0, 0 ) )
        s.Add( t, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        t = wx.StaticText( self.panel, -1, "It often succeeds for minimum angles up to 33 degrees. It usually doesn't terminate for angles above 34 degrees.", pos = ( 0, 0 ) )
        s.Add( t, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        
        s2 = wx.BoxSizer( wx.HORIZONTAL )
        t = wx.StaticText( self.panel, -1, "Minimum angle:" )
        s2.Add( t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5 )
        self.angle_text_box = wx.TextCtrl( self.panel, 0, "", size = ( 100, -1 ) )
        s2.Add( self.angle_text_box, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5 )
        t = wx.StaticText( self.panel, -1, "degrees" )
        s2.Add( t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5 )
        s.Add( s2, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        
        self.sizer.Add( s, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        
        ## maximum triangle area
        
        box = wx.StaticBox( self.panel, -1, "Maximum Triangle Area", pos = ( 0, 0 ), size = ( 400, 170 ) )
        s = wx.StaticBoxSizer( box, wx.VERTICAL )
        t = wx.StaticText( self.panel, -1, "You can specify a maximum triangle area (or leave blank if you don't want to specify a maximum).", pos = ( 0, 0 ) )
        s.Add( t, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        t = wx.StaticText( self.panel, -1, "The units are those of the point coordinates on this layer.", pos = ( 0, 0 ) )
        s.Add( t, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        
        s2 = wx.BoxSizer( wx.HORIZONTAL )
        t = wx.StaticText( self.panel, -1, "Maximum area:" )
        s2.Add( t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5 )
        self.area_text_box = wx.TextCtrl( self.panel, 0, "", size = ( 100, -1 ) )
        s2.Add( self.area_text_box, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5 )
        t = wx.StaticText( self.panel, -1, "units" )
        s2.Add( t, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5 )
        s.Add( s2, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        
        self.sizer.Add( s, 0, wx.ALIGN_TOP | wx.ALL, 5 )
        
        ## the buttons
        
        self.button_sizer = wx.BoxSizer( wx.HORIZONTAL )

        self.triangulate_button_id = wx.NewId()
        self.triangulate_button = wx.Button(
            self.panel,
            self.triangulate_button_id,
            "Triangulate",
        )
        self.triangulate_button.SetDefault()

        self.close_button_id = wx.NewId()
        self.close_button = wx.Button(
            self.panel,
            self.close_button_id,
            "Close",
        )

        # Dialog button ordering, by convention, is different between Mac and Windows.
        button_a = self.close_button
        button_b = self.triangulate_button
        if sys.platform.startswith( "win" ):
            ( button_a, button_b ) = ( button_b, button_a )
        self.button_sizer.Add( button_a, 0, wx.LEFT, border = self.SPACING )
        self.button_sizer.Add( button_b, 0, wx.LEFT, border = self.SPACING )
        
        self.sizer.Add(
            self.button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL,
            border = self.SPACING,
        )
        
        self.panel.SetSizer( self.sizer )
        # txt = wx.TextCtrl(self, -1, "20", size=(30,-1))
        # self.outer_sizer.Add( txt, 0, wx.ALIGN_CENTRE|wx.ALL, 5 )
        self.SetSizer( self.outer_sizer )
        
        self.sizer.Layout()
        self.Fit()
        self.Show()
        
        self.triangulate_button.Bind( wx.EVT_BUTTON, self.triangulate )
        self.close_button.Bind( wx.EVT_BUTTON, self.close )
        
        self.inbox = maproomlib.ui.Wx_inbox()
            
    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = ( "triangulate", "show", "close" ),
            )
            
            request = message.pop( "request" )
            
            if request == "triangulate":
                self.triangulate_button.Enable( False )
                self.triangulate_button.SetLabel( "Triangulating..." )
                self.sizer.Layout()
                
                self.command_stack.inbox.send( request = "start_command" )
                self.root_layer.inbox.send( 
                    request = "triangulate_selected",
                    **message
                )

                try:
                    self.inbox.receive()
                except ( NotImplementedError ), error:
                    self.triangulate_button.Enable( True )
                    self.triangulate_button.SetLabel( "Triangulate" )
                    self.sizer.Layout()
                    wx.MessageDialog(
                        self.frame,
                        message = str( error ),
                        style = wx.OK | wx.ICON_ERROR,
                    ).ShowModal()
                    continue

                self.triangulate_button.Enable( True )
                self.triangulate_button.SetLabel( "Triangulate" )
                self.sizer.Layout()
            elif request == "show":
                self.Show()
                self.Raise()
            elif request == "close":
                return

    def triangulate( self, event ):
        q = None
        if ( self.angle_text_box.GetValue().strip() != "" ):
            try:
                q = float( self.angle_text_box.GetValue().strip() )
            except:
                wx.MessageBox( "The minimum angle you entered is not a valid number.", "Value Error")
                
                return
        
        a = None
        if ( self.area_text_box.GetValue().strip() != "" ):
            try:
                a = float( self.area_text_box.GetValue().strip() )
            except:
                wx.MessageBox( "The maximum area you entered is not a valid number.", "Value Error")
                
                return
        
        self.inbox.send(
            request = "triangulate",
            q = q,
            a = a,
            transformer = self.transformer,
            response_box = self.inbox
        )

    def close( self, event ):
        event.Skip()
        self.Hide()
