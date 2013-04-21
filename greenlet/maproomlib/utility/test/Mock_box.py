class Mock_box:
    def __init__( self ):
        self.sent_message = None

    def send( self, **message ):
        self.sent_message = message
