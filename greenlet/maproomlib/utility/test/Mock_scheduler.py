class Mock_scheduler:
    def __init__( self, sleep_callback = None ):
        self.running = []
        self.sleeping = []
        self.sleep_callback = sleep_callback

    def add( self, task ):
        if task in self.sleeping:
            self.sleeping.remove( task )

        self.running.append( task )

    def remove( self, task ):
        if task in self.running:
            self.running.remove( task )

        self.running.remove( task )

    def sleep( self, task = None, timeout = None ):
        # If there's a timeout value, then just simulate a timeout
        # unconditionally.
        if timeout:
            return True

        if task in self.running:
            self.running.remove( task )

        self.sleeping.append( task )

        if self.sleep_callback:
            self.sleep_callback()

        return False

    def wake( self, task ):
        self.add( task )

    def current_task( self ):
        return "foobar" # TODO

    @staticmethod
    def current():
        raise NotImplementedError()
