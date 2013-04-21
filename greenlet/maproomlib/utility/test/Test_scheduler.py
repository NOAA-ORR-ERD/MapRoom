from maproomlib.utility.Scheduler import Scheduler


class Test_scheduler:
    def xsetUp( self ):
        self.scheduler = Scheduler()

    def xtest_switch( self ):
        raise NotImplementedError()

    def xtest_run( self ):
        raise NotImplementedError()

    def xtest_add( self ):
        raise NotImplementedError()

    def xtest_remove( self ):
        raise NotImplementedError()

    def xtest_sleep( self ):
        raise NotImplementedError()

    def xtest_wake( self ):
        raise NotImplementedError()

    def xtest_current_task( self ):
        raise NotImplementedError()

    def xtest_current( self ):
        raise NotImplementedError()

    def xtest_current_without_scheduler( self ):
        raise NotImplementedError()

    def xtest_shutdown( self ):
        raise NotImplementedError()
