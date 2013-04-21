import os.path
from StringIO import StringIO


class Mock_file:
    filesystem = {}

    def __init__( self, filename, mode = None ):
        self.filename = filename
        self.mode = mode

    def read( self, size = None ):
        return Mock_file.filesystem[ self.filename ].read( size )

    def readline( self, size = None ):
        return Mock_file.filesystem[ self.filename ].readline( size )

    def write( self, data ):
        if self.filename in Mock_file.filesystem:
            Mock_file.filesystem[ self.filename ].write( data )
        else:
            Mock_file.filesystem[ self.filename ] = StringIO( data )

    def tell( self ):
        return Mock_file.filesystem[ self.filename ].tell()

    def seek( self, offset ):
        Mock_file.filesystem[ self.filename ].seek( offset )

    def close( self ):
        Mock_file.filesystem[ self.filename ].seek( 0 )

    @staticmethod
    def exists( filename ):
        if filename in Mock_file.filesystem:
            return True

        for existing_filename in Mock_file.filesystem:
            if existing_filename.startswith( filename + os.path.sep ):
                return True

        return False

    @staticmethod
    def listdir( dirname ):
        filenames = []

        for filename in Mock_file.filesystem.keys():
            if filename.startswith( dirname + os.path.sep ):
                filename = filename[ len( dirname + os.path.sep ) : ]
                filenames.append( filename.split( os.path.sep )[ 0 ] )

        return filenames

    @staticmethod
    def isdir( dirname ):
        for filename in Mock_file.filesystem.keys():
            if os.path.dirname( filename ) == dirname:
                return True

        return False

    @staticmethod
    def reset():
        Mock_file.filesystem = {}
