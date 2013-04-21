import re
import numpy as np


class Mock_file_scanner:
    # Simulate scanf()'s "%g"
    # From http://docs.python.org/library/re.html#simulating-scanf
    FLOAT_PATTERN = re.compile( "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?" )
    WHITESPACE_PATTERN = re.compile( "\s+" )

    @staticmethod
    def FileScanN_single( in_file, N ):
        data = np.empty( ( N, ), np.float32 )
        index = 0
        pos = in_file.tell()
        in_file.seek( 0 )
        contents = in_file.read()

        match = Mock_file_scanner.WHITESPACE_PATTERN.match( contents, pos )
        if match:
            pos += len( match.group( 0 ) )

        while True:
            match = Mock_file_scanner.FLOAT_PATTERN.match( contents, pos )
            if not match:
                break

            number = match.group( 0 )
            data[ index ] = np.float32( number )
            index += 1
            pos += len( number ) + 1

            match = Mock_file_scanner.WHITESPACE_PATTERN.match( contents, pos )
            if match:
                pos += len( match.group( 0 ) )

        if index != N:
            raise RuntimeError(
                "End of File reached before all numbers found",
            )

        in_file.seek( pos )

        return data
