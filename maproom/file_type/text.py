from traits.api import HasTraits, provides

from omnivore.file_type.i_file_recognizer import IFileRecognizer, RecognizerBase

from maproom.serializer import magic_template
from maproom.library.lat_lon_parser import parse_coordinate_text

@provides(IFileRecognizer)
class MapRoomProjectRecognizer(RecognizerBase):
    """Finds project files using magic text at the beginning of the file
    
    """
    id = "application/x-maproom-project-json"

    before = "text/plain"

    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith("# -*- MapRoom project file -*-"):
            return self.id


@provides(IFileRecognizer)
class MapRoomCommandRecognizer(RecognizerBase):
    """Finds command log files using magic text at the beginning of the file
    
    """
    id = "application/x-maproom-command-log"

    before = "text/plain"

    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith(magic_template):
            return self.id


@provides(IFileRecognizer)
class VerdatRecognizer(RecognizerBase):
    """Finds verdat files -- looks for the "DOGS" in the header.
    
    """
    id = "application/x-maproom-verdat"

    before = "text/plain"

    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith("DOGS"):
            return "application/x-maproom-verdat"


@provides(IFileRecognizer)
class BNARecognizer(RecognizerBase):
    """Finds bna files -- simply looks for extension
    
    """
    id = "application/x-maproom-bna"

    before = "text/plain"

    def identify(self, guess):
        if guess.metadata.uri.lower().endswith(".bna"):
            byte_stream = guess.get_utf8()[:1000]
            lines = byte_stream.splitlines()
            if ".KAP" in lines[0]:
                return "application/x-maproom-rncloader"
            return "application/x-maproom-bna"


@provides(IFileRecognizer)
class BSBRecognizer(RecognizerBase):
    """Finds BSB files -- simply looks for extension
    
    """
    id = "application/x-maproom-bsb"

    before = "text/plain"

    def identify(self, guess):
        if guess.metadata.uri.lower().endswith(".bsb"):
            return "application/x-maproom-bsb"


@provides(IFileRecognizer)
class PlainTextRecognizer(RecognizerBase):
    """Finds plain-text lat/lon or lon/lat files
    
    """
    id = "text/latlon"

    before = "text/plain"

    after = "text/*"

    def identify(self, guess):
        byte_stream = guess.get_utf8()
        mime, _, _ = parse_coordinate_text(byte_stream)
        if mime is not None:
            return mime
