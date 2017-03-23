from traits.api import HasTraits, provides

from omnivore.file_type.i_file_recognizer import IFileRecognizer

from maproom.serializer import magic_template


@provides(IFileRecognizer)
class MapRoomProjectRecognizer(HasTraits):
    """Finds project files using magic text at the beginning of the file
    
    """
    id = "application/x-maproom-project-json"

    before = "text/plain"

    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith("# -*- MapRoom project file -*-"):
            return self.id


@provides(IFileRecognizer)
class MapRoomCommandRecognizer(HasTraits):
    """Finds command log files using magic text at the beginning of the file
    
    """
    id = "application/x-maproom-command-log"

    before = "text/plain"

    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith(magic_template):
            return self.id


@provides(IFileRecognizer)
class VerdatRecognizer(HasTraits):
    """Finds verdat files -- looks for the "DOGS" in the header.
    
    """
    id = "application/x-maproom-verdat"

    before = "text/plain"

    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith("DOGS"):
            return "application/x-maproom-verdat"


@provides(IFileRecognizer)
class BNARecognizer(HasTraits):
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
class BSBRecognizer(HasTraits):
    """Finds BSB files -- simply looks for extension
    
    """
    id = "application/x-maproom-bsb"

    before = "text/plain"

    def identify(self, guess):
        if guess.metadata.uri.lower().endswith(".bsb"):
            return "application/x-maproom-bsb"
