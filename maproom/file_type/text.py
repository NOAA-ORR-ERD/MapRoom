from traits.api import HasTraits, provides

from peppy2.file_type.i_file_recognizer import IFileRecognizer

@provides(IFileRecognizer)
class MapRoomProjectRecognizer(HasTraits):
    """Default plain text identifier based on percentage of non-ASCII bytes.
    
    """
    id = "application/x-maproom-project-json"
    
    before = "text/plain"
    
    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith("# -*- MapRoom project file -*-"):
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
            return "application/x-maproom-bna"
