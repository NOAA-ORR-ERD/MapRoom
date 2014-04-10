from traits.api import HasTraits, provides

from peppy2.file_type.i_file_recognizer import IFileRecognizer

@provides(IFileRecognizer)
class VerdatRecognizer(HasTraits):
    """Default plain text identifier based on percentage of non-ASCII bytes.
    
    """
    id = "application/x-maproom-verdat"
    
    before = "text/plain"
    
    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream.startswith("DOGS"):
            return "application/x-maproom-verdat"

@provides(IFileRecognizer)
class BNARecognizer(HasTraits):
    """Default plain text identifier based on percentage of non-ASCII bytes.
    
    """
    id = "application/x-maproom-bna"
    
    before = "text/plain"
    
    def identify(self, guess):
        if guess.metadata.uri.lower().endswith(".bna"):
            return "application/x-maproom-bna"
