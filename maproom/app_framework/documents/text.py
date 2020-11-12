import os

from ..document import MafDocument
from ..filesystem import fsopen as open

import logging
log = logging.getLogger(__name__)


class TextDocument(MafDocument):
    def load_raw_data(self):
        fh = open(self.uri, 'r')
        self.raw_data = str(fh.read())

    def save_raw_data(self, uri):
        fh = open(uri, 'w')
        log.debug("saving to %s" % uri)
        fh.write(self.raw_data)
        fh.close()

    # won't automatically match anything; must force this editor with the -t
    # command line flag
    @classmethod
    def can_load_file_exact(cls, file_metadata):
        return False

    @classmethod
    def can_load_file_generic(cls, file_metadata):
        return file_metadata['mime'].startswith("text/")
