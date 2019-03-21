import re

from sawx.filesystem import fsopen as open
from sawx.utils.textutil import guessBinary

from ..magic import magic_template
from .common import BaseLoader

WHITESPACE_PATTERN = re.compile("\s+")


def identify_mime(uri, fh, header):
    is_binary = guessBinary(header)
    if not is_binary and header.startswith(magic_template.encode('utf-8')):
        return dict(mime=CommandLogLoader.mime, loader=CommandLogLoader())


class CommandLogLoader(BaseLoader):
    mime = "application/x-maproom-command-log"

    extensions = [".mrc"]

    name = "MapRoom Command Log"

    def iter_log(self, uri, manager):
        with open(uri, "r") as fh:
            text = fh.read()
            for cmd in manager.undo_stack.unserialize_text(text, manager):
                yield cmd
