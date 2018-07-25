import re

from fs.opener import fsopen

from .common import BaseLoader

WHITESPACE_PATTERN = re.compile("\s+")


class CommandLogLoader(BaseLoader):
    mime = "application/x-maproom-command-log"

    extensions = [".mrc"]

    name = "MapRoom Command Log"

    def iter_log(self, metadata, manager):
        with fsopen(metadata.uri, "r") as fh:
            text = fh.read()
            for cmd in manager.undo_stack.unserialize_text(text, manager):
                yield cmd
