import re

from maproom.app_framework.filesystem import fsopen as open

from ..magic import magic_template
from .common import BaseLoader

WHITESPACE_PATTERN = re.compile("\s+")


def identify_loader(file_guess):
    if file_guess.is_text and file_guess.sample_data.startswith(magic_template.encode('utf-8')):
        return dict(mime=CommandLogLoader.mime, loader=CommandLogLoader())


class CommandLogLoader(BaseLoader):
    mime = "application/x-maproom-command-log"

    extensions = [".mrc"]

    name = "MapRoom Command Log"

    def can_save_layer(self, layer):
        return False

    def iter_log(self, uri, manager):
        with open(uri, "r") as fh:
            text = fh.read()
            for cmd in manager.undo_stack.unserialize_text(text, manager):
                yield cmd
