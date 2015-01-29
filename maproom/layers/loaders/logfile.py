import os
import numpy as np
import re
import json

from common import BaseLoader
from maproom.layers import Layer
from maproom.serializer import magic_template, TextDeserializer

WHITESPACE_PATTERN = re.compile("\s+")


class CommandLogLoader(BaseLoader):
    mime = "application/x-maproom-command-log"
    
    extensions = [".mrc"]
    
    name = "MapRoom Command Log"
    
    def iter_log(self, metadata, manager):
        project = []
        with open(metadata.uri, "r") as fh:
            text = fh.read()
            s = TextDeserializer(text)
            for cmd in s.iter_cmds(manager):
                yield cmd
