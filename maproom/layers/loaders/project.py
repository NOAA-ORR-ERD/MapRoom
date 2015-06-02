import os
import numpy as np
import re
import json

from common import BaseLoader
from maproom.layers import Layer

WHITESPACE_PATTERN = re.compile("\s+")


class ProjectLoader(BaseLoader):
    mime = "application/x-maproom-project-json"
    
    layer_types = []
    
    extensions = [".maproom"]
    
    name = "MapRoom Project"
    
    load_type = "project"
    
    def load_project(self, metadata, manager, batch_flags):
        project = []
        with open(metadata.uri, "r") as fh:
            line = fh.readline()
            if line != "# -*- MapRoom project file -*-\n":
                return "Not a MapRoom project file!"
            
            project = json.load(fh)
            layers = manager.load_all_from_json(project)
            batch_flags.layers.extend(layers)
            manager.add_all(layers)
