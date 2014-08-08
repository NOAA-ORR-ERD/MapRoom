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
    
    project = True
    
    def load(self, metadata, manager):
        project = []
        with open(metadata.uri, "r") as fh:
            line = fh.readline()
            if line != "# -*- MapRoom project file -*-\n":
                return "Not a MapRoom project file!"
            
            project = json.load(fh)
        
        layers = []
        for serialized_data in project:
            loaded = Layer.load_from_json(serialized_data, manager)
            layers.extend(loaded)

        return layers
