import re
import json
import zipfile

from fs.opener import fsopen

from common import BaseLoader

import logging
log = logging.getLogger(__name__)


WHITESPACE_PATTERN = re.compile("\s+")


class ProjectLoader(BaseLoader):
    mime = "application/x-maproom-project-json"

    layer_types = []

    extensions = [".maproom"]

    name = "MapRoom Project"

    load_type = "project"

    def load_project(self, metadata, manager, batch_flags):
        project = []
        with fsopen(metadata.uri, "r") as fh:
            line = fh.readline()
            if line != "# -*- MapRoom project file -*-\n":
                return "Not a MapRoom project file!"

            project = json.load(fh)
            layer_data, extra = manager.load_all_from_json(project, batch_flags)
            layers = manager.add_all(layer_data)
            batch_flags.layers.extend(layers)
            return extra


class ZipProjectLoader(BaseLoader):
    mime = "application/x-maproom-project-zip"

    layer_types = []

    extensions = [".maproom"]

    name = "MapRoom Project Zip File"

    load_type = "project"

    def load_project(self, metadata, manager, batch_flags):
        log.debug("project file: %s" % metadata.uri)
        if zipfile.is_zipfile(metadata.syspath):
            log.debug("found zipfile")
            with zipfile.ZipFile(metadata.syspath, 'r') as zf:
                try:
                    info = zf.getinfo("extra json data")
                except KeyError:
                    pass
                else:
                    log.debug("found extra json data")
                    layer_data, extra = manager.load_all_from_zip(metadata.syspath, zf, batch_flags)
                    layers = manager.add_all(layer_data)
                    batch_flags.layers.extend(layers)
                    return extra
        return "Not a MapRoom project zipfile!"
