import re
import json
import zipfile

from sawx.filesystem import fsopen as open

from .common import BaseLoader

import logging
log = logging.getLogger(__name__)


WHITESPACE_PATTERN = re.compile("\s+")


def identify_loader(file_guess):
    if file_guess.is_binary:
        if file_guess.is_zipfile and file_guess.zipfile_contains("pre json data"):
            return dict(mime="application/x-maproom-project-zip", loader=ZipProjectLoader())
    else:
        if file_guess.sample_data.startswith(b"# -*- MapRoom project file -*-"):
            return dict(mime="application/x-maproom-project-json", loader=ProjectLoader())


class ProjectLoader(BaseLoader):
    mime = "application/x-maproom-project-json"

    layer_types = []

    extensions = [".maproom"]

    name = "MapRoom Project"

    load_type = "project"

    def can_save_layer(self, layer):
        return False

    def load_project(self, uri, manager, batch_flags):
        project = []
        with open(uri, "r") as fh:
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

    def can_save_layer(self, layer):
        return False

    def load_project(self, uri, manager, batch_flags):
        log.debug("project file: %s" % uri)
        fh = open(uri, "rb")
        if zipfile.is_zipfile(fh):
            log.debug("found zipfile")
            with zipfile.ZipFile(fh, 'r') as zf:
                valid = True
                try:
                    info = zf.getinfo("extra json data")
                    valid = True
                except KeyError:
                    pass
                try:
                    info = zf.getinfo("pre json data")
                    valid = True
                except KeyError:
                    pass
                if valid:
                    log.debug("found extra json data")
                    layer_data, extra = manager.load_all_from_zip(zf, batch_flags)
                    layers = manager.add_all(layer_data)
                    batch_flags.layers.extend(layers)
                    manager.restore_layer_relationships_after_load()
                    return extra
        return "Not a MapRoom project zipfile!"
