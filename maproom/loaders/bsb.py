import os

import wx


from omnivore_framework.framework.errors import ProgressCancelError

from maproom.layers import RasterLayer
from maproom.library.bsb_utils import BSBParser

from .common import BaseLoader
from .gdal import load_image_file, get_dataset

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class BSBLoader(BaseLoader):
    mime = "application/x-maproom-bsb"

    layer_types = []

    extensions = [".bsb"]

    name = "NOAA BSB"

    def load_query(self, metadata, manager):
        bsb = BSBParser(metadata.uri)
        items = []
        for image in bsb.info.images:
            error, dataset = get_dataset(image.filename)
            items.append("%s: pane #%s (%d x %d) %s" % (os.path.basename(image.filename), image.pane, dataset.RasterXSize, dataset.RasterYSize, image.name))
            dataset = None  # close the GDAL dataset

        self.selected = []
        dlg = wx.MultiChoiceDialog(manager.project.window.control, "Select images from BSB file", "Choose Images", items)
        if (dlg.ShowModal() == wx.ID_OK):
            selections = dlg.GetSelections()
            for index in selections:
                self.selected.append(bsb.info.images[index].filename)

    def load_layers(self, metadata, manager, **kwargs):
        if not self.selected:
            raise ProgressCancelError("No files selected from BSB")
        layers = []
        for filename in self.selected:
            layer = RasterLayer(manager=manager)

            progress_log.info("Loading from %s" % metadata.uri)
            (layer.load_error_string, layer.image_data) = load_image_file(filename)
            if (layer.load_error_string == ""):
                progress_log.info("Finished loading %s" % filename)
                layer.file_path = filename
                layer.name = os.path.split(layer.file_path)[1]
                layer.mime = "image/*"
                layer.update_bounds()
                layers.append(layer)
        return layers
