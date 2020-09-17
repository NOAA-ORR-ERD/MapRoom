import os

from maproom.app_framework.utils.runtime import get_all_subclasses

from . import common

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


_loaders = None

def get_known_loaders():
    global _loaders

    if _loaders is None:
        _loaders = get_all_subclasses(common.BaseLoader)
    log.debug(f"known loaders: {_loaders}")
    return _loaders

def load_layers_from_url(url, mime, manager=None):
    from maproom.app_framework.utils.file_guess import FileGuess

    guess = FileGuess(url)
    guess.metadata.mime = mime
    metadata = guess.get_metadata()
    return load_layers(metadata, manager)


def get_loader(metadata):
    for loader in get_known_loaders():
        log.debug("trying loader %s" % loader.name)
        if loader.can_load(metadata):
            log.debug(" loading using loader %s!" % loader.name)
            return loader
    return None


def load_layers(metadata, manager=None, **kwargs):
    for loader in get_known_loaders():
        log.debug("trying loader %s" % loader.name)
        if loader.can_load(metadata):
            log.debug(" loading using loader %s!" % loader.name)
            layers = loader.load_layers(metadata, manager=manager, **kwargs)
            log.debug(" loaded layers: \n  %s" % "\n  ".join([str(a) for a in layers]))
            return loader, layers
    return None, None


def valid_save_formats(layer):
    valid = []
    log.debug(f"checking layer {layer.type}")
    for loader in get_known_loaders():
        loader = loader()
        log.debug(f"checking loader {loader}")
        if loader.can_save_layer(layer):
            valid.append((loader, "%s: %s" % (loader.name, loader.get_pretty_extension_list())))
    log.debug(f"valid: {valid}")
    return valid


def get_valid_string(valid, capitalize=True):

    return "This layer can be saved in the following formats:\n(with allowed filename extensions)\n\n" + "\n\n".join(v[1] for v in valid)


def find_best_saver(savers, ext):
    for saver in savers:
        if saver.is_valid_extension(ext):
            return saver
    return None


def save_layer(layer, uri, saver=None):
    if uri is None:
        uri = layer.file_path

    name, ext = os.path.splitext(uri)

    if not saver:
        savers = []
        for loader in get_known_loaders():
            loader = loader()
            if loader.can_save_layer(layer):
                savers.append(loader)
        saver = find_best_saver(savers, ext)

    if not saver:
        valid = valid_save_formats(layer)
        if valid:
            return "The extension '%s' doesn't correspond to any format\nthat can save the '%s' layer type.\n\n%s" % (ext, layer.type, get_valid_string(valid))

    progress_log.info("TITLE=Saving %s" % uri)
    error = saver.save_layer(uri, layer)
    return error
