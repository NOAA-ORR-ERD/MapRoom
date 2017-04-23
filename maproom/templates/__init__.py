import os

from traits.trait_base import get_resource_path

from omnivore.utils.fileutil import get_latest_file


def get_template_path(name):
    path = get_resource_path(1)
    pathname = os.path.normpath("%s/%s" % (path, name))
    pathname = get_latest_file(pathname)
    return pathname


def get_template(name):
    pathname = get_template_path(name)
    if os.path.exists(pathname):
        with open(pathname, "rb") as fh:
            source = fh.read()
        return source
