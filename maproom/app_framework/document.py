import os
import types
import io as BytesIO
import uuid

import numpy as np
import jsonpickle

# Enthought library imports.
from .events import EventHandler
from .utils.command import UndoStack
from .utils import jsonutil
from .utils.nputil import to_numpy
from .utils.pyutil import get_plugins
from .persistence import get_template
from . import filesystem
from .filesystem import fsopen as open
from . import errors

import logging
log = logging.getLogger(__name__)


def find_document_class_for_file(file_metadata):
    """Find the "best" document for a given MIME type string.

    First attempts all documents with exact matches for the MIME string,
    and if no exact matches are found, returns through the list to find
    one that can edit that class of MIME.
    """
    all_documents = get_plugins('maproom.app_framework.documents', MafDocument)
    log.debug(f"find_document_class_for_file: file={file_metadata}, known documents: {all_documents}")
    matching_documents = [document for document in all_documents if document.can_load_file_exact(file_metadata)]
    log.debug(f"find_document_class_for_file: exact matches: {matching_documents}")

    if matching_documents:
        return matching_documents[0]

    # Try generic matches if all else fails
    for document in all_documents:
        if document.can_load_file_generic(file_metadata):
            return document
    raise errors.UnsupportedFileType(f"No document available for {file_metadata}")


def identify_document(file_metadata):
    doc_cls = find_document_class_for_file(file_metadata)
    return doc_cls(file_metadata)


class MafDocument:
    # Class properties

    json_expand_keywords = {}

    def __init__(self, file_metadata):
        self.undo_stack = UndoStack()
        self.extra_metadata = {}
        self.load(file_metadata)
        self.uuid = str(uuid.uuid4())
        self.change_count = 0
        self.permute = None
        self.baseline_document = None

        # events
        self.refresh_event = EventHandler(self)
        self.recalc_event = EventHandler(self)
        self.structure_changed_event = EventHandler(self)
        self.byte_values_changed_event = EventHandler(self)  # and possibly style, but size of array remains unchanged

        self.byte_style_changed_event = EventHandler(self)  # only styling info may have changed, not any of the data byte values

    def load(self, file_metadata):
        if file_metadata is None:
            self.create_empty()
        else:
            self.file_metadata = file_metadata
            self.load_raw_data()

    def load_raw_data(self):
        raise RuntimeError("Implement in subclass!")

    def create_empty(self):
        self.file_metadata = {'uri': '', 'mime': "application/octet-stream"}

    @property
    def can_revert(self):
        return self.uri != ""

    @property
    def uri(self):
        return self.file_metadata['uri']

    @property
    def mime(self):
        return self.file_metadata['mime']

    @property
    def name(self):
        return os.path.basename(self.uri)

    @property
    def menu_name(self):
        if self.uri:
            return "%s (%s)" % (self.name, self.uri)
        return self.name

    @property
    def root_name(self):
        name, _ = os.path.splitext(self.name)
        return name

    @property
    def extension(self):
        _, ext = os.path.splitext(self.name)
        return ext

    @property
    def is_on_local_filesystem(self):
        try:
            self.filesystem_path()
        except OSError:
            return False
        return True

    def __str__(self):
        return f"Document: uuid={self.uuid}, mime={self.mime}, {self.uri}"

    @property
    def is_dirty(self):
        return self.undo_stack.is_dirty()

    def load_permute(self, editor):
        if self.permute:
            self.permute.load(self, editor)

    def filesystem_path(self):
        return filesystem.filesystem_path(self.uri)

    def save(self, uri=None):
        if uri is None:
            uri = self.uri
        if not self.verify_writeable_uri(uri):
            raise errors.ReadOnlyFilesystemError(uri)
        if self.verify_ok_to_save():
            self.save_raw_data(uri)
            self.file_metadata['uri'] = uri
            self.undo_stack.set_save_point()

    def verify_writeable_uri(self, uri):
        return filesystem.is_user_writeable_uri(uri)

    def verify_ok_to_save(self):
        return True

    def save_raw_data(self, uri):
        raise RuntimeError("Implement in subclass!")

    def save_adjacent(self, ext, data, mode="w"):
        if ext:
            path = self.filesystem_path() + ext
            with open(path, mode) as fh:
                fh.write(data)
        else:
            raise RuntimeError(f"Must specify non-blank extension to write a file adjacent to the data file")
        return path

    #### Cleanup functions

    def halt_background_processing(self):
        """Phase 1 of destruction: stop and cleanup any background processing
        before removing any attributes.

        UI elements may by refreshing on timers or in response to background
        processing events, so stop all of that before attempting any other
        attribute cleanup."""
        pass

    def prepare_destroy(self):
        """Phase 2 of destruction: actually destroy resources"""
        pass

    #### file identification

    @classmethod
    def can_load_file_exact(cls, file_metadata):
        return False

    @classmethod
    def can_load_file_generic(cls, file_metadata):
        return False

    @classmethod
    def can_load_file(cls, file_metadata):
        return cls.can_load_file_exact(file_metadata) or cls.can_load_file_generic(file_metadata)
