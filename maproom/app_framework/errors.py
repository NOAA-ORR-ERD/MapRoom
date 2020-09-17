import logging
log = logging.getLogger(__name__)


class MafError(RuntimeError):
    pass

class RecreateDynamicMenuBar(MafError):
    pass

class ProcessKeystrokeNormally(MafError):
    pass

class EditorNotFound(MafError):
    pass

class UnsupportedFileType(MafError):
    pass

class ProgressCancelError(MafError):
    pass

class DocumentError(MafError):
    pass

class MissingDocumentationError(MafError):
    pass

class ClipboardError(MafError):
    pass

class ReadOnlyFilesystemError(MafError):
    pass
