import os

from peppy2.utils.runtime import get_all_subclasses


class Serializer(object):
    def __init__(self):
        self.serialized_commands = []
    
    def __str__(self):
        lines = []
        for cmd in self.serialized_commands:
            lines.append(str(cmd))
        return "\n".join(lines)
    
    def add(self, cmd):
        sc = SerializedCommand(cmd)
        self.serialized_commands.append(sc)


class MetadataConverter(object):
    stype = None
    
    @classmethod
    def to_string(cls, value):
        pass
    
    @classmethod
    def from_string(cls, text):
        pass

class FileMetadataConverter(MetadataConverter):
    stype = "file_metadata"
    
    @classmethod
    def to_string(cls, value):
        return "%s %s" % (value.uri, value.mime)
    
    @classmethod
    def from_string(cls, text):
        pass


def get_converters():
    s = get_all_subclasses(MetadataConverter)
    c = {}
    for cls in s:
        if cls.stype is not None:
            c[cls.stype] = cls
    return c


class SerializedCommand(object):
    converters = get_converters()
    
    def __init__(self, cmd):
        self.cmd_cls = cmd.__class__.__name__
        p = []
        for name, stype in [(n[0], n[1]) for n in cmd.serialize_order]:
            p.append((stype, getattr(cmd, name)))
        self.params = p

    def __str__(self):
        output = []
        for stype, value in self.params:
            print stype, value
            try:
                c = self.converters[stype]
                output.append(c.to_string(value))
            except KeyError:
                output.append(str(value))
        
        text = " ".join(output)
        return "%s %s" % (self.cmd_cls, text)
