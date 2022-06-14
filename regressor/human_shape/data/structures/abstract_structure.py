from abc import ABC, abstractmethod
from loguru import logger


class AbstractStructure(ABC):
    def __init__(self):
        super(AbstractStructure, self).__init__()
        self.extra_fields = {}

    def __del__(self):
        if hasattr(self, 'extra_fields'):
            self.extra_fields.clear()

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field, default=None):
        return self.extra_fields.get(field, default)

    def has_field(self, field):
        return field in self.extra_fields

    def delete_field(self, field):
        if field in self.extra_fields:
            del self.extra_fields[field]

    def shift(self, vector, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.shift(vector)
            self.add_field(k, v)
        self.add_field('motion_blur_shift', vector)
        return self

    def transpose(self, method):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            self.add_field(k, v)
        self.add_field('is_flipped', True)
        return self

    def normalize(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.normalize(*args, **kwargs)
        return self

    def rotate(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.rotate(*args, **kwargs)
        self.add_field('rot', kwargs.get('rot', 0))
        return self

    def crop(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(*args, **kwargs)
        return self

    def resize(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.resize(*args, **kwargs)
            self.add_field(k, v)
        return self

    def to_tensor(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)
            self.add_field(k, v)

    def to(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
        return self
