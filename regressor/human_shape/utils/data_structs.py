from dataclasses import make_dataclass, fields, field
from loguru import logger


class Struct(object):
    def __new__(cls, **kwargs):
        class_fields = []
        for key, val in kwargs.items():
            if isinstance(val, dict):
                class_fields.append([
                    key, type(val), field(
                        default_factory=dict if isinstance(val, dict) else None,
                    )]
                )
            else:
                class_fields.append(
                    [key, type(val), field(
                        default=val,
                        #  default_factory=dict if isinstance(val, dict) else None,
                    )]
                )

        object_type = make_dataclass(
            'Struct',
            class_fields,
            namespace={
                'keys': lambda self: [f.name for f in fields(self)],
            },
        )
        return object_type()
