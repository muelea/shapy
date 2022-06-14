from fvcore.common.registry import Registry

BODY_HEAD_REGISTRY = Registry('BODY_HEAD_REGISTRY')
BODY_HEAD_REGISTRY.__doc__ = """
Registry for the body prediction heads, which predict a 3D head/face
from a single image.
"""
