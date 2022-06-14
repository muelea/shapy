from typing import Union, List

import numpy as np

import PIL.Image as pil_img


class ImageList:
    def __init__(self, images: List):
        assert isinstance(images, (list, tuple))

    def to_tensor(self):
        pass
