import numpy as np
import jpeg4py as jpeg
import PIL.Image as pil_img

from loguru import logger

import cv2
import PIL.ExifTags


def read_img(img_fn, dtype=np.float32):
    if img_fn.endswith('jpeg') or img_fn.endswith('jpg'):
        try:
            with open(img_fn, 'rb') as f:
                img = jpeg.JPEG(f).decode()
            exif_raw_dict = pil_img.open(img_fn)._getexif()
            if exif_raw_dict is not None:
                exif_data = {
                    PIL.ExifTags.TAGS[k]: v
                    for k, v in exif_raw_dict.items()
                    if k in PIL.ExifTags.TAGS
                }
                orientation = exif_data.get('Orientation', None)
                if orientation is not None:
                    if orientation == 1 or orientation == 0:
                        # Normal image - nothing to do!
                        pass
                    elif orientation == 2:
                        # Mirrored left to right
                        img = np.fliplr(img)
                    elif orientation == 3:
                        # Rotated 180 degrees
                        img = np.rot90(img, k=2)
                    elif orientation == 4:
                        # Mirrored top to bottom
                        img = np.fliplr(np.rot90(img, k=2))
                    elif orientation == 5:
                        # Mirrored along top-left diagonal
                        img = np.fliplr(np.rot90(img, axes=(1, 0)))
                    elif orientation == 6:
                        # Rotated 90 degrees
                        img = np.rot90(img, axes=(1, 0))
                    elif orientation == 7:
                        # Mirrored along top-right diagonal
                        img = np.fliplr(np.rot90(img))
                    elif orientation == 8:
                        # Rotated 270 degrees
                        img = np.rot90(img)
                    else:
                        raise NotImplementedError
        except jpeg.JPEGRuntimeError:
            logger.warning('{} produced a JPEGRuntimeError', img_fn)
            img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
        except SyntaxError:
            pass
    else:
        #  elif img_fn.endswith('png') or img_fn.endswith('JPG') or img_fn.endswith(''):
        img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    if dtype == np.float32:
        if img.dtype == np.uint8:
            img = img.astype(dtype) / 255.0
            img = np.clip(img, 0, 1)
    return img
