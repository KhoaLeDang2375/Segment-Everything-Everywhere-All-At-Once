"""Object-only, square image reference for CLIP/IP-Adapter."""
import numpy as np
from PIL import Image


def object_reference(image, mask):
    mask = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Mask tham chiếu rỗng.')
    left, top, right, bottom = x.min(), y.min(), x.max()+1, y.max()+1
    # Hide source context so image conditioning does not encourage duplicating it.
    foreground = np.where(mask[..., None], image, 127).astype(np.uint8)
    crop = Image.fromarray(foreground).crop((int(left), int(top), int(right), int(bottom)))
    side = max(crop.size)
    padding = max(4, round(side * .1))
    square = Image.new('RGB', (side+2*padding, side+2*padding), (127, 127, 127))
    square.paste(crop, ((square.width-crop.width)//2, (square.height-crop.height)//2))
    return square
