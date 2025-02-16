"""
    Converts RGB to Y'CbCr according to ITU-T T.871.
    https://www.itu.int/rec/T-REC-T.871
"""

import numpy as np
from PIL import Image

def RGB2YCbCr(img):
    """Converts RGB to Y'CbCr according to ITU-T T.871

    Args:
        img: 3-dimensional numpy array of RGB with shape (H, W, 3)
    
    Returns:
        3-dimensional numpy array of Y'CbCr with shape (H, W, 3) 
        channels
    """

    R = img[:,:,0] # Red colour channel
    G = img[:,:,1] # Green colour channel
    B = img[:,:,2] # Blue colour channel

    # Convert and clamp to 8 bits according to ITU-T T.871
    Y  = np.uint8(np.round(  0.299 * R + 0.587 * G + 0.144 * B)\
        .clip(0, 255))
    Cb = np.uint8(np.round((-0.299 * R - 0.587 * G + 0.866 * B) / 1.722 + 128)\
        .clip(0, 255))
    Cr = np.uint8(np.round(( 0.701 * R - 0.587 * G - 0.144 * B) / 1.402 + 128)\
        .clip(0, 255))

    # Put all channels into one 3-dimensional array
    return np.stack([Y, Cb, Cr], axis=2)

if __name__ == '__main__':
    # 3x5 image of random pixels
    __img = np.random.randint(0, 256, size=(5, 3, 3))
    Image.fromarray(np.uint8(__img), 'RGB').show()
    __im_rt = RGB2YCbCr(__img)
    Image.fromarray(__im_rt, "YCbCr").show()
