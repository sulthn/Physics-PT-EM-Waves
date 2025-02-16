"""
    Chroma Subsampling Examples.

    Converts an RGB image to the following:
    - 4:4:4 # Nothing changes its just in Y'CbCr
    - 4:2:2
    - 4:2:0
    - 4:1:1
    - 4:1:0
"""

from PIL import Image
from numpy import newaxis 
import numpy as np
import logging

from time import sleep

import RGB2YCbCr

logger = logging.getLogger(__name__)

class SamplingError(Exception):
    """Raised when "a" or "b" is bigger than 4
    """
    pass

def convert(img, a: int, b: int):
    """Converts a Y'CbCr image in a 3D numpy array to one of the sampling
    schematics:
    J:a:b
    4:4:4
    4:2:2
    4:2:0
    4:1:1
    4:1:0

    Args:
        img: 3-dimensional numpy array with shape (H, W, 3) in Y'CbCr.
        a: number of chrominance samples in the first row of “J” pixels.
        b: number of chrominance samples in the second row of “J” pixels.
    
    Returns:
        A tuple consisting of the size in bytes of the image before upsampling
        and a converted 3-dimensional numpy array with shape (H, W, 3)
    """
    
    # A couple checks...
    if a > 4 or b > 4:
        logger.error("This function assumes that the maximum horizontal "\
                     "sample size is 4.")
        raise(SamplingError)
    
    if a <= 0 or b < 0:
        logger.error("\"a\" cannot be zero or negative or \"b\" less than "\
                     "zero.")
        raise(SamplingError)

    if a == 3 or b == 3:
        raise(SamplingError)
    
    converted_bytes = 0
    
    # Odd sized images create problems, round the nearest odd number to even.
    img_size = [img.shape[0], img.shape[1]]
    if img.shape[0] % 2 == 1:
        img_size[0] = img.shape[0] + 1
    if img.shape[1] % 2 == 1:
        img_size[1] = img.shape[1] + 1
    img_size = tuple(img_size)

    # Seperate the channels.
    Y     = img[:,:,0] # Y channel
    Cb_in = img[:,:,1] # Blue difference chroma channel
    Cr_in = img[:,:,2] # Red difference chroma channel

    # Divide factors for later.
    if a == 4: a_divide_factor = 1
    if a == 2: a_divide_factor = 2
    if a == 1: a_divide_factor = 4

    if b == 4: b_divide_factor = 1
    if b == 2: b_divide_factor = 2
    if b == 1: b_divide_factor = 4
    if b == 0: b_divide_factor = 0

    # Sample pixels for each divide_factor.
    # Cb
    Cb_a = Cb_in[::2, ::a_divide_factor]
    converted_bytes += Cb_a.size
    Cb_a = np.repeat(np.repeat(Cb_a, 1, axis=0), a_divide_factor, axis=1)
    if b_divide_factor != 0: # If "b" is zero.
        Cb_b = Cb_in[1::2, ::b_divide_factor]
        converted_bytes += Cb_b.size
        Cb_b = np.repeat(np.repeat(Cb_b, 1, axis=0), b_divide_factor, axis=1)

    # Cr
    Cr_a = Cr_in[::2, ::a_divide_factor]
    converted_bytes += Cr_a.size
    Cr_a = np.repeat(np.repeat(Cr_a, 1, axis=0), a_divide_factor, axis=1)
    if b_divide_factor != 0: # If "b" is zero.
        Cr_b = Cr_in[1::2, ::b_divide_factor]
        converted_bytes += Cr_b.size
        Cr_b = np.repeat(np.repeat(Cr_b, 1, axis=0), b_divide_factor, axis=1)
    
    converted_bytes += Y.size
    
    # Upsampling for visualization.
    # If "b" is bigger than zero then we have to interleave a and b samples.
    if b_divide_factor > 0:
        Cb = np.empty(img_size, dtype=Cb_a.dtype)
        Cb[0::2, :] = Cb_a
        Cb[1::2, :] = Cb_b

        Cr = np.empty(img_size, dtype=Cr_a.dtype)
        Cr[0::2, :] = Cr_a
        Cr[1::2, :] = Cr_b
    else: # Else copy "a" samples to the second line.
        Cb = np.repeat(np.repeat(Cb_a, 2, axis=0), 1, axis=1)
        Cr = np.repeat(np.repeat(Cr_a, 2, axis=0), 1, axis=1)

    # Put all channels into one 3-dimensional array
    return (converted_bytes, np.stack([Y, Cb, Cr], axis=2))

if __name__ == "__main__":
    __img = Image.open("images/example1.png").convert("RGB")
    # Change this (^) line for other images
    __dummy = np.array(__img) # Convert to numpy array
    #Image.fromarray(np.uint8(__dummy), "RGB").show()
    __converted = RGB2YCbCr.RGB2YCbCr(__dummy) # Convert to Y'CbCr

    print("Original byte size:", __dummy.size)

    # Extracted Luma component
    Y = convert(__converted, 4, 4)[1][:,:,0]
    Image.fromarray(np.pad(Y[:,:,newaxis], ((0,0), (0,0), (0,2)), mode="maximum"), "RGB").show()
    sleep(0.2) # needs a little delay inbetween images.

    # Chroma scheme J:a:b
    # 4:4:4
    converted = convert(__converted, 4, 4)
    Image.fromarray(converted[1], mode="YCbCr").show()
    print("4:4:4 byte size:", converted[0])
    sleep(0.2) # needs a little delay inbetween images.
    # 4:2:2
    converted = convert(__converted, 2, 2)
    Image.fromarray(converted[1], mode="YCbCr").show()
    print("4:2:2 byte size:", converted[0])
    sleep(0.2) # needs a little delay inbetween images.
    # 4:2:0
    converted = convert(__converted, 2, 0)
    Image.fromarray(converted[1], mode="YCbCr").show()
    print("4:2:0 byte size:", converted[0])
    sleep(0.2) # needs a little delay inbetween images.
    # 4:1:1
    converted = convert(__converted, 1, 1)
    Image.fromarray(converted[1], mode="YCbCr").show()
    print("4:1:1 byte size:", converted[0])
    sleep(0.2) # needs a little delay inbetween images.
    # 4:1:0
    converted = convert(__converted, 1, 0)
    Image.fromarray(converted[1], mode="YCbCr").show()
    print("4:1:0 byte size:", converted[0])
    sleep(0.2) # needs a little delay inbetween images.
