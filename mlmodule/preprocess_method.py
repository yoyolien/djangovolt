import matplotlib.pyplot as plt
import pywt
import pandas as pd
import numpy as np

class DWT_denoise:
    def __init__(self):
        pass
    
    """ Mean absolute deviation of a signal """
    def __madev(self,d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)
## using the wavelet of DWT to denoise the data
    def transform(self, x, wavelet='db4', level=1):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1/0.6745) * self.__madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')