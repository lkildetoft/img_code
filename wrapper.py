from ctypes import c_double, c_int, CDLL
import os
from numpy.ctypeslib import ndpointer
import numpy as np

class PYWrapper:
    def __init__(self, lib_path: str) -> None:
        os.add_dll_directory(r"C:\msys64\mingw64\bin\libgomp-1.dll")
        os.add_dll_directory(r"C:\msys64\mingw64\bin")
        self.lib_path: str = lib_path
        self.function_wrapper: CDLL = CDLL(self.lib_path)
        self.C_fillMask = self.function_wrapper.fillMask

        C_MASK_PTR: ndpointer = ndpointer(
                                    dtype = np.float64, ndim = 2, flags = "C")
        C_MV_PTR: ndpointer = ndpointer(
                                    dtype = np.int32, ndim = 3, flags = "C")

        self.C_fillMask.argtypes: tuple[ndpointer, c_int, c_double] = (
                        C_MASK_PTR, C_MV_PTR, c_int, c_int, c_int, c_double)
        self.C_fillMask.restype = None
    