import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from wrapper import PYWrapper


class FluorescenceMasking:
    def __init__(self, mv: ArrayLike, fps: float, lib_path: str) -> None:
        self.wrapper: PYWrapper = PYWrapper(lib_path)
        self.mv: ArrayLike = mv
        self.fps: float = fps
        self.mask: NDArray

        self.mask = np.zeros(
                        np.shape(self.mv[-1]), dtype = np.float64, order = "C")
        self.wrapper.C_fillMask(self.mask, self.mv, len(self.mv), 
                                np.shape(self.mv[-1])[0], 
                                np.shape(self.mv[-1])[1], self.fps)

    def plot_mask(self) -> None:
        plt.imshow(self.mask)
        plt.colorbar("jet")
        plt.show()


