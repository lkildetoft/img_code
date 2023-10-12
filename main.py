from gen_mask import FluorescenceMasking
from ADAMfuncs import readmovie
import matplotlib.pyplot as plt
def main() -> None:
    mv, fps = readmovie("anim_abs_1.avi") 
    fmask: FluorescenceMasking = FluorescenceMasking(mv, fps, r"C:\Users\love\OneDrive - Lund University\Skrivbordet\img_code\ADAM_lib.dll")
    plt.imshow(fmask.mask)
    plt.show()
if __name__ == "__main__":
    main()