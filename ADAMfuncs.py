import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
import itertools as it

def readmovie(fname: str) -> np.ndarray:
    """
    Reads a movie file and returns an array of frames, stored 
    as multidimensional numpy arrays (pixel matrices) in a 
    numpy array.

    Arguments:
        fname: str
        The name of the file you would like to parse. Must be an 
        appropriate file format (eg. *.avi or similar).

    Returns:
        frames: np.ndarray
        Array containing each frame from the input file. 
    """
    #Check for errors
    if isinstance(fname, str):
        vidcap = cv2.VideoCapture(fname)
        nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = float(vidcap.get(cv2.CAP_PROP_FPS))
        print(f"Image output will be {frame_h}*{frame_w} pixels")
    else:
        raise TypeError("fname is not a string")
        
    frames = np.fromiter((cv2.cvtColor(vidcap.read()[1], cv2.COLOR_BGR2GRAY) for i in range(nframes)), dtype = np.dtype(object, (nframes, frame_h, frame_w)), count = nframes)

    vidcap.release()
    #Check if parsing succeeded 
    if len(frames) == 0:
        raise ValueError("Parsing did not succeed and an empty array was returned")
    else:
        print("Done")
        return frames, fps
        
def gen_danger_matrix(framearr: np.ndarray, thresh: float, fps: float = 25, binfactor: int = 10) -> np.ndarray:
    """
    Generates a "danger area" from the input video where it takes a long time to 
    reach peak pixel intensity (our "ADAM" technique)

    Arguments:
        fname: str
        The name of the file you would like to parse. Must be an 
        appropriate file format (eg. *.avi or similar).

    Returns:
        frames: np.ndarray
        Array containing each frame from the input file. 
    """   
    if len(framearr) != 0:
        if framearr[-1].ndim > 1:
            print("Calculating mask")
            mask = np.zeros(np.shape(framearr[-1]))
            for i in range(len(framearr)):
                mask[(framearr[i] >= thresh) & (mask == 0)] = i/fps
        else:
            raise ValueError("Input frames are one dimensional, must be multidimensional")
    else:
        raise ValueError("frame array is empty")    
    
    if not np.any(mask):
        raise ValueError("Mask was not filled and is empty")
    else:
        print("Done")
        return mask 

        
def gen_danger_matrix_deriv(framearr: np.ndarray, fps: float = 25) -> np.ndarray:
    mask = []
    for i in range(0, len(framearr) - 1):
        mask.append((framearr[i+1] - framearr[i])/((1/fps)))
    
    mask = np.sum(mask, axis = 0)/len(mask)

    return mask

def gen_typical_distr(framearr: np.ndarray, mask: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray]:
    typ_pixels, typ_idx = np.unique(mask, return_counts = True)
    typ_idx = np.asarray(list(it.product(range(np.shape(mask)[1]), range(np.shape(mask)[0]))))[np.where(typ_idx == typ_idx.max())]

    larg_idx = np.where(mask == np.max(mask))
    
    typ_arr = np.empty(len(framearr))
    larg_arr = np.empty(len(framearr))
    
    typ_arr = [np.mean(framearr[i][typ_idx]) if (np.mean(framearr[i][typ_idx]) <= thresh) else thresh for i in range(len(framearr))]
    larg_arr = [np.mean(framearr[i][larg_idx]) if (np.mean(framearr[i][larg_idx]) <= thresh) else thresh for i in range(len(framearr))]
    
    return typ_arr, larg_arr
    
def gen_pixel_hist(mask: np.ndarray, nbins: int) -> tuple[np.array, np.array]:
    """
    Outputs the one dimensional distribution of times
    from the pixel mask.

    Arguments:
        mask: np.ndarray
        The previously generated danger-area matrix.

    Returns:
        hist, bins: tuple[np.array, np.array]
        The computed histogram and the corresponding bins. 
    """
    
    if mask.ndim > 1:
        pixel_cts: np.array = mask.flatten()
        bin_cts: np.array = np.linspace(0, np.amax(pixel_cts), nbins)
        hist: np.array
        bins: np.array
        hist, bins = np.histogram(pixel_cts, bins = bin_cts)
    else:
        raise ValueError("Mask is empty or one dimensional, histogram could not be generated")
        
    return hist, bins, pixel_cts, bin_cts
    
def avg_pixel_time(mask: np.ndarray) -> tuple[np.array, np.array]:
    pixel_time = np.mean(mask, axis = 1)
    pixel_nums = np.linspace(0, np.shape(pixel_time)[0], np.shape(pixel_time)[0])
    return pixel_nums, pixel_time