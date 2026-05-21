def histogram_equalization(img: np.ndarray) -> np.ndarray:
    channels = cv2.split(img)                                                                           # split the image into three 2d arrays for each of rgb values
    equalized = []                                                                                      # empty list where each channel will be stored
    for ch in channels:
        hist, _ = np.histogram(ch.flatten(), 256, [0, 256])                                             # stores the frequency count for each pixel
        cdf = hist.cumsum()                                                                             # converts the frequency of pixels into a running count 
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())                              # Shifts the entire CDF to start at 0 and then scales it to max value becoming 255 
        cdf_normalized = cdf_normalized.astype(np.uint8)                                                # converts the lookup table values to integers. No deciamls
        equalized.append(cdf_normalized[ch])                                                            # rempas each pixel to a new equalized intensity
    return cv2.merge(equalized)                                                                         # merge the channels back together

def transparency(img1: np.ndarray, img2: np.ndarray, d: float = 0.5) -> np.ndarray:             
    A = img1.astype(np.float32)                                                                         # converts both images to floats for math
    B = img2.astype(np.float32)                                                                         # same as above
    result = d * A + (1.0 - d) * B                                                                      # blend the two images. d controls the ratio
    return np.clip(result, 0, 255).astype(np.uint8)                                                     # returns all values to range from 0-255

def enhance_contrast(rgb: np.ndarray, d: float = 0.5) -> np.ndarray:                            
    log_result    = logarithmic_correction(rgb)                                                         # runs log correction on the original image
    histeq_result = histogram_equalization(rgb)                                                         # runs histogram equalization on the original image
    combined      = transparency(log_result, histeq_result, d=d)                                        # belnds the two together using transparency function 
    return combined                                                                                 

def compute_mse(reference: np.ndarray, processed: np.ndarray) -> float:                         
    diff = reference.astype(np.float64) - processed.astype(np.float64)                                  # subtracts every pixel in the processed image fromt he corresponding pixel in reference. Cast to 64 bit float  
    return float(np.mean(diff ** 2))                                                                    # Squares every difference, then takes average across all pixels 
def compute_psnr(reference: np.ndarray, processed: np.ndarray, max_val: float = 255.0) -> float:         
    mse = compute_mse(reference, processed)                                                             # reuse MSE from earlier
    if mse == 0:                                                                                        # if the images are 0, MSE is 0. dividing by 0 is undefined so infinity is returned instead
        return float("inf")
    return float(10 * np.log10((max_val ** 2) / mse))                                                   # applies PNSR formula. 255 squared / by the MSE, converted to decibels with log10
def compute_ssim(reference: np.ndarray, processed: np.ndarray) -> float:                                
    return float(ssim(reference, processed, channel_axis=2, data_range=255))                            # scikit-learn's built in SSIM. Tells it the color channels are on the last axis of the array. 
