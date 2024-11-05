import numpy as np
import cv2
from skimage.measure import shannon_entropy as entropy
from sklearn.preprocessing import normalize as norm


class FourierTransformExtractor():
    def __init__(self):
        pass

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Ensure the image is in HWC format if it's in CHW
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
        if image.ndim == 3:
            image = self.convert_color_space(image)

        masked_image = self.apply_threshold_mask(image)
        masked_image = np.nan_to_num(masked_image)
        masked_image = norm(masked_image, np.float32)

        # Compute the Fourier transform
        f_transform = np.fft.fft2(masked_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Calculate mean and standard deviation of the magnitude spectrum
        mean_val = np.nanmean(magnitude_spectrum)
        std_val = np.nanstd(magnitude_spectrum)
        return np.array([mean_val, std_val])

    def get_feature_name(self) -> list:
        return [f"{self.name}_magnitude_mean", f"{self.name}_magnitude_std"]

def compute_low_high_frequency_energy(magnitude_spectrum, cutoff_ratio=0.1):
    rows, cols = np.array(magnitude_spectrum).shape
    crow, ccol = rows // 2, cols // 2  # Center of the spectrum
    cutoff_radius = int(min(rows, cols) * cutoff_ratio)

    # Create a mask for low frequencies
    low_freq_mask = np.zeros(magnitude_spectrum.shape, dtype=np.uint8)
    cv2.circle(low_freq_mask, (ccol, crow), cutoff_radius, 1, -1)

    low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask) / magnitude_spectrum.size
    high_freq_energy = np.sum(magnitude_spectrum * (1 - low_freq_mask)) / magnitude_spectrum.size

    return low_freq_energy, high_freq_energy

def compute_radial_mean_variance(magnitude_spectrum):
    rows, cols = np.array(magnitude_spectrum).shape
    crow, ccol = rows // 2, cols // 2
    max_radius = int(np.hypot(crow, ccol))

    radial_means = []
    radial_variances = []
    for r in range(max_radius):
        # Create a mask for the current radius
        y, x = np.ogrid[:rows, :cols]
        mask = (x - ccol) ** 2 + (y - crow) ** 2 <= r ** 2

        # Calculate mean and variance within this radius
        ring_values = magnitude_spectrum[mask]
        radial_means.append(np.mean(ring_values))
        radial_variances.append(np.var(ring_values))

    return radial_means, radial_variances

def compute_magnitude_entropy(magnitude_spectrum):
    # Normalize the magnitude spectrum to form a probability distribution
    magnitude_spectrum_norm = magnitude_spectrum / np.sum(magnitude_spectrum)
    magnitude_entropy = entropy(magnitude_spectrum_norm.flatten())
    return magnitude_entropy

def compute_log_magnitude_stats(magnitude_spectrum):
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)  # Use log(1 + magnitude) to handle zero values
    log_mean = np.mean(log_magnitude_spectrum)
    log_variance = np.var(log_magnitude_spectrum)
    return log_mean, log_variance

class FFTExtractor():
    def __init__(self, cutoff_ratio=0.1):
        self.cutoff_ratio = cutoff_ratio
        self.radial_means_length = None
        self.radial_variances_length = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Ensure the image is in HWC format if it's in CHW
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Fourier transform
        f_transform = np.fft.fft2(im_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Extract features
        features = []

        # 1. Low and High Frequency Energy
        low_freq_energy, high_freq_energy = compute_low_high_frequency_energy(magnitude_spectrum, self.cutoff_ratio)
        features.extend([low_freq_energy, high_freq_energy])

        # 2. Radial Mean and Variance
        radial_means, radial_variances = compute_radial_mean_variance(magnitude_spectrum)
        features.extend(radial_means + radial_variances)

        # Store lengths for feature naming
        self.radial_means_length = len(radial_means)
        self.radial_variances_length = len(radial_variances)

        # 3. Entropy of Magnitude Spectrum
        magnitude_entropy = compute_magnitude_entropy(magnitude_spectrum)
        features.append(magnitude_entropy)

        # 4. Log-Magnitude Spectrum Mean and Variance
        log_mean, log_variance = compute_log_magnitude_stats(magnitude_spectrum)
        features.extend([log_mean, log_variance])

        return np.array(features)

    