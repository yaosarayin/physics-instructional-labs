import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

sigma_cut = 6.
# -------------------- Step 1: Load the Histogram Data --------------------

def load_histogram_data(file_path):
    """
    Loads histogram data from a TSV file.

    Parameters:
        file_path (str): Path to the TSV file.

    Returns:
        x (np.ndarray): Pixel positions.
        y (np.ndarray): Intensity values.
    """
    try:
        data = pd.read_csv(file_path, sep='\t')
        x = data['X'].values
        y = data['Y'].values
        return x, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# -------------------- Step 2: Define Gaussian Function --------------------

def gaussian(x, a, mu, sigma):
    """
    Gaussian function.

    Parameters:
        x (np.ndarray): Independent variable.
        a (float): Amplitude.
        mu (float): Mean.
        sigma (float): Standard deviation.

    Returns:
        np.ndarray: Gaussian function values.
    """
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# -------------------- Step 3: Detect Peaks --------------------

def detect_peaks(x, y, height_factor=0.1, distance=5):
    """
    Detects peaks in the histogram data.

    Parameters:
        x (np.ndarray): Pixel positions.
        y (np.ndarray): Intensity values.
        height_factor (float): Minimum height of peaks as a fraction of the maximum intensity.
        distance (int): Minimum number of pixels between peaks.

    Returns:
        peaks (np.ndarray): Indices of detected peaks.
    """
    peak_height = height_factor * np.max(y)
    peaks, properties = find_peaks(y, height=peak_height, distance=distance)
    return peaks

# -------------------- Step 4: Fit Gaussian to Each Peak --------------------

def fit_gaussians(x, y, peaks, window=10):
    """
    Fits Gaussian profiles to each detected peak.

    Parameters:
        x (np.ndarray): Pixel positions.
        y (np.ndarray): Intensity values.
        peaks (np.ndarray): Indices of detected peaks.
        window (int): Number of pixels to include on each side of the peak for fitting.

    Returns:
        fitted_params (list): List of fitted parameters [a, mu, sigma] for each peak.
    """
    fitted_params = []
    for peak in peaks:
        # Define window around the peak
        start = max(peak - window, 0)
        end = min(peak + window, len(x) - 1)
        x_fit = x[start:end]
        y_fit = y[start:end]

        # Initial guesses for a, mu, sigma
        a_initial = y[peak]
        mu_initial = x[peak]
        sigma_initial = 3 # Arbitrary initial guess

        try:
            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[a_initial, mu_initial, sigma_initial])
            # if popt[2]<sigma_cut:
            fitted_params.append(popt)
        except RuntimeError:
            print(f"Gaussian fit did not converge for peak at x={x[peak]:.2f}")
    return fitted_params

# -------------------- Step 5: Calculate Distances Between Peaks --------------------

def calculate_peak_distances(fitted_params):
    """
    Calculates distances between adjacent peaks based on their mean positions.

    Parameters:
        fitted_params (list): List of fitted parameters [a, mu, sigma] for each peak.

    Returns:
        peak_positions (list): Mean positions of the peaks.
        peak_distances (list): Distances between adjacent peaks.
    """
    peak_positions = sorted([params[1] for params in fitted_params])
    peak_distances = np.diff(peak_positions)
    return peak_positions, peak_distances

# -------------------- Step 6: Plotting --------------------

def plot_histogram_with_fits(x, y, fitted_params, peaks):
    """
    Plots the histogram data with Gaussian fits overlaid.

    Parameters:
        x (np.ndarray): Pixel positions.
        y (np.ndarray): Intensity values.
        fitted_params (list): List of fitted parameters [a, mu, sigma] for each peak.
        peaks (np.ndarray): Indices of detected peaks.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Histogram Data', color='blue')

    # Plot detected peaks
    plt.plot(x[peaks], y[peaks], "x", label='Detected Peaks', color='red', markersize=10)

    # Plot Gaussian fits
    for idx, (a, mu, sigma) in enumerate(fitted_params):
        if sigma <= sigma_cut: # sigma cut
            plt.plot(x, gaussian(x, a, mu, sigma), 
                label=f'Gaussian Fit {idx+1}: μ={mu:.2f}, σ={sigma:.2f}', linestyle='--')

    plt.xlabel('X (Pixels)')
    plt.ylabel('Intensity')
    plt.title('Histogram with Gaussian Fits')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------- Main Function --------------------

def main():
    # Specify the path to your TSV file
    # data_tsv = pd.read_csv('pi_old_setup/907A-p0.xls', sep='\t')

    file_path = 'pi_old_setup/907A-p0.xls'  # Replace with your actual file path

    # Load data
    x, y = load_histogram_data(file_path)
    if x is None or y is None:
        return

    # Detect peaks
    peaks = detect_peaks(x, y, height_factor=0.5, distance=10)
    print(f"Detected peaks at indices: {peaks}")
    print(f"Detected peak positions (X): {x[peaks]}")

    # Fit Gaussians
    fitted_params = fit_gaussians(x, y, peaks, window=10)
    print("\nFitted Gaussian Parameters:")
    for idx, (a, mu, sigma) in enumerate(fitted_params, 1):
        print(f"Peak {idx}: Amplitude={a:.2f}, Mean={mu:.2f}, Std Dev={sigma:.2f}")

    # # Calculate distances between peaks
    # peak_positions, peak_distances = calculate_peak_distances(fitted_params)
    # print("\nPeak Positions (Mean of Gaussian):", peak_positions)
    # print("Distances Between Adjacent Peaks (Pixels):", peak_distances)

    # Plot the results
    plot_histogram_with_fits(x, y, fitted_params, peaks)

    # Optionally, save the results to a file
    results_df = pd.DataFrame({
        'Amplitude': [params[0] for params in fitted_params],
        'Mean (Pixels)': [params[1] for params in fitted_params],
        'Std Dev (Pixels)': [params[2] for params in fitted_params]
    })
    results_df['Distance to Previous Peak (Pixels)'] = np.insert(peak_distances, 0, np.nan)
    results_df.to_csv('gaussian_fit_results.csv', index=False)
    print("\nGaussian fit results saved to 'gaussian_fit_results.csv'.")

if __name__ == "__main__":
    main()


# # Extract data
# x = data_tsv['X'].values
# y = data_tsv['Y'].values

# # Define a Gaussian function
# def gaussian(x, a, mu, sigma):
#     return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Arrays to store recorded points
x_coords = []
y_coords = []

# Callback function to handle mouse clicks
def onclick(event):
    if event.xdata is not None and event.ydata is not None:  # Ensure click is within bounds
        x_coords.append(event.xdata)
        y_coords.append(event.ydata)
        print(f"Recorded: X={event.xdata:.2f}, Y={event.ydata:.2f}")
        plt.scatter(event.xdata, event.ydata, c='red', s=50)  # Highlight clicked point
        plt.draw()


# Display the image
fig, ax = plt.subplots()
ax.plot(x, y, label='Data', color='blue')
ax.set_xlabel('X (Pixels)')
ax.set_ylabel('Intensity')
ax.set_title('Histogram Data')
ax.legend()
ax.grid(True)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Show the interactive plot
plt.show()

# Print the recorded coordinates after the window is closed
print("X coordinates:", x_coords)
print("Y coordinates:", y_coords)