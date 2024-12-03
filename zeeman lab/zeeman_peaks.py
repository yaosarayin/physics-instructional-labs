import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from glob import glob
import time

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

def detect_peaks(x, y, height_factor, distance, prominence=2):
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
    peaks, properties = find_peaks(y, height=peak_height, prominence=prominence, distance=distance)
    return peaks

# -------------------- Step 4: Fit Gaussian to Each Peak --------------------

def fit_gaussians(x, y, peaks, window=2):
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
        sigma_initial = 2 # Arbitrary initial guess

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


xcenters = []
ycenters = []

def plot_histogram_with_peaks(x, y, peaks, fitted_params, fig, ax, label="", title=""):

    # Plot detected peaks
    ax.plot(x, y, label=title)

    ax.plot(x[peaks], y[peaks], "x", label='Detected Peaks', color='red', markersize=10)

    # Callback function to handle mouse clicks
    # click_count = 0
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:  # Ensure click is within bounds
            xcenters.append(event.xdata)
            ycenters.append(event.ydata)
            print(f"Recorded: X={event.xdata:.2f}, Y={event.ydata:.2f}")
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    for idx, (a, mu, sigma) in enumerate(fitted_params):
        # if sigma <= sigma_cut: # sigma cut
        # if mu > event.xdata+10: #only plot the peaks that we care about
        ax.plot(x, gaussian(x, a, mu, sigma), linestyle='--')
                # label=f'{idx+1}: μ={mu:.2f}, σ={sigma:.2f}', linestyle='--')
            # plt.draw()

    ax.set_xlabel('X (Pixels)')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.legend(frameon=False, loc='upper left', ncol=1)
    ax.grid(True)
    fig.tight_layout()
    # ax.set_xlim(xcenters[-1]+10, max(x))
    fig.savefig(label,dpi=300)
    plt.show()

    



# -------------------- Main Function --------------------
plt.rcParams['font.size'] = 18 # Adjust as needed
plt.rcParams['legend.fontsize'] = 15 # Adjust as needed
plt.rcParams['font.family'] = 'serif' # Adjust as needed

def main():
    # Specify the path to your TSV file
    # data_tsv = pd.read_csv('pi_old_setup/907A-p0.xls', sep='\t')
    peaks_array = []
    amps_array = []
    sigma_array = []

    pangle_array = []

    window = 2 # for old setup perpendicular
    height_factor=0.5
    distance=1
    prominence=2

    # window = 8 # for new setup parallel
    # height_factor=0.7
    # distance=8
    # prominence=2

    for xls_file in glob('./pi_old_setup/*.xls'):
    # for xls_file in glob('./new_setup/*.xls'):
    # for xls_file in glob('./anomalous_zeeman/pi/*.xls'):

        fig, ax = plt.subplots(figsize=(12,6))

        print(xls_file)
        amps = xls_file.split('/')[-1].split('A')[0]
        pangle = xls_file.split('p')[-1].split('.')[0]
        # pangle = xls_file.split('p')[-1].split('-')[0]
        # file_path = 'pi_old_setup/907A-p0.xls'  # Replace with your actual file path

        # Load data
        x, y = load_histogram_data(xls_file)
        # if x is None or y is None:
        #     return

        # Detect peaks
        peaks = detect_peaks(x, y, height_factor=height_factor,distance=distance,prominence=prominence)
        print(f"Detected peaks at indices: {peaks}")
        print(f"Detected peak positions (X): {x[peaks]}")


        # Fit Gaussians
        fitted_peaks = []
        fitted_sigma = []
        fitted_params = fit_gaussians(x, y, peaks,window)
        print("\nFitted Gaussian Parameters:")
        for idx, (a, mu, sigma) in enumerate(fitted_params, 1):
            fitted_peaks.append(float(mu))
            fitted_sigma.append(float(sigma))
            print(f"Peak {idx}: Amplitude={a:.2f}, Mean={mu:.2f}, Std Dev={sigma:.2f}")


        # Plot the results
        plot_histogram_with_peaks(x, y, peaks, fitted_params, fig, ax, amps+'A-'+pangle+".png", str(float(amps)*0.01)+' Amps,'+pangle+" deg")

        # print([i for i in peaks ], xcenter)
        # print("#############")
        inds = [fitted_peaks.index(i) for i in fitted_peaks if i > xcenters[-1]+5]

        sigma_array.append(np.array(fitted_sigma)[inds].tolist())
        peaks_array.append(np.array(fitted_peaks)[inds].tolist())
        amps_array.append(amps)
        pangle_array.append(pangle)
        # print(xcenters,ycenters)
        print(peaks_array)

    # print("\nGaussian fit results saved to 'gaussian_fit_results.csv'.")
    # Optionally, save the results to a file
    results_df = pd.DataFrame({
        'peaks': peaks_array,
        'sigmas': sigma_array,
        "xcenter": xcenters,
        "amps": amps_array,
        "pangle": pangle_array,     
    })
    # results_df['Distance to Previous Peak (Pixels)'] = np.insert(peak_distances, 0, np.nan)
    results_df.to_csv('peaks.csv', index=False)

if __name__ == "__main__":
    main()