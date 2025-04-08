# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean # For downsampling images
from sklearn.datasets import fetch_openml # To download the MNIST dataset
import time # To measure execution time
import matplotlib.gridspec as gridspec # For creating complex subplot layouts

# --- MNIST Data Loading Class ---
class MNIST_loader:
    """
    Handles loading and basic access to the MNIST dataset.
    Downloads the dataset if not available locally (via fetch_openml).
    """
    def __init__(self):
        """Loads the full MNIST dataset upon initialization."""
        try:
            mnist = fetch_openml('mnist_784', version=1, parser='liac-arff')
        except Exception:
             mnist = fetch_openml('mnist_784', version=1)
        self.images = np.array(mnist.data.astype(np.float32) / 255.0) # Normalize
        self.labels = np.array(mnist.target.astype(np.int_)) # Labels as integers

    def get_image(self, index):
        """Returns a single original image (784 pixels) reshaped into a 28x28 numpy array."""
        if index < 0 or index >= len(self.images):
            raise IndexError("Index out of bounds.")
        return self.images[index].reshape(28, 28)

    def get_label(self, index):
        """Returns the integer label for the image at the specified index."""
        if index < 0 or index >= len(self.labels):
            raise IndexError("Index out of bounds.")
        return self.labels[index]

# --- SNN Input Stimulus Generator Class (MODIFIED) ---
class SNNStimulator:
    """
    Generates deterministic spike trains for SNN input based on image pixel intensity.
    Each pixel's intensity is treated as a rate, determining the *number* of
    evenly spaced spikes generated over the total simulation time.
    MODIFIED: Ensures pixels with intensity 0.0 generate exactly one spike,
              placed randomly within the first half of the simulation time.
    """
    def __init__(self, total_time_ms=5.0, max_freq_hz=400.0):
        """
        Initializes the stimulator parameters.

        Args:
            total_time_ms (float): Total simulation duration in milliseconds.
            max_freq_hz (float): Max firing frequency (Hz) corresponding to pixel intensity 1.0.
        """
        if total_time_ms <= 0 or max_freq_hz < 0:
            raise ValueError("Time parameter and frequency must be positive.")
        self.total_time_ms = total_time_ms
        self.max_freq_hz = max_freq_hz
        self.total_time_s = total_time_ms / 1000.0 # Convert to seconds

    def generate_spikes(self, image_example):
        """
        Generates spike times for each pixel based on its intensity.
        - Pixels with intensity > 0 get N evenly spaced spikes (N=round(rate*T)).
        - Pixels with intensity <= 0 get exactly one spike with random timing
          in the first half of the interval [0, T/2].

        Args:
            image_example (np.ndarray): A 2D numpy array with normalized pixel values ([0, 1]).

        Returns:
            list[list[float]]: A list where each inner list contains the spike times
                               (in milliseconds) for the corresponding flattened pixel/neuron.
        """
        if image_example.ndim != 2:
            raise ValueError("Input image_example must be a 2D numpy array.")
        image_example = np.clip(image_example, 0, 1) # Ensure range [0, 1]

        height, width = image_example.shape
        num_pixels = height * width
        pixel_intensities = image_example.flatten()

        all_spike_times_ms = []
        # Iterate through each pixel's intensity value
        for intensity in pixel_intensities:
            # Initialize list for the current neuron's spike times
            neuron_spike_times = []

            if intensity <= 0:
                # No spike is generated for zero or negative intensity pixels
                neuron_spike_times = []

            # --- MODIFICATION END ---
            else:
                # --- Original logic for intensity > 0 ---
                # Calculate the target firing rate
                rate_hz = intensity * self.max_freq_hz
                # Calculate the *expected* number of spikes in the total time window
                num_spikes_float = rate_hz * self.total_time_s
                # Round to the nearest integer number of spikes
                num_spikes_int = int(round(num_spikes_float))

                # Generate spikes based on num_spikes_int (will be >= 0)
                if num_spikes_int == 1:
                    # If intensity > 0 results in 1 spike, place it in the middle
                    # (Could also apply jitter here if desired, but keeping it simple)
                    neuron_spike_times = [self.total_time_ms / 2.0]
                elif num_spikes_int > 1:
                    # If more than one spike, distribute them evenly
                    interval = self.total_time_ms / num_spikes_int
                    start_time = interval / 2.0
                    end_time = self.total_time_ms - interval / 2.0
                    neuron_spike_times = np.linspace(start_time, end_time, num_spikes_int).tolist()
                # If num_spikes_int is 0 (low intensity rounds down), neuron_spike_times remains []
                # --- End Original logic ---

            # Append the generated spike times for the current neuron
            all_spike_times_ms.append(neuron_spike_times)

        return all_spike_times_ms


# Image Downsampling Function (if needed, copied from MNIST script)
def downsample_image(image_2d, factor):
    """ Downsamples a 2D image using local averaging. """
    if factor <= 1:
        return image_2d
    downsampled = downscale_local_mean(image_2d, (factor, factor))
    return downsampled



# --- Main Execution Block (Largely Unchanged) ---
if __name__ == "__main__":

    # --- Apply Dark Style ---
    try:
        plt.style.use('dark_background')
        # print("Applied 'dark_background' plot style.") # Suppressed
    except OSError:
        print("Warning: 'dark_background' style not found. Using default style.")
    # --- End Apply Dark Style ---

    # 1. Load or generate an example image
    print("Getting MNIST example...")
    loader = MNIST_loader()
    example_index = np.random.randint(0, len(loader.images))
    original_image = loader.get_image(example_index)
    original_label = loader.get_label(example_index)
    print(f"Using MNIST image index {example_index}, Label: {original_label}")

    # Optional: Downsample the image
    downsample_factor = 2
    if downsample_factor > 1:
        image_to_stimulate = downscale_local_mean(original_image, (downsample_factor, downsample_factor))
        print(f"Image downsampled by factor {downsample_factor} to shape: {image_to_stimulate.shape}")
    else:
        image_to_stimulate = original_image
        print(f"Using original image shape: {image_to_stimulate.shape}")

    # 2. Initialize the stimulator
    stimulator = SNNStimulator(
        total_time_ms=100,     # Use the same parameters as before
        max_freq_hz=200.0
    )

    # 3. Generate the spike regime (using the modified SNNStimulator)
    print("\nGenerating spike times...")
    start_time = time.time()
    spike_times_list = stimulator.generate_spikes(image_to_stimulate)
    end_time = time.time()
    print(f"Spike generation took: {end_time - start_time:.4f} seconds")

    # 4. Analyze the output
    num_neurons = len(spike_times_list)
    image_h, image_w = image_to_stimulate.shape
    total_spikes = sum(len(spikes) for spikes in spike_times_list)
    # Verify minimum spikes and check their timing
    zero_intensity_indices = np.where(image_to_stimulate.flatten() <= 0)[0]
    spikes_for_zero_intensity = [spike_times_list[i] for i in zero_intensity_indices]
    all_zeros_have_one_spike = all(len(spikes) == 1 for spikes in spikes_for_zero_intensity)
    spikes_in_first_half = all(spk[0] <= stimulator.total_time_ms / 2.0 for spk in spikes_for_zero_intensity if spk)

    print(f"Total spikes generated: {total_spikes}")
    print(f"Found {len(zero_intensity_indices)} pixels with intensity 0.")
    print(f"All zero-intensity pixels have exactly 1 spike: {all_zeros_have_one_spike}")
    if zero_intensity_indices.size > 0:
        print(f"All zero-intensity spikes occur in first half (<= {stimulator.total_time_ms / 2.0} ms): {spikes_in_first_half}")


    # --- 5. Visualization (Using GridSpec for layout) ---
    print("Preparing visualization...")
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])
    ax_img = fig.add_subplot(gs[0, 0])      # Top-left
    ax_hist = fig.add_subplot(gs[0, 1])     # Top-right
    ax_raster = fig.add_subplot(gs[1, :])   # Bottom row, spanning all columns

    # --- Plot 1: Input MNIST Image ---
    im = ax_img.imshow(image_to_stimulate, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax_img.set_title(f"Input Image (Label: {original_label})")
    ax_img.axis('off')

    # --- Plot 2: Histogram of Pixel Intensities ---
    pixel_data = image_to_stimulate.flatten()
    num_bins = min(30, image_h * image_w // 5) if image_h * image_w > 0 else 10
    ax_hist.hist(pixel_data, bins=num_bins, range=(0, 1), color='skyblue', alpha=0.8)
    ax_hist.set_title("Intensity Distribution")
    ax_hist.set_xlabel("Pixel Intensity")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
    ax_hist.set_xlim(0, 1)

    # --- Plot 3: Spike Raster Plot ---
    num_neurons_to_plot = num_neurons
    for neuron_idx in range(num_neurons_to_plot):
        spikes = spike_times_list[neuron_idx]
        if spikes:
             # Plot spike lines - ensure they are visible
             ax_raster.vlines(spikes, neuron_idx + 0.5, neuron_idx + 1.5, color='white', linewidth=2) # Increased linewidth

    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel("Neuron Index (Flattened Pixel)")
    ax_raster.set_title(f"Rate-Coded Spikes (Deterministic, Jittered Min Spike for I=0)\n({total_spikes} spikes)") # Updated title
    ax_raster.set_xlim(0, stimulator.total_time_ms)
    ax_raster.set_ylim(0, num_neurons_to_plot + 1)
    ax_raster.invert_yaxis() # Neuron 0 at top
    ax_raster.grid(axis='x', linestyle='--', alpha=0.6)

    # Add a visual marker for the halfway point where jitter occurs
    half_time = stimulator.total_time_ms / 2.0
    ax_raster.axvline(half_time, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'Jitter Half-time ({half_time:.1f} ms)')
    ax_raster.legend(loc='upper right', fontsize='small')


    # --- Final Figure Adjustments and Display ---
    plt.suptitle(f"MNIST Example, Intensity, and Spikes (Max Freq: {stimulator.max_freq_hz}Hz)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()