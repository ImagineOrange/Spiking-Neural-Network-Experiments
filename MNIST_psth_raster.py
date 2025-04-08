import numpy as np
import matplotlib.pyplot as plt
import time
import random

# --- Assumed Imports from your project structure ---
# (Make sure MNIST_stimulation_encodings.py is accessible)
try:
    from MNIST_stimulation_encodings import MNIST_loader, SNNStimulator, downsample_image
    # Check if scikit-image is available for downsampling, otherwise downsample_image might be a dummy
    try:
        from skimage.transform import downscale_local_mean
    except ImportError:
        print("Warning: scikit-image not found. downsample_image might use a basic fallback.")

except ImportError as e:
    print(f"--- ERROR ---")
    print(f"Could not import necessary modules from MNIST_stimulation_encodings.py: {e}")
    print("Please ensure MNIST_stimulation_encodings.py is in your Python path or the same directory.")
    print("Using dummy functions/classes as placeholders - script will likely fail.")
    print("---------------")
    # Dummy placeholders if imports fail
    class MNIST_loader:
        def __init__(self):
            print("Using dummy MNIST_loader.")
            self.labels = np.array([0, 1] + list(np.random.randint(2, 10, 98)))
            self.images = np.random.rand(100, 784)
            np.random.shuffle(self.labels) # Mix 0 and 1
        def get_label(self, idx): return self.labels[idx] if idx < len(self.labels) else -1
        def get_image(self, idx): return self.images[idx].reshape(28,28) if idx < len(self.labels) else np.zeros((28,28))
        # get_data_size is replaced by len(self.labels) below

    class SNNStimulator:
        def __init__(self, total_time_ms, max_freq_hz):
            print("Using dummy SNNStimulator.")
            self.total_time_ms = total_time_ms
            self.max_freq_hz = max_freq_hz
        def generate_spikes(self, image):
            print("Dummy SNNStimulator: Generating random spikes.")
            num_pixels = image.size
            return [list(np.sort(np.random.uniform(0, self.total_time_ms, np.random.randint(0,3)))) for _ in range(num_pixels)]

    def downsample_image(image, factor):
        print(f"Dummy downsample_image called with factor {factor}. Returning original.")
        # Basic fallback if skimage is not present in real import either
        if factor > 1:
             try:
                  # Try skimage's function if it was imported somehow
                  from skimage.transform import downscale_local_mean
                  return downscale_local_mean(image, (factor, factor))
             except ImportError:
                  # Very basic averaging if skimage is unavailable
                  M, N = image.shape
                  m, n = factor, factor
                  col_extent = N // n
                  row_extent = M // m
                  img = np.zeros((row_extent, col_extent))
                  for i in range(row_extent):
                      for j in range(col_extent):
                           img[i, j] = np.mean(image[i*m:(i+1)*m, j*n:(j+1)*n])
                  return img
        return image


# --- Configuration ---
DOWNSAMPLE_FACTOR = 4         # Downscale factor as requested (28x28 -> 7x7)
MNIST_STIM_DURATION_MS = 100  # Duration for generating MNIST spikes (ms)
MAX_FREQ_HZ = 200.0           # Max frequency for MNIST encoding (Hz)

# --- Main Script Logic ---
if __name__ == "__main__":
    start_time = time.time()
    plt.style.use('dark_background') # Use dark theme for plots

    # 1. Load MNIST Data
    print("Loading MNIST dataset...")
    try:
        mnist_loader = MNIST_loader()
        # FIX: Use len(mnist_loader.labels) instead of non-existent get_data_size()
        data_size = len(mnist_loader.labels)
        if data_size == 0:
             raise ValueError("MNIST loader returned empty dataset.")
    except Exception as e:
        print(f"Error loading MNIST: {e}. Exiting.")
        exit()

    # 2. Find Indices for Digit 0 and 1
    index_0, index_1 = -1, -1
    print("Searching for MNIST digits 0 and 1...")
    for i in range(data_size):
        label = mnist_loader.get_label(i)
        if label == 0 and index_0 == -1:
            index_0 = i
        elif label == 1 and index_1 == -1:
            index_1 = i
        if index_0 != -1 and index_1 != -1:
            break # Stop once both are found
    if index_0 == -1 or index_1 == -1:
        print("Error: Could not find both digit 0 and 1 in the loaded dataset.")
        exit()
    print(f"Using Digit 0 (Index: {index_0}), Digit 1 (Index: {index_1})")

    # 3. Initialize Stimulator
    mnist_stimulator = SNNStimulator(
        total_time_ms=MNIST_STIM_DURATION_MS,
        max_freq_hz=MAX_FREQ_HZ
    )

    processed_data = {} # To store results

    # 4. Process Digit 0 and Digit 1
    for digit, index in [('0', index_0), ('1', index_1)]:
        print(f"\n--- Processing Digit {digit} ---")
        original_image = mnist_loader.get_image(index)

        # Downsample the image
        print(f"Downsampling image by factor {DOWNSAMPLE_FACTOR}...")
        downsampled_image = downsample_image(original_image, DOWNSAMPLE_FACTOR)
        img_h, img_w = downsampled_image.shape
        num_input_neurons = img_h * img_w
        print(f"Downsampled shape: {img_h}x{img_w} ({num_input_neurons} pixels/neurons)")

        # Generate spikes
        print("Generating spikes...")
        spike_times_list = mnist_stimulator.generate_spikes(downsampled_image)
        total_spikes = sum(len(spikes) for spikes in spike_times_list)
        print(f"Generated {total_spikes} total spikes for digit {digit}.")

        # Store results
        processed_data[digit] = {
            'image': downsampled_image,
            'spikes': spike_times_list,
            'num_neurons': num_input_neurons
        }

    # 5. Plotting the Rasters
    print("\nGenerating raster plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, facecolor='#1a1a1a')
    fig.suptitle(f'Input Spike Rasters (Downsampled {DOWNSAMPLE_FACTOR}x)', fontsize=16, color='white')

    for i, digit in enumerate(['0', '1']):
        ax = axes[i]
        data = processed_data[digit]
        spike_times_list = data['spikes']
        num_neurons = data['num_neurons']

        # Prepare data for scatter plot
        raster_times = []
        raster_neurons = []
        for neuron_idx, spikes in enumerate(spike_times_list):
            for spike_time in spikes:
                raster_times.append(spike_time)
                # Plot neuron index (0 to num_neurons-1)
                raster_neurons.append(neuron_idx)

        # Create raster plot using scatter
        if raster_times:
            ax.scatter(raster_times, raster_neurons, s=3, color='white', alpha=0.8, marker='|')

        # Style the subplot
        ax.set_facecolor('#1a1a1a')
        ax.set_title(f"Digit '{digit}' Input", color='white', fontsize=12)
        ax.set_xlabel("Time (ms)", color='white')
        if i == 0: # Only add y-label to the first plot
            ax.set_ylabel("Input Neuron Index (Pixel)", color='white')
        ax.set_xlim(0, MNIST_STIM_DURATION_MS)
        ax.set_ylim(-0.5, num_neurons - 0.5)
        ax.invert_yaxis() # Match image orientation (pixel 0 at top)
        ax.tick_params(axis='both', colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555555')
        ax.grid(True, axis='x', linestyle=':', color='gray', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    save_path = f"input_rasters_0_vs_1_downsample_{DOWNSAMPLE_FACTOR}x.png"
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')
    print(f"Saved raster comparison plot to {save_path}")
    plt.show()

    print(f"\nScript finished in {time.time() - start_time:.2f} seconds.")