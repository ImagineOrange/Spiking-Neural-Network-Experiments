# MNIST_utils/MNIST_stimulation_encodings.py

# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean # For downsampling images
try:
    from sklearn.datasets import fetch_openml # To download the MNIST dataset
except ImportError:
    print("Warning: scikit-learn not found. MNIST_loader might fail.")
    fetch_openml = None
import time # To measure execution time
import matplotlib.gridspec as gridspec # For creating complex subplot layouts
import os
import torch
import torch.nn as nn
# Import necessary torchvision transform if needed for ConvNet input
from torchvision import transforms

# --- MNIST Data Loading Class (Unchanged) ---
class MNIST_loader:
    """
    Handles loading and basic access to the MNIST dataset.
    Downloads the dataset if not available locally (via fetch_openml).
    """
    def __init__(self):
        """Loads the full MNIST dataset upon initialization."""
        if fetch_openml is None:
            raise ImportError("scikit-learn is required to load MNIST data. Please install it.")
        try:
            # Use as_frame=False to get NumPy arrays
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            # Data comes as flattened 784 vectors
            self.images = mnist.data.astype(np.float32) / 255.0 # Normalize
            self.labels = mnist.target.astype(np.int_) # Labels as integers
        except Exception as e:
            print(f"Error fetching MNIST via fetch_openml: {e}")
            print("Trying legacy parser...")
            try:
                # Fallback parser
                mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
                self.images = mnist.data.astype(np.float32) / 255.0 # Normalize
                self.labels = mnist.target.astype(np.int_) # Labels as integers
            except Exception as e_fallback:
                print(f"Fallback MNIST fetch also failed: {e_fallback}")
                raise RuntimeError("Could not download or load MNIST dataset.")


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

# --- Define the Convolutional Network (Copied from previous step) ---
# This needs to be defined here so SNNStimulator can use it.
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, output_features=49): # Default to 10 classes, 49 features
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential( # Feature extraction layer
            nn.Conv2d(32, 1, kernel_size=1), # Project to 1 channel -> 1x7x7
            nn.ReLU() # Apply ReLU to features
        )
        # Linear layers (not used by SNNStimulator directly, but part of the trained model)
        self.fc1 = nn.Linear(output_features, 128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # For the full network pass:
        # features = out.reshape(out.size(0), -1)
        # out = self.fc1(features)
        # out = self.relu_fc1(out)
        # out = self.fc2(out)
        # return out
        # For feature extraction only (used by SNNStimulator):
        return out # Return the output of layer3 (1x7x7)

    def extract_features(self, x):
        """Passes input through conv layers up to the feature layer."""
        with torch.no_grad(): # Ensure no gradients are calculated
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out) # Output of shape [batch, 1, 7, 7]
            # Flatten the output to get the feature vector
            features = out.reshape(out.size(0), -1) # Shape: [batch, 49]
        return features


# --- SNN Input Stimulator Class (MODIFIED) ---
class SNNStimulator:
    """
    Generates Poisson spike trains for SNN input (Diehl & Cook 2015 style).
    Supports two modes:
    1. 'intensity_to_neuron': Pixel intensity maps directly to firing rate.
    2. 'conv_feature_to_neuron': CNN feature activations map to firing rate.

    Spike times are generated using Poisson statistics for biological realism.
    """
    def __init__(self, total_time_ms=5.0, max_freq_hz=400.0,
                 mode='intensity_to_neuron', # New mode parameter
                 conv_weights_path='conv_model_weights/conv_model_weights.pth', # Path for new mode
                 device=None): # Device for CNN model
        """
        Initializes the stimulator parameters and mode-specific components.

        Args:
            total_time_ms (float): Total simulation duration in milliseconds.
            max_freq_hz (float): Max firing frequency (Hz) corresponding to max input value (1.0).
            mode (str): Operation mode ('intensity_to_neuron' or 'conv_feature_to_neuron').
            conv_weights_path (str): Path to the trained ConvNet weights file (used in conv mode).
            device (torch.device, optional): Device to run the ConvNet on. Defaults to CPU if None.
        """
        if total_time_ms <= 0 or max_freq_hz < 0:
            raise ValueError("Time parameter and frequency must be positive.")
        self.total_time_ms = total_time_ms
        self.max_freq_hz = max_freq_hz
        self.total_time_s = total_time_ms / 1000.0
        self.mode = mode

        # --- Mode-specific initialization ---
        self.feature_extractor = None
        self.device = device if device else torch.device("cpu") # Default to CPU

        if self.mode == 'conv_feature_to_neuron':
            print(f"SNNStimulator initializing in '{self.mode}' mode.")
            print(f"Attempting to load ConvNet weights from: {conv_weights_path}")
            if not os.path.exists(conv_weights_path):
                raise FileNotFoundError(f"ConvNet weights file not found at {conv_weights_path}. Please train the CNN first.")

            try:
                # Instantiate the ConvNet (assuming default 10 classes is fine, only feature part is used)
                model = ConvNet(num_classes=10, output_features=49).to(self.device)
                # Load the saved weights
                model.load_state_dict(torch.load(conv_weights_path, map_location=self.device, weights_only=True))
                model.eval() # Set to evaluation mode
                self.feature_extractor = model # Store the whole model, will call extract_features
                print(f"ConvNet model loaded successfully onto device: {self.device}")
            except Exception as e:
                print(f"Error loading or initializing ConvNet model: {e}")
                raise

        elif self.mode == 'intensity_to_neuron':
            print(f"SNNStimulator initializing in '{self.mode}' mode.")
        else:
            raise ValueError(f"Invalid mode specified: {self.mode}. Choose 'intensity_to_neuron' or 'conv_feature_to_neuron'.")

        # --- Image Transform for ConvNet ---
        # Define the same normalization used during CNN training
        self.cnn_transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _generate_spikes_from_values(self, values):
        """
        Internal helper to generate Poisson spikes based on firing rates.

        This implements Poisson spike generation like Diehl & Cook 2015,
        where spike times are randomly distributed according to the rate.
        """
        all_spike_times_ms = []
        for value in values:
            value = max(0, value) # Ensure value is non-negative
            neuron_spike_times = []

            if value <= 1e-6: # Threshold for effectively zero
                neuron_spike_times = [] # No spikes for zero/near-zero activation
            else:
                # Calculate firing rate
                rate_hz = value * self.max_freq_hz

                # Generate Poisson spike train
                # Expected number of spikes = rate * duration
                expected_spikes = rate_hz * self.total_time_s

                # Draw actual number from Poisson distribution
                num_spikes = np.random.poisson(expected_spikes)

                if num_spikes > 0:
                    # Generate random spike times uniformly distributed over the window
                    # (Poisson process = uniform random times given count)
                    spike_times = np.random.uniform(0, self.total_time_ms, num_spikes)
                    # Sort for temporal ordering
                    neuron_spike_times = sorted(spike_times.tolist())

            all_spike_times_ms.append(neuron_spike_times)
        return all_spike_times_ms

    def generate_spikes(self, image_example):
        """
        Generates spike times based on the configured mode.

        Args:
            image_example (np.ndarray): A 2D numpy array (e.g., 28x28 or 14x14)
                                       with **unnormalized** pixel values ([0, 255] or similar).
                                       Normalization happens internally if needed.

        Returns:
            list[list[float]]: A list where each inner list contains the spike times
                               (in milliseconds) for the corresponding SNN input neuron.
                               Length matches input (e.g., 784 or 49).
        """
        if image_example.ndim != 2:
            raise ValueError("Input image_example must be a 2D numpy array.")

        if self.mode == 'intensity_to_neuron':
            # --- Intensity Mode: Use pixel values directly ---
            # Normalize image to [0, 1] for rate calculation
            normalized_image = np.clip(image_example / 255.0, 0, 1) if image_example.max() > 1.0 else np.clip(image_example, 0, 1)
            pixel_intensities = normalized_image.flatten()
            return self._generate_spikes_from_values(pixel_intensities)

        elif self.mode == 'conv_feature_to_neuron':
            # --- Conv Feature Mode: Use CNN features ---
            if self.feature_extractor is None:
                raise RuntimeError("ConvNet model not loaded for 'conv_feature_to_neuron' mode.")

            # Preprocess image for the CNN:
            # 1. Ensure image is 28x28 (required by the ConvNet architecture)
            if image_example.shape != (28, 28):
                 # Attempt to resize if needed, or raise error if input size mismatch is critical
                 # For simplicity, assuming input 'image_example' is the original 28x28 here.
                 # If downsampling happened before calling this, it needs reconsideration.
                 # Let's assume the user provides the original 28x28 unnormalized image.
                 raise ValueError(f"Input image shape must be (28, 28) for 'conv_feature_to_neuron' mode, got {image_example.shape}")

            # 2. Convert to tensor, add channel and batch dimensions -> (1, 1, 28, 28)
            #    Normalize to [0, 1] first if it's [0, 255]
            image_tensor = torch.from_numpy(image_example.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

            # 3. Apply the specific normalization the CNN was trained with
            image_tensor = self.cnn_transform(image_tensor)

            # 4. Move tensor to the correct device
            image_tensor = image_tensor.to(self.device)

            # Extract features using the loaded model
            with torch.no_grad():
                # Get feature activations (shape: [1, 49])
                feature_activations_tensor = self.feature_extractor.extract_features(image_tensor)

            # Move features to CPU, convert to numpy, flatten
            feature_activations = feature_activations_tensor.squeeze().detach().cpu().numpy() # Shape: (49,)

            # Feature activations are already post-ReLU in the model, so >= 0.
            # Normalize features to [0, 1] for rate coding consistency.
            max_activation = feature_activations.max()
            if max_activation > 1e-6: # Avoid division by zero
                 normalized_features = feature_activations / max_activation
            else:
                 normalized_features = np.zeros_like(feature_activations)

            # Generate spikes based on these 49 normalized feature values
            return self._generate_spikes_from_values(normalized_features)
        else:
            # This case should not be reached due to __init__ check
            raise ValueError(f"Invalid mode '{self.mode}' encountered in generate_spikes.")
    
    def extract_feature_map(self, image_example):
        """
        Extracts the 7x7 feature map from the ConvNet for a given image.
        Only works if mode is 'conv_feature_to_neuron'.

        Args:
            image_example (np.ndarray): A 2D 28x28 numpy array (unnormalized 0-255).

        Returns:
            np.ndarray or None: The 7x7 feature map (as numpy array) or None if error/wrong mode.
        """
        if self.mode != 'conv_feature_to_neuron' or self.feature_extractor is None:
            print("Warning: Feature map extraction only available in 'conv_feature_to_neuron' mode with a loaded model.")
            return None

        if image_example.shape != (28, 28):
            print(f"Warning: Input image shape must be (28, 28) for feature map extraction, got {image_example.shape}")
            # Optionally add resizing logic here if needed, but for now return None
            return None

        try:
            # Preprocess image (same as in generate_spikes)
            image_tensor = torch.from_numpy(image_example.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            image_tensor = self.cnn_transform(image_tensor)
            image_tensor = image_tensor.to(self.device)

            # Extract features up to layer3 output (before flattening)
            with torch.no_grad():
                out = self.feature_extractor.layer1(image_tensor)
                out = self.feature_extractor.layer2(out)
                feature_map_tensor = self.feature_extractor.layer3(out) # Shape: [1, 1, 7, 7]

            # Move to CPU, remove batch/channel dims, convert to numpy
            feature_map_numpy = feature_map_tensor.squeeze().detach().cpu().numpy() # Shape: (7, 7)
            return feature_map_numpy

        except Exception as e:
            print(f"Error extracting feature map: {e}")
            return None


# Image Downsampling Function (if needed, copied from MNIST script)
def downsample_image(image_2d, factor):
    """ Downsamples a 2D image using local averaging. """
    if factor <= 1:
        return image_2d
    # Ensure input is float for downscale_local_mean
    image_2d_float = image_2d.astype(np.float32)
    downsampled = downscale_local_mean(image_2d_float, (factor, factor))
    return downsampled



# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":

    # --- Apply Dark Style ---
    try:
        plt.style.use('dark_background')
    except OSError:
        print("Warning: 'dark_background' style not found. Using default style.")

    # --- Configuration ---
    MODE = 'conv_feature_to_neuron' # <<< CHANGE MODE HERE ('intensity_to_neuron' or 'conv_feature_to_neuron')
    CONV_WEIGHTS = 'conv_model_weights/conv_model_weights.pth' # Make sure this file exists
    TOTAL_TIME = 100 # ms
    MAX_FREQ = 200 # Hz
    DOWNSAMPLE_FACTOR = 1 # Use 1 for conv mode (needs 28x28), can use >1 for intensity mode

    # --- Select Device for CNN ---
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device for CNN: {device}")

    # --- Load MNIST Example ---
    print("Getting MNIST example...")
    loader = MNIST_loader()
    example_index = np.random.randint(0, len(loader.images))
    # Get the original 28x28 image (assuming values 0-1 from loader)
    # The loader already reshapes and normalizes.
    original_image_0_1 = loader.get_image(example_index)
    original_label = loader.get_label(example_index)
    print(f"Using MNIST image index {example_index}, Label: {original_label}")

    # --- Prepare Image based on Mode ---
    if MODE == 'intensity_to_neuron':
        # Downsample if needed for intensity mode
        if DOWNSAMPLE_FACTOR > 1:
            image_to_stimulate = downsample_image(original_image_0_1 * 255.0, DOWNSAMPLE_FACTOR) # Pass 0-255
            print(f"Downsampled by {DOWNSAMPLE_FACTOR} for intensity mode -> {image_to_stimulate.shape}")
        else:
            image_to_stimulate = original_image_0_1 * 255.0 # Pass 0-255
            print(f"Using original image for intensity mode -> {image_to_stimulate.shape}")
        num_expected_neurons = image_to_stimulate.size
    elif MODE == 'conv_feature_to_neuron':
        # Conv mode requires the original 28x28 image (pass 0-255)
        image_to_stimulate = original_image_0_1 * 255.0 # Pass 0-255 to generate_spikes
        print(f"Using original 28x28 image for conv mode.")
        num_expected_neurons = 49 # Expect 49 feature outputs
    else:
        raise ValueError("Invalid MODE selected for main execution block.")


    # --- Initialize the Stimulator ---
    try:
        stimulator = SNNStimulator(
            total_time_ms=TOTAL_TIME,
            max_freq_hz=MAX_FREQ,
            mode=MODE,
            conv_weights_path=CONV_WEIGHTS,
            device=device
        )
    except Exception as e:
        print(f"Error initializing SNNStimulator: {e}")
        exit()

    # --- Generate the spike regime ---
    print(f"\nGenerating spike times using mode: '{MODE}'...")
    start_time = time.time()
    try:
        spike_times_list = stimulator.generate_spikes(image_to_stimulate) # Pass the prepared image
    except Exception as e:
        print(f"Error generating spikes: {e}")
        exit()
    end_time = time.time()
    print(f"Spike generation took: {end_time - start_time:.4f} seconds")

    # --- Analyze the output ---
    num_neurons_output = len(spike_times_list)
    total_spikes = sum(len(spikes) for spikes in spike_times_list)
    print(f"Generated spikes for {num_neurons_output} neurons (Expected: ~{num_expected_neurons})")
    print(f"Total spikes generated: {total_spikes}")
    if num_neurons_output != num_expected_neurons:
         print(f"Warning: Output neuron count ({num_neurons_output}) doesn't match expected ({num_expected_neurons}) for mode '{MODE}'.")

    # --- Visualization ---
    print("Preparing visualization...")
    fig, ax_raster = plt.subplots(figsize=(12, 6), facecolor='#1a1a1a')

    # Plot Spike Raster
    color_map = plt.cm.viridis(np.linspace(0, 1, num_neurons_output)) if num_neurons_output > 1 else ['white']
    for neuron_idx in range(num_neurons_output):
        spikes = spike_times_list[neuron_idx]
        if spikes:
             ax_raster.vlines(spikes, neuron_idx + 0.5, neuron_idx + 1.5, color=color_map[neuron_idx], linewidth=1.5)

    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel(f"SNN Input Neuron Index ({MODE})")
    ax_raster.set_title(f"Generated Spikes (Mode: {MODE}, Label: {original_label}) - {total_spikes} spikes")
    ax_raster.set_xlim(0, stimulator.total_time_ms)
    ax_raster.set_ylim(0, num_neurons_output + 1)
    ax_raster.invert_yaxis() # Neuron 0 at top
    ax_raster.grid(axis='x', linestyle='--', alpha=0.6)
    ax_raster.set_facecolor('#1a1a1a')
    ax_raster.tick_params(colors='white')
    for spine in ax_raster.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.show()









'''
alter main function in GA...

# Relevant section from MNIST_GA_experiment.py (mental model based on previous file content)

# --- Configuration ---
# ... (other configs)
# ADD A MODE CONFIGURATION HERE:
ENCODING_MODE = 'intensity_to_neuron' # Or 'conv_feature_to_neuron'
CONV_WEIGHTS_PATH = 'conv_model_weights/conv_model_weights.pth' # Needed if using conv mode

# --- Dependency Check ---
if ENCODING_MODE == 'conv_feature_to_neuron':
    if not os.path.exists(CONV_WEIGHTS_PATH):
        print(f"FATAL ERROR: Encoding mode is '{ENCODING_MODE}' but ConvNet weights file not found at:")
        print(f"'{CONV_WEIGHTS_PATH}'")
        print("Please train the CNN first using the appropriate script.")
        exit()
    else:
        print(f"ConvNet weights file found at: {CONV_WEIGHTS_PATH}")


# ...

# --- Load and Filter MNIST Data ---
# ... (loads filtered_images_global)

# --- Instantiate Stimulator (BEFORE precomputation) ---
# This instantiation needs to be mode-aware
print(f"Initializing SNN Stimulator in '{ENCODING_MODE}' mode...")
try:
    # Determine device for potential CNN use
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device for SNNStimulator (if needed): {device}")

    mnist_stimulator_precompute = SNNStimulator(
        total_time_ms=mnist_stim_duration_global,
        max_freq_hz=max_freq_hz_global,
        mode=ENCODING_MODE,                 # Pass the selected mode
        conv_weights_path=CONV_WEIGHTS_PATH, # Pass the weights path
        device=device                       # Pass the device
    )
except Exception as e:
    print(f"Error initializing SNNStimulator for precomputation: {e}. Exiting.")
    exit()

# +++ START PRECOMPUTATION +++
print("\n--- Precomputing MNIST Spike Trains ---")
precomputed_spike_trains_global = {} # Dictionary to store {index: spike_trains}
indices_to_precompute = np.arange(num_filtered)
precompute_start_time = time.time()
for idx in tqdm(indices_to_precompute, desc="Precomputing Spikes", ncols=80):
    try:
        mnist_original_image = filtered_images_global[idx].reshape(28,28) # Start with original 28x28

        # --- MODIFICATION NEEDED HERE ---
        # Prepare the image *specifically* for the chosen encoding mode
        if ENCODING_MODE == 'intensity_to_neuron':
            # Downsample if needed for intensity mode
            if downsample_factor_global > 1:
                # The stimulator expects 0-255 range input for intensity mode
                image_for_stimulator = downsample_image(mnist_original_image * 255.0, downsample_factor_global)
            else:
                image_for_stimulator = mnist_original_image * 255.0
        elif ENCODING_MODE == 'conv_feature_to_neuron':
            # Conv mode in stimulator expects the original 28x28 image (0-255 range)
            image_for_stimulator = mnist_original_image * 255.0
        else:
             raise ValueError(f"Invalid ENCODING_MODE: {ENCODING_MODE}")
        # --- END MODIFICATION ---

        # Generate and store spikes using the appropriately prepared image
        precomputed_spike_trains_global[idx] = mnist_stimulator_precompute.generate_spikes(image_for_stimulator) # Pass correct image

    except Exception as e:
        print(f"\nWarning: Error precomputing spikes for index {idx}: {e}. Skipping.")
        precomputed_spike_trains_global[idx] = None # Mark as failed

precompute_end_time = time.time()
print(f"Finished precomputing {len(precomputed_spike_trains_global)} spike trains in {precompute_end_time - precompute_start_time:.2f}s.")
# +++ END PRECOMPUTATION +++

# --- Prepare Fixed Arguments Tuple for Parallel Fitness Function ---
# Ensure precomputed_spike_trains_global is passed correctly here
# ...








will need to alter the config file saved, and also the eval script to read in:


# Inside the saving block:
config_to_save = {
    "encoding_mode_used": ENCODING_MODE, # Add this line
    "layers_config": layers_config_global,
    # ... other parameters ...
}



Clarity of downsample_factor_global:

Problem: The variable downsample_factor_global is only relevant when ENCODING_MODE is 
'intensity_to_neuron'. Its presence might be confusing when running in conv mode.
Suggestion: Rename it to something like DOWNSAMPLE_FACTOR_INTENSITY_MODE 
(as done in point 1) to make its specific use case clearer in the configuration section.






'''