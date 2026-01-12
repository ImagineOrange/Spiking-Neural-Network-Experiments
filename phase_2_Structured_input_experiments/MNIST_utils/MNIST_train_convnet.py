# MNIST_train_convnet_sklearn.py

import torch
import torch.nn as nn
import torch.optim as optim
# Removed torchvision datasets import
from torch.utils.data import DataLoader, TensorDataset # Keep DataLoader and add TensorDataset
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm # For progress bar
from sklearn.datasets import fetch_openml # Import fetch_openml

print("--- Starting Convolutional MNIST Training (using sklearn loader) ---")

# --- Configuration (Same as before) ---
NUM_CLASSES = 10 # Digits 0-9
OUTPUT_FEATURES = 49 # Features from conv layers
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10 # Adjust as needed
OUTPUT_DIR = "conv_model_weights"
PLOT_FILENAME = os.path.join(OUTPUT_DIR, "training_plot.png")
WEIGHTS_FILENAME = os.path.join(OUTPUT_DIR, "conv_model_weights.pth")
FIG_SIZE = (12, 7)

# --- 1. Device Selection (Same as before) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# --- 2. Define the Convolutional Network (Same as before) ---
class ConvNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, output_features=OUTPUT_FEATURES):
        super(ConvNet, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # Input: 1x28x28 -> Output: 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 16x14x14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Input: 16x14x14 -> Output: 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32x7x7
        )
        # Flattening layer - note 32 channels * 7 * 7 = 1568 features, not 49.
        # To get exactly 49 features, we need a different architecture or a projection layer.
        # Let's add a Conv layer to reduce channels to 1 before flattening to get 1x7x7 = 49 features.
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1), # Project 32 channels to 1 -> Output: 1x7x7
            nn.ReLU()
        )
        # Linear layers
        self.fc1 = nn.Linear(output_features, 128) # Input 49 features (1*7*7)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # Output layer matching N_CLASSES

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out) # Apply projection layer
        out = out.reshape(out.size(0), -1) # Flatten -> should be size [batch_size, 49]
        out = self.fc1(out)
        out = self.relu_fc1(out)
        out = self.fc2(out)
        return out

# --- 3. Load MNIST Data using sklearn ---
print("Loading MNIST dataset using fetch_openml...")
try:
    # Fetch data (might take time on first run)
    # Use as_frame=False to get NumPy arrays directly
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    # Data comes as flattened 784 vectors. Labels are strings.
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64) # Use int64 for CrossEntropyLoss

    # Normalize pixel values
    X /= 255.0

    # Reshape images to (N, 1, 28, 28) for Conv2d
    X = X.reshape(-1, 1, 28, 28)

    # Split into training (60k) and testing (10k) sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Convert to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data loaded and processed. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

except Exception as e:
    print(f"Error loading or processing MNIST data with fetch_openml: {e}")
    print("Please ensure scikit-learn is installed (`pip install scikit-learn`) and check your network connection.")
    exit() # Exit if data loading fails

# --- 4. Initialize Model, Loss, Optimizer (Same as before) ---
model = ConvNet(num_classes=NUM_CLASSES, output_features=OUTPUT_FEATURES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("Model, Loss, and Optimizer initialized.")
print("Model Architecture:")
print(model)

# --- 5. Training Loop (Same as before) ---
print(f"--- Starting Training for {EPOCHS} Epochs ---")
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    for i, (images, labels) in enumerate(train_loop):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Update tqdm progress bar
        train_loop.set_postfix(loss=loss.item(), acc=100. * correct_train / total_train)

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100. * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # --- 6. Evaluation Loop (Same as before) ---
    model.eval() # Set model to evaluation mode
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    test_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Test ]", leave=False)
    with torch.no_grad():
        for images, labels in test_loop:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            test_loop.set_postfix(loss=loss.item(), acc=100. * correct_test / total_test)

    epoch_test_loss = test_loss / len(test_loader)
    epoch_test_acc = 100. * correct_test / total_test
    test_losses.append(epoch_test_loss)
    test_accuracies.append(epoch_test_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
          f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

print("--- Training Finished ---")

# --- 7. Create Output Directory (Same as before) ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# --- 8. Plotting (Same as before) ---
print("Plotting training results...")
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=FIG_SIZE, facecolor='#1a1a1a')

# Plot Loss
color = 'tab:red'
ax1.set_xlabel('Epoch', color='white')
ax1.set_ylabel('Loss', color=color)
ax1.plot(range(1, EPOCHS + 1), train_losses, color=color, linestyle='-', marker='o', label='Training Loss')
ax1.plot(range(1, EPOCHS + 1), test_losses, color=color, linestyle='--', marker='x', label='Test Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors=color)
ax1.grid(True, alpha=0.3, axis='y')

# Instantiate a second axes that shares the same x-axis for Accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(range(1, EPOCHS + 1), train_accuracies, color=color, linestyle='-', marker='o', label='Training Accuracy')
ax2.plot(range(1, EPOCHS + 1), test_accuracies, color=color, linestyle='--', marker='x', label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(axis='y', colors=color)
ax2.set_ylim(0, 105) # Accuracy limits

fig.suptitle('CNN Model Training History', color='white', fontsize=16)
ax1.set_facecolor('#1a1a1a')
ax2.set_facecolor('#1a1a1a')
ax1.spines['left'].set_color(color)
ax2.spines['right'].set_color(color)
ax1.spines['bottom'].set_color('white')
ax1.spines['top'].set_color('white')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', framealpha=0.7)

fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# Save plot
try:
    plt.savefig(PLOT_FILENAME, dpi=150, facecolor='#1a1a1a')
    print(f"Saved training plot to {PLOT_FILENAME}")
except Exception as e:
    print(f"Error saving plot: {e}")
# plt.show() # Optionally display the plot
plt.close(fig)

# --- 9. Saving Model Weights (Same as before) ---
try:
    torch.save(model.state_dict(), WEIGHTS_FILENAME)
    print(f"Saved model weights to {WEIGHTS_FILENAME}")
except Exception as e:
    print(f"Error saving model weights: {e}")

print("--- Script Finished ---")