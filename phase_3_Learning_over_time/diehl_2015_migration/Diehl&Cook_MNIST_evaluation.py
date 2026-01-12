'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import pickle
import matplotlib.pyplot as plt
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

# Initialize configuration
cfg = Config()

# Initialize data loader
data_loader = MNISTDataLoader(cfg)


# functions

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    # Determine which classes to iterate over based on config
    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    for j in classes_to_check:
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments

# Load parameters from config
MNIST_data_path = cfg.mnist_data_path
data_path = cfg.data_path + 'activity/'
n_e = cfg.n_e
n_input = cfg.n_input
ending = cfg.ending

# Evaluation settings - auto-detect from available files
# Look for training activity files (prioritize clean post-training activity)
import glob
training_files = glob.glob(data_path + 'resultPopVecs*.npy')
if not training_files:
    print(f"\nERROR: No training activity files found in {data_path}")
    print("Please run training mode first.")
    exit(1)

# Extract numbers from filenames and find the most recent
# Track both clean and regular files separately
clean_sizes = []
regular_sizes = []
for f in training_files:
    fname = f.split('/')[-1]
    if fname.startswith('resultPopVecs'):
        size_str = fname.replace('resultPopVecs', '').replace('.npy', '').replace('_clean', '')
        try:
            size = int(size_str)
            if '_clean' in fname:
                clean_sizes.append(size)
            else:
                regular_sizes.append(size)
        except ValueError:
            continue

# Prioritize clean files for training (assignments)
if clean_sizes:
    training_ending = str(max(clean_sizes)) + '_clean'
    is_clean = True
    print(f"\nUsing CLEAN post-training activity for assignments (STDP off, final weights)")
elif regular_sizes:
    training_ending = str(max(regular_sizes))
    is_clean = False
    print(f"\nUsing training activity for assignments (STDP was ON - may reduce accuracy)")
    print(f"   For best results, run a full training session to generate clean activity.")
else:
    print(f"\nERROR: Could not parse training file sizes")
    exit(1)

# For testing, use non-clean files (from test mode run)
if regular_sizes:
    testing_ending = str(max(regular_sizes))
    print(f"Using test activity for evaluation")
else:
    print(f"\nWARNING: No test activity files found!")
    print(f"   Please run test mode (set test_mode=True) to generate test activity.")
    print(f"   Attempting to use training activity for both...")
    testing_ending = training_ending

print(f"   Training file (for assignments): resultPopVecs{training_ending}.npy")
print(f"   Testing file (for evaluation): resultPopVecs{testing_ending}.npy")

start_time_training = 0
# Extract numeric part for end time (remove '_clean' suffix if present)
training_numeric_ending = int(training_ending.replace('_clean', ''))
end_time_training = training_numeric_ending
start_time_testing = 0
testing_numeric_ending = int(testing_ending.replace('_clean', ''))
end_time_testing = testing_numeric_ending

print('load MNIST')
print('Loading training data...')
training_images, training_labels = data_loader.load_training_data()
print('Loading test data...')
testing_images, testing_labels = data_loader.load_test_data()
print(f'Training set: {len(training_labels)} examples')
print(f'Test set: {len(testing_labels)} examples')

print('load results')
try:
    training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
    training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + '.npy')
    testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + '.npy')
    testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + '.npy')
    print(training_result_monitor.shape)
except FileNotFoundError as e:
    print(f"\nERROR: Could not find simulation results!")
    print(f"Missing file: {e.filename}")
    print(f"\nPlease run 'Diehl&Cook_spiking_MNIST.py' first to generate the activity files.")
    print(f"Expected files in '{data_path}':")
    print(f"  - resultPopVecs{training_ending}.npy")
    print(f"  - inputNumbers{training_ending}.npy")
    print(f"  - resultPopVecs{testing_ending}.npy")
    print(f"  - inputNumbers{testing_ending}.npy")
    exit(1)

print('get assignments')
test_results = np.zeros((10, end_time_testing-start_time_testing))
test_results_max = np.zeros((10, end_time_testing-start_time_testing))
test_results_top = np.zeros((10, end_time_testing-start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing-start_time_testing))

# Match sizes: activity buffer is circular, use last N labels
num_activity_examples = training_result_monitor.shape[0]
training_labels_subset = training_input_numbers[-num_activity_examples:]

assignments = get_new_assignments(training_result_monitor[start_time_training:end_time_training],
                                  training_labels_subset[start_time_training:end_time_training])
print(assignments)

# Calculate accuracy in chunks (max 10000 at a time to avoid memory issues)
chunk_size = 10000
num_tests = max(1, (end_time_testing + chunk_size - 1) // chunk_size)  # Ceiling division, at least 1
sum_accurracy = [0] * num_tests

print(f'\nCalculating accuracy on {end_time_testing} test examples in {num_tests} chunk(s)...')

counter = 0
while (counter < num_tests):
    start_time = min(chunk_size * counter, end_time_testing)
    end_time = min(chunk_size * (counter + 1), end_time_testing)

    if start_time >= end_time:
        break

    chunk_examples = end_time - start_time
    test_results = np.zeros((10, chunk_examples))

    print(f'\nChunk {counter+1}/{num_tests}: Processing examples {start_time} to {end_time}')

    for i in range(chunk_examples):
        test_results[:,i] = get_recognized_number_ranking(assignments,
                                                          testing_result_monitor[i+start_time,:])

    # testing_input_numbers contains the actual labels used during evaluation
    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct/float(chunk_examples) * 100
    print(f'  Accuracy: {sum_accurracy[counter]:.2f}% ({correct}/{chunk_examples} correct, {len(incorrect)} incorrect)')
    counter += 1

print(f'\n{"="*60}')
print(f'FINAL RESULTS')
print(f'{"="*60}')
print(f'Overall accuracy: {np.mean(sum_accurracy):.2f}% ± {np.std(sum_accurracy):.2f}%')
print(f'Total test examples: {end_time_testing}')
print(f'{"="*60}')


plt.show()
