# Code Review: Bug Report

## Critical Bugs Found

### 1. CRITICAL: Missing downsample_factor in saved config
**Location**: `run_experiment.py` line 134-145
**Severity**: HIGH
**Impact**: evaluate_network.py will fail to reconstruct the correct configuration

**Problem**:
```python
config_dict = {
    "experiment_name": cfg.name,
    "target_classes": cfg.target_classes,
    "layer_config": cfg.layer_config,
    "encoding_mode": cfg.encoding_mode,
    "n_neurons": structure.n_neurons,
    "n_connections": structure.n_connections,
    "best_fitness": float(best_fitness),
    "ga_generations": cfg.num_generations,
    "ga_population": cfg.population_size,
    "random_seed": cfg.random_seed,
}
```

**Missing**:
- `downsample_factor` - Required for intensity_to_neuron mode
- `conv_weights_path` - Required for conv_feature_to_neuron mode
- Simulation parameters (sim_duration_ms, mnist_stim_duration_ms, etc.)
- Neuron parameters
- Connection probabilities

**Fix**: Add all necessary parameters to saved config:
```python
config_dict = {
    "experiment_name": cfg.name,
    "target_classes": cfg.target_classes,
    "layer_config": cfg.layer_config,
    "encoding_mode": cfg.encoding_mode,
    "downsample_factor": cfg.downsample_factor,  # CRITICAL
    "conv_weights_path": cfg.conv_weights_path,
    "n_neurons": structure.n_neurons,
    "n_connections": structure.n_connections,
    "best_fitness": float(best_fitness),
    "ga_generations": cfg.num_generations,
    "ga_population": cfg.population_size,
    "random_seed": cfg.random_seed,
    # Simulation parameters
    "sim_duration_ms": cfg.sim_duration_ms,
    "mnist_stim_duration_ms": cfg.mnist_stim_duration_ms,
    "sim_dt_ms": cfg.sim_dt_ms,
    "max_freq_hz": cfg.max_freq_hz,
    "stim_strength": cfg.stim_strength,
    # Network structure parameters
    "base_transmission_delay": cfg.base_transmission_delay,
    "inhibitory_fraction": cfg.inhibitory_fraction,
    "connection_probs": cfg.connection_probs,
    "neuron_params": cfg.neuron_params,
}
```

---

### 2. CRITICAL: evaluate_network.py cannot reconstruct config properly
**Location**: `evaluate_network.py` class TrainedNetwork, line ~40-60
**Severity**: HIGH
**Impact**: Will crash or use wrong parameters when evaluating

**Problem**: The TrainedNetwork._load_config() method tries to reconstruct config from incomplete saved JSON.

**Current approach**:
```python
def _load_config(self):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    self.cfg = ExperimentConfig()  # Creates with defaults
    self.cfg.target_classes = config_dict['target_classes']
    self.cfg.hidden_layers = config_dict['layer_config'][1:-1]
    self.cfg.encoding_mode = config_dict.get('encoding_mode', 'intensity_to_neuron')
    self.cfg.downsample_factor = config_dict.get('downsample_factor', 4)  # MISSING FROM SAVED FILE!
```

**Fix**: Either:
1. Fix run_experiment.py to save complete config (recommended)
2. Make evaluate_network.py more robust with better defaults

---

### 3. MEDIUM: Balanced accuracy assigns 1.0 to missing classes
**Location**: `mnist_ga/evaluation.py` line 101-103
**Severity**: MEDIUM
**Impact**: Could inflate fitness scores if some classes are never seen

**Problem**:
```python
if num_true_cls == 0:
    # Class not present in this batch - assign neutral score
    class_recalls[cls_label] = 1.0  # <-- QUESTIONABLE
```

**Analysis**:
- If a class is not in the evaluation batch, it gets perfect recall (1.0)
- This could inflate the balanced accuracy
- Better approach: Only average over classes that are actually present

**Recommended Fix**:
```python
# Calculate balanced accuracy only over present classes
present_classes = [cls for cls in range(n_classes) if np.sum(true_labels == cls) > 0]
if not present_classes:
    return 0.0

class_recalls = []
for cls_label in present_classes:
    true_mask = (true_labels == cls_label)
    correct_cls_count = np.sum(predicted_labels[true_mask] == cls_label)
    class_recalls.append(correct_cls_count / np.sum(true_mask))

balanced_accuracy = np.mean(class_recalls)
```

---

### 4. LOW: Inefficient connection map lookup
**Location**: `mnist_ga/network_builder.py` line 285
**Severity**: LOW
**Impact**: Slower structure creation (O(n) lookups in list)

**Problem**:
```python
connection_exists = (i, j) in connection_map  # O(n) lookup
```

**Fix**: Use a set for O(1) lookups:
```python
connection_map_set = set(connection_map)
for i in range(output_start, output_end):
    for j in range(output_start, output_end):
        if i == j or j not in positions:
            continue

        connection_exists = (i, j) in connection_map_set  # O(1)
        # ... rest of code
```

---

## Logic Verification Results

### ✓ Network Builder (network_builder.py)
- Layer indexing: CORRECT
- Distance-dependent delays: CORRECT
- Connection probabilities: CORRECT
- Output mutual inhibition: CORRECT
- Position assignment: CORRECT

### ✓ Spike Encoding (encoding.py)
- MNIST data loading: CORRECT
- Train/test split: CORRECT
- Label mapping: CORRECT
- Spike precomputation: CORRECT (uses existing MNIST_utils)

### ✓ Simulation (simulation.py)
- MNIST spike timing: CORRECT
- Stimulation logic: CORRECT
- Activity recording: CORRECT
- Network update loop: CORRECT

### ✓ Prediction (evaluation.py)
- Winner-take-all logic: CORRECT
- Output layer identification: CORRECT
- Spike counting: CORRECT

### ⚠ Fitness Evaluation (evaluation.py)
- Network reset: CORRECT
- Weight setting: CORRECT
- Balanced accuracy: QUESTIONABLE (see bug #3)
- Error handling: GOOD

### ✗ Config Saving (run_experiment.py)
- **INCOMPLETE** - Missing critical parameters (see bug #1)

### ✗ Config Loading (evaluate_network.py)
- **FRAGILE** - Depends on incomplete saved config (see bug #2)

---

## Recommendations

### Immediate Fixes (Before Production Use):
1. **Fix saved config** - Add all necessary parameters
2. **Fix config loading** - Make evaluate_network.py robust
3. **Review balanced accuracy** - Decide on missing class handling

### Nice to Have:
4. Use set for connection_map lookups in mutual inhibition
5. Add config validation in ExperimentConfig
6. Add unit tests for critical functions

---

## Test Scenarios to Verify

1. **Round-trip test**: Train → Save → Load → Evaluate
   - Current status: Will FAIL due to missing config parameters

2. **Different encoding modes**:
   - intensity_to_neuron with downsample_factor=4: Should work
   - conv_feature_to_neuron: Will FAIL (missing conv_weights_path in saved config)

3. **Class imbalance**:
   - Balanced accuracy with missing classes: QUESTIONABLE behavior

4. **Config reconstruction**:
   - evaluate_network.py loading saved config: Will use DEFAULT values, not saved ones!

---

## Severity Summary

- **CRITICAL (2)**: Config save/load bugs - will cause evaluation to fail or use wrong parameters
- **MEDIUM (1)**: Balanced accuracy behavior with missing classes
- **LOW (1)**: Inefficient lookup (minor performance)

**OVERALL**: Code logic is mostly correct, but the config save/load pipeline is broken and will cause issues.
