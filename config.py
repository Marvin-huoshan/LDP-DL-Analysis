"""Configuration constants for attacker detection."""

# High importance features from feature importance analysis
HIGH_IMPORTANCE_FEATURES = [
    'mean_item_freq_ratio',
    'num_ones',
    'overlap_anomalous_items_count',
    'wasserstein_distance_k',
    'js_divergence_k',
    'max_item_freq_ratio',
    'std_item_freq_ratio'
]

# Training features - Statistical features only
TRAINING_FEATURES = HIGH_IMPORTANCE_FEATURES + [
    'k_theoretical_frequency',
    'k_observed_frequency',
    'log_likelihood'
]

# Context features for sensitivity analysis
CONTEXT_FEATURES = ['target_set_size', 'attacker_ratio', 'epsilon']

# Parameter display names for plots
PARAM_DISPLAY_MAP = {
    'epsilon': 'Epsilon ($\\epsilon$)',
    'attacker_ratio': 'Attacker Ratio ($\\beta$)',
    'target_set_size': 'Target Set Size ($r$)',
    'dataset_type': 'Dataset Type'
}

# Default training hyperparameters
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42

DEFAULT_EPSILONS = [0.5, 0.7, 1.0, 1.5]
DEFAULT_RATIOS = [0.10, 0.15, 0.20]
DEFAULT_TARGET_SIZES = [2, 4, 6, 8]
DEFAULT_SPLITS = [2, 4, 6, 8]

DATASET_CONFIGS = {
    'zipf':  {'domain': 1024, 'n': 5000},
    'emoji': {'domain': 1496, 'n': 5000},
    'fire':  {'domain': 296,  'n': 5000},
}

DATASET_CONFIGS_FULL = {
    'zipf':  {'domain': 1024, 'n': 100000},
    'emoji': {'domain': 1496, 'n': 218477},
    'fire':  {'domain': 296,  'n': 723090},
}

DATASET_FEATURE_NAMES = [
    'num_ones', 'k_discrepancy', 'k_observed_frequency',
    'k_theoretical_frequency', 'freq_ratio', 'is_anomalous_k',
    'overlap_anomalous_items_count', 'overlap_anomalous_items_ratio',
    'mean_item_freq_ratio', 'max_item_freq_ratio', 'std_item_freq_ratio',
    'support_entropy', 'max_support_value',
    'theoretical_probability_k', 'log_likelihood',
    'wasserstein_distance_k', 'js_divergence_k',
]

DATASET_CONFIG_COLUMNS = [
    'target_set_size', 'attacker_ratio', 'protocol',
    'splits', 'epsilon', 'dataset_type'
]

DATASET_TYPES = ['zipf', 'emoji', 'fire']

