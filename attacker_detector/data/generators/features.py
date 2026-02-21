"""DiffStats-style feature extraction for user-level attacker detection."""

import math
import numpy as np
from scipy.stats import binom, wasserstein_distance
from scipy.spatial.distance import jensenshannon


def extract_user_level_features_diffstats_style(support_list, ideal_support_list,
                                                  one_list, ideal_one_list,
                                                  epsilon, protocol, domain, n):
    """
    Extract features following DiffStats methodology:
    1. Detect anomalous k-values at distribution level
    2. Flag users with those k-values
    3. Differentiate attackers from benign users with same k-value
    """

    # Setup protocol parameters
    if protocol == 'OUE':
        p = 1 / 2
        q = 1 / (math.exp(epsilon) + 1)
        expected_ones = p + (domain - 1) * q
        p_binomial = (1 / domain) * (p + (domain - 1) * q)
    elif protocol == 'OLH':
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1 / g
        expected_ones = p + (domain - 1) * q
        p_binomial = (1 / domain) * (p + (domain - 1) * q)
    elif protocol in ('HST_User', 'HST_Server'):
        p_binomial = 1 / 2
        expected_ones = domain / 2
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    k_values, k_counts = np.unique(one_list, return_counts=True)
    k_values = k_values.astype(int)
    observed_freq = k_counts / len(one_list)

    theoretical_freq = np.array([binom.pmf(k, domain, p_binomial) for k in k_values])

    k_discrepancies = np.abs(observed_freq - theoretical_freq)

    k_to_discrepancy = dict(zip(k_values, k_discrepancies))
    k_to_observed_freq = dict(zip(k_values, observed_freq))
    k_to_theoretical_freq = dict(zip(k_values, theoretical_freq))

    # Calculate item probabilities from ideal distribution
    item_probabilities = np.mean(ideal_support_list, axis=0)  # (domain,)
    item_probabilities = item_probabilities / (np.sum(item_probabilities) + 1e-10)

    # Sort items by probability
    sorted_item_indices = np.argsort(item_probabilities)[::-1]

    features_list = []

    for i in range(n):
        user_features = []

        num_ones = one_list[i]
        k_int = int(num_ones)

        # F1: The k-value itself
        user_features.append(num_ones)

        # F2: Discrepancy of this k-value from theoretical distribution
        k_discrepancy = k_to_discrepancy.get(k_int, 0)
        user_features.append(k_discrepancy)

        # F3: Observed frequency of this k-value (how common is it?)
        k_obs_freq = k_to_observed_freq.get(k_int, 0)
        user_features.append(k_obs_freq)

        # F4: Theoretical frequency of this k-value
        k_theo_freq = k_to_theoretical_freq.get(k_int, 0)
        user_features.append(k_theo_freq)

        # F5: Ratio of observed to theoretical frequency
        freq_ratio = k_obs_freq / (k_theo_freq + 1e-10)
        user_features.append(freq_ratio)

        # F6: Is this k-value in the top-N most discrepant values?
        top_n_discrepant = 10
        discrepancy_threshold = sorted(k_discrepancies, reverse=True)[min(top_n_discrepant-1, len(k_discrepancies)-1)]
        is_anomalous_k = 1 if k_discrepancy >= discrepancy_threshold else 0
        user_features.append(is_anomalous_k)

        user_reported_items = np.where(support_list[i] > 0)[0]

        item_counts = np.sum(support_list, axis=0)
        expected_item_counts = n * item_probabilities
        item_frequency_ratio = item_counts / (expected_item_counts + 1e-10)

        anomaly_threshold = 1.5  # Items appearing 50% more than expected
        anomalous_items = set(np.where(item_frequency_ratio > anomaly_threshold)[0])

        # F7: Overlap with anomalous items (count)
        overlap_anomalous_items = len(set(user_reported_items) & anomalous_items)
        user_features.append(overlap_anomalous_items)

        # F8: Overlap ratio with anomalous items
        overlap_anomalous_ratio = overlap_anomalous_items / (len(user_reported_items) + 1e-10)
        user_features.append(overlap_anomalous_ratio)

        # F9: Mean frequency ratio of reported items
        if len(user_reported_items) > 0:
            mean_freq_ratio = np.mean(item_frequency_ratio[user_reported_items])
        else:
            mean_freq_ratio = 0
        user_features.append(mean_freq_ratio)
        # F10: Max frequency ratio of reported items (most anomalous item)
        if len(user_reported_items) > 0:
            max_freq_ratio = np.max(item_frequency_ratio[user_reported_items])
        else:
            max_freq_ratio = 0
        user_features.append(max_freq_ratio)

        # F12: Support diversity (entropy of support distribution)
        support_probs = support_list[i] / (num_ones + 1e-10)
        support_probs = support_probs[support_probs > 0]
        if len(support_probs) > 0:
            entropy = -np.sum(support_probs * np.log(support_probs + 1e-10))
        else:
            entropy = 0
        user_features.append(entropy)

        # F13: Max support value (attackers may concentrate on fewer items)
        max_support = np.max(support_list[i])
        user_features.append(max_support)

        # F15: Theoretical probability of this k-value
        theoretical_prob_k = binom.pmf(k_int, domain, p_binomial)
        user_features.append(theoretical_prob_k)

        # F16: Log likelihood
        log_likelihood = np.log(theoretical_prob_k + 1e-10)
        user_features.append(log_likelihood)

        # F16: Deviation from expected number of ones Wasserstein Distance
        deviation = abs(num_ones - expected_ones)
        user_features.append(deviation)

        # Create user's empirical distribution (point mass at their k-value)
        user_k_observed = np.zeros(len(k_values))
        k_idx = np.where(k_values == k_int)[0]
        if len(k_idx) > 0:
            user_k_observed[k_idx[0]] = 1.0  # Point mass at user's k-value
        else:
            user_k_observed = np.zeros(1)
            user_k_observed[0] = 1.0

        user_theoretical_freq = theoretical_freq.copy()

        user_k_observed_norm = user_k_observed / (np.sum(user_k_observed) + 1e-10)
        user_theoretical_freq_norm = user_theoretical_freq / (np.sum(user_theoretical_freq) + 1e-10)

        # F18: Wasserstein distance
        wass_dist_k = wasserstein_distance(
            k_values, k_values,
            u_weights=user_k_observed_norm,
            v_weights=user_theoretical_freq_norm
        )
        user_features.append(wass_dist_k)

        # F19: JS divergence
        js_divergence_k = jensenshannon(user_k_observed_norm, user_theoretical_freq_norm)
        user_features.append(js_divergence_k)

        features_list.append(user_features)


    return np.array(features_list)
