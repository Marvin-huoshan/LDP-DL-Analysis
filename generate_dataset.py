#!/usr/bin/env python
"""
Dataset Generation CLI - Generate LDP attack detection training data.

Usage:
    python generate_dataset.py --output dataset.csv
    python generate_dataset.py --output custom.csv --protocols OUE --epsilons 0.5 1.0 --experiments 3
"""

import argparse
import math
import os
import sys
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DEFAULT_EPSILONS,
    DEFAULT_RATIOS,
    DEFAULT_TARGET_SIZES,
    DEFAULT_SPLITS,
    DEFAULT_SEED,
    DATASET_CONFIGS,
    DATASET_CONFIGS_FULL,
    DATASET_FEATURE_NAMES,
    DATASET_CONFIG_COLUMNS,
)
from attacker_detector.data.generators import (
    generate_zipf_dist,
    generate_emoji_dist,
    generate_fire_dist,
    construct_omega,
    build_normal_lists_from_mechanism_stochastic,
    build_support_list_1_OUE,
    build_support_list_1_OLH,
    build_support_list_1_OLH_Server,
    extract_user_level_features_diffstats_style,
)
from attacker_detector.data.generators.attacks import perturb_OUE_multi, HST_Server, HST_Users


def get_distribution_generator(dataset_type: str):
    """Get distribution generator function by type."""
    generators = {
        'zipf': generate_zipf_dist,
        'emoji': generate_emoji_dist,
        'fire': generate_fire_dist,
    }
    return generators[dataset_type]


def generate_user_level_dataset(
    epsilon: float,
    domain: int,
    n: int,
    protocol: str,
    ratio: float,
    target_set_size: int,
    splits: int,
    dataset_type: str = 'zipf',
    h_ao: int = 1,
    seed: int = None,
    processors: int = 4,
    olh_setting: str = 'server'
) -> tuple:
    """
    Generate user-level dataset with features and labels.
    """
    if seed is not None:
        np.random.seed(seed)

    generator = get_distribution_generator(dataset_type)
    X, REAL_DIST = generator(n, domain, seed=seed)

    target_set = set(np.random.choice(domain, size=target_set_size, replace=False))

    base_mechanism = 'OLH' if protocol in ('OLH', 'OLH_User', 'OLH_Server') else protocol
    ideal_support_list, ideal_one_list, ideal_ESTIMATE_DIST, _ = \
        build_normal_lists_from_mechanism_stochastic(
            epsilon=epsilon,
            d=domain,
            n=n,
            mechanism=base_mechanism,
            seed=seed if seed else 42
        )

    # Generate attacked batch (contains both benign and attackers)
    if protocol == "OLH":
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        User_Seed = np.arange(n)
        Y = np.zeros(n)

        if olh_setting == 'user':
          support_list, one_list, ESTIMATE_DIST, _ = \
            build_support_list_1_OLH(
                domain, Y, n, User_Seed, ratio, g, target_set,
                p, splits, h_ao, epsilon, processor=processors
            )
        else:
          support_list, one_list, ESTIMATE_DIST, _ = \
                  build_support_list_1_OLH_Server(
                      domain, Y, n, User_Seed, ratio, g, target_set,
                      p, splits, h_ao, epsilon, processor=processors
                  )

    elif protocol == "OUE":
        Y_data = perturb_OUE_multi(
            X=X,
            epsilon=epsilon,
            domain=domain,
            n=n,
            target_set=target_set,
            ratio=ratio,
            h_ao=h_ao,
            splits=splits,
            num_processes=processors
        )

        support_list, one_list, ESTIMATE_DIST, _ = \
            build_support_list_1_OUE(Y_data, n, epsilon)


    elif protocol == "HST_User":
        support_list, one_list, ESTIMATE_DIST, _ = \
            HST_Users(
                X=X,
                ratio=ratio,
                domain=domain,
                epsilon=epsilon,
                n=n,
                target_set=target_set,
                h_ao=h_ao,
                splits=splits
            )

    elif protocol == "HST_Server":
        support_list, one_list, ESTIMATE_DIST, _ = \
            HST_Server(
                X=X,
                ratio=ratio,
                domain=domain,
                epsilon=epsilon,
                n=n,
                target_set=target_set,
                splits=splits
            )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    user_features = extract_user_level_features_diffstats_style(
        support_list=support_list,
        ideal_support_list=ideal_support_list,
        one_list=one_list,
        ideal_one_list=ideal_one_list,
        epsilon=epsilon,
        protocol=protocol,
        domain=domain,
        n=n
    )
    # First n*(1-ratio) users are benign (label=0)
    # Last n*ratio users are attackers (label=1)
    num_benign = int(n * (1 - ratio))
    user_labels = np.zeros(n)
    user_labels[num_benign:] = 1

    return user_features, user_labels


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate LDP attack detection training dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--protocols',
        nargs='+',
        default=['OUE', 'OLH'],
        choices=['OUE', 'OLH', 'OLH_User', 'OLH_Server', 'HST_User', 'HST_Server'],
        help='LDP protocols to use'
    )

    parser.add_argument(
        '--epsilons',
        nargs='+',
        type=float,
        default=DEFAULT_EPSILONS,
        help='Privacy parameters (epsilon values)'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['zipf', 'emoji', 'fire'],
        choices=['zipf', 'emoji', 'fire'],
        help='Dataset types to generate'
    )

    parser.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        default=DEFAULT_RATIOS,
        help='Attacker ratios'
    )

    parser.add_argument(
        '--target-sizes',
        nargs='+',
        type=int,
        default=DEFAULT_TARGET_SIZES,
        help='Target set sizes'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        type=int,
        default=DEFAULT_SPLITS,
        help='Split values'
    )

    parser.add_argument(
        '--experiments',
        type=int,
        default=5,
        help='Number of experiments per configuration'
    )

    parser.add_argument(
        '--full-scale',
        action='store_true',
        help='Use full-scale dataset sizes (100k+ users)'
    )

    parser.add_argument(
        '--n',
        type=int,
        default=None,
        help='Override number of users (applies to all datasets)'
    )

    parser.add_argument(
        '--domain',
        type=int,
        default=None,
        help='Override domain size (applies to all datasets)'
    )

    parser.add_argument(
        '--processors',
        type=int,
        default=4,
        help='Number of parallel processes'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    configs = DATASET_CONFIGS_FULL if args.full_scale else DATASET_CONFIGS

    total_configs = (
        len(args.epsilons) *
        len(args.datasets) *
        len(args.protocols) *
        len(args.ratios) *
        len(args.target_sizes) *
        len(args.splits)
    )

    print("=" * 70)
    print("LDP Attack Detection Dataset Generator")
    print("=" * 70)
    print(f"Output: {args.output}")
    print(f"Protocols: {args.protocols}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Datasets: {args.datasets}")
    print(f"Ratios: {args.ratios}")
    print(f"Target sizes: {args.target_sizes}")
    print(f"Splits: {args.splits}")
    print(f"Experiments per config: {args.experiments}")
    print(f"Total configurations: {total_configs}")
    print(f"Processors: {args.processors}")
    print("=" * 70)

    all_features = []
    all_labels = []
    config_count = 0

    for epsilon in args.epsilons:
        for dataset_type in args.datasets:
            dataset_config = configs[dataset_type]
            domain = args.domain if args.domain else dataset_config['domain']
            n = args.n if args.n else dataset_config['n']

            for protocol in args.protocols:
                for ratio in args.ratios:
                    for target_size in args.target_sizes:
                        for splits in args.splits:
                            config_count += 1
                            print(f"\n[{config_count}/{total_configs}] "
                                  f"ε={epsilon}, {dataset_type}, {protocol}, "
                                  f"ratio={ratio}, target={target_size}, splits={splits}")

                            for exp_i in range(args.experiments):
                                try:
                                    seed = args.seed + config_count * 1000 + exp_i

                                    if protocol == "OLH_User":
                                        base_protocol = "OLH"
                                        olh_setting = "user"
                                    elif protocol == "OLH_Server":
                                        base_protocol = "OLH"
                                        olh_setting = "server"
                                    else:
                                        base_protocol = protocol
                                        olh_setting = "server"

                                    features, labels = generate_user_level_dataset(
                                        epsilon=epsilon,
                                        domain=domain,
                                        n=n,
                                        protocol=base_protocol,
                                        ratio=ratio,
                                        target_set_size=target_size,
                                        splits=splits,
                                        dataset_type=dataset_type,
                                        h_ao=1,
                                        seed=seed,
                                        processors=args.processors,
                                        olh_setting=olh_setting
                                    )

                                    # Add config columns
                                    num_users = len(labels)
                                    config_features = np.array([
                                        [target_size, ratio, protocol, splits, epsilon, dataset_type]
                                    ] * num_users)

                                    features_with_config = np.hstack((features, config_features))
                                    all_features.append(features_with_config)
                                    all_labels.append(labels)

                                    print(f"  Exp {exp_i + 1}: {num_users} users, "
                                          f"{int(labels.sum())} attackers")

                                except Exception as e:
                                    print(f"  Error in exp {exp_i + 1}: {e}")
                                    traceback.print_exc()
                                    continue

    if len(all_features) == 0:
        print("ERROR: No data generated!")
        sys.exit(1)

    X = np.vstack(all_features)
    y = np.hstack(all_labels)

    print("\n" + "=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print(f"Total users: {len(y):,}")
    print(f"Attackers: {int(y.sum()):,} ({y.sum()/len(y)*100:.1f}%)")
    print(f"Benign: {int(len(y)-y.sum()):,} ({(1-y.sum()/len(y))*100:.1f}%)")

    feature_names = DATASET_FEATURE_NAMES + DATASET_CONFIG_COLUMNS
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()
