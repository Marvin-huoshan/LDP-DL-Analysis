#!/usr/bin/env python
"""
Parallel Dataset Generation CLI - Generate LDP attack detection training data.

Example:
    python generate_dataset_parallel.py --output dataset.csv
    python generate_dataset_parallel.py --output custom.csv --protocols OUE --epsilons 0.5 1.0 --experiments 3 --workers 8
"""

import argparse
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    build_normal_lists_from_mechanism_stochastic,
    build_support_list_1_OUE,
    build_support_list_1_OLH,
    build_support_list_1_OLH_Server,
    extract_user_level_features_diffstats_style,
)
from attacker_detector.data.generators.attacks import perturb_OUE_multi, HST_Server, HST_Users

def get_distribution_generator(dataset_type: str):
    generators = {
        'zipf': generate_zipf_dist,
        'emoji': generate_emoji_dist,
        'fire': generate_fire_dist,
    }
    if dataset_type not in generators:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
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



def build_tasks(args):
    configs = DATASET_CONFIGS_FULL if args.full_scale else DATASET_CONFIGS
    tasks = []
    config_count = 0

    for epsilon in args.epsilons:
        for dataset_type in args.datasets:
            dataset_config = configs[dataset_type]
            domain = args.domain if args.domain is not None else dataset_config['domain']
            n = args.n if args.n is not None else dataset_config['n']

            for protocol in args.protocols:
                for ratio in args.ratios:
                    for target_size in args.target_sizes:
                        for splits in args.splits:
                            config_count += 1
                            for exp_i in range(args.experiments):
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

                                tasks.append({
                                    "epsilon": epsilon,
                                    "dataset_type": dataset_type,
                                    "domain": domain,
                                    "n": n,
                                    "protocol": base_protocol,
                                    "ratio": ratio,
                                    "target_size": target_size,
                                    "splits": splits,
                                    "exp_i": exp_i,
                                    "seed": seed,
                                    "inner_processors": args.inner_processors,
                                    "config_id": config_count,
                                    "olh_setting": olh_setting
                                })
    return tasks

def run_one_task(task):
    """
    Worker function for one independent experiment.
    Returns a pandas DataFrame chunk and metadata.
    """
    try:
        features, labels = generate_user_level_dataset(
            epsilon=task["epsilon"],
            domain=task["domain"],
            n=task["n"],
            protocol=task["protocol"],
            ratio=task["ratio"],
            target_set_size=task["target_size"],
            splits=task["splits"],
            dataset_type=task["dataset_type"],
            h_ao=1,
            seed=task["seed"],
            inner_processors=task["inner_processors"],
        )

        num_users = len(labels)

        df_features = pd.DataFrame(features, columns=DATASET_FEATURE_NAMES)
        df_features["target_set_size"] = task["target_size"]
        df_features["attacker_ratio"] = task["ratio"]
        df_features["protocol"] = task["protocol"]
        df_features["splits"] = task["splits"]
        df_features["epsilon"] = task["epsilon"]
        df_features["dataset_type"] = task["dataset_type"]
        df_features["label"] = labels

        meta = {
            "ok": True,
            "num_users": num_users,
            "num_attackers": int(labels.sum()),
            "desc": (
                f'cfg={task["config_id"]}, exp={task["exp_i"] + 1}, '
                f'eps={task["epsilon"]}, data={task["dataset_type"]}, '
                f'protocol={task["protocol"]}, ratio={task["ratio"]}, '
                f'target={task["target_size"]}, splits={task["splits"]}'
            ),
            "df": df_features,
        }
        return meta

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "desc": (
                f'cfg={task["config_id"]}, exp={task["exp_i"] + 1}, '
                f'eps={task["epsilon"]}, data={task["dataset_type"]}, '
                f'protocol={task["protocol"]}, ratio={task["ratio"]}, '
                f'target={task["target_size"]}, splits={task["splits"]}'
            ),
        }


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
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if os.path.exists(args.output) and not args.append:
        print(f"Output file already exists, removing: {args.output}")
        os.remove(args.output)

    tasks = build_tasks(args)

    total_configs = (
        len(args.epsilons) *
        len(args.datasets) *
        len(args.protocols) *
        len(args.ratios) *
        len(args.target_sizes) *
        len(args.splits)
    )
    total_runs = len(tasks)

    print("=" * 80)
    print("LDP Attack Detection Dataset Generator (Parallel)")
    print("=" * 80)
    print(f"Output: {args.output}")
    print(f"Protocols: {args.protocols}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Datasets: {args.datasets}")
    print(f"Ratios: {args.ratios}")
    print(f"Target sizes: {args.target_sizes}")
    print(f"Splits: {args.splits}")
    print(f"Experiments per config: {args.experiments}")
    print(f"Total configurations: {total_configs}")
    print(f"Total experiment runs: {total_runs}")
    print(f"Outer workers: {args.workers}")
    print(f"Inner processors per task: {args.inner_processors}")
    print("=" * 80)

    total_users = 0
    total_attackers = 0
    num_success = 0
    num_failed = 0
    wrote_header = False

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one_task, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            result = future.result()

            if result["ok"]:
                df_chunk = result["df"]
                write_header_now = not wrote_header and not (args.append and os.path.exists(args.output) and os.path.getsize(args.output) > 0)

                df_chunk.to_csv(
                    args.output,
                    mode='a',
                    header=write_header_now,
                    index=False
                )
                wrote_header = True

                total_users += result["num_users"]
                total_attackers += result["num_attackers"]
                num_success += 1

                print(
                    f'[DONE] {result["desc"]} | '
                    f'users={result["num_users"]}, attackers={result["num_attackers"]}'
                )
            else:
                num_failed += 1
                print(f'[FAIL] {result["desc"]} | error={result["error"]}')
                print(result["traceback"])

    print("\n" + "=" * 80)
    print("Dataset Generation Complete")
    print("=" * 80)
    print(f"Successful runs: {num_success}")
    print(f"Failed runs: {num_failed}")
    print(f"Total users written: {total_users:,}")
    print(f"Total attackers written: {total_attackers:,}")

    if total_users > 0:
        benign = total_users - total_attackers
        print(f"Benign: {benign:,} ({100.0 * benign / total_users:.2f}%)")
        print(f"Attackers: {total_attackers:,} ({100.0 * total_attackers / total_users:.2f}%)")
        print(f"Saved to: {args.output}")
        print(f"File size: {os.path.getsize(args.output) / (1024 * 1024):.2f} MB")
    else:
        print("ERROR: No data generated.")
        sys.exit(1)


if __name__ == '__main__':
    main()
