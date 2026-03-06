"""Attack simulation functions for OUE and OLH protocols."""

import math
import random
from functools import partial
from multiprocessing import Pool
from typing import Set, Tuple

import numpy as np
import xxhash
from scipy import stats
from tqdm import tqdm

from .protocols import construct_omega

# Global worker state for multiprocessing
_worker_X = None
_worker_domain = None
_worker_q_OUE = None


def _init_worker(X, domain, q_OUE):
    """Initialize worker process with shared data."""
    global _worker_X, _worker_domain, _worker_q_OUE
    _worker_X = X
    _worker_domain = domain
    _worker_q_OUE = q_OUE


def _perturb_oue_process(args):
    """Worker function for OUE perturbation."""
    start, end, ratio, target_set, h_ao, splits, average_1_num_list = args
    
    n = _worker_X.shape[0]
    domain = _worker_domain
    q_OUE = _worker_q_OUE
    
    local_user_data = np.zeros((end - start, domain), dtype=int)
    h_ao_local = int(h_ao * 10)

    for idx, i in enumerate(range(start, end)):
        v = int(_worker_X[i])

        if i < n * (1 - ratio):
            # Benign user
            random_flip = (np.random.rand(domain) < q_OUE).astype(int)
            local_user_data[idx, :] = random_flip
            local_user_data[idx, v] = 1 if np.random.rand() < 0.5 else 0
            continue

        # Attacker
        avg1 = int(average_1_num_list[i])
        if splits < avg1:
            splits_k = min(int(splits), len(target_set))
            if splits_k > 0:
                splits_list = random.sample(list(target_set), splits_k)
                local_user_data[idx, splits_list] = 1
            remaining_set = list(set(range(domain)) - set(splits_list if splits_k > 0 else []))
            diff = avg1 - len(splits_list if splits_k > 0 else [])
            diff_AO = random.randint(max(0, diff - h_ao_local), diff + h_ao_local) if diff > 0 else 0
            if diff_AO > 0 and len(remaining_set) >= diff_AO:
                random_numbers = random.sample(remaining_set, diff_AO)
                local_user_data[idx, random_numbers] = 1
        else:
            k = min(avg1, len(target_set))
            if k > 0:
                splits_list = random.sample(list(target_set), k)
                local_user_data[idx, splits_list] = 1
            remaining_set = list(set(range(domain)) - set(splits_list if k > 0 else []))
            diff = avg1 - len(splits_list if k > 0 else [])
            diff_AO = random.randint(max(0, diff - h_ao_local), diff + h_ao_local) if diff > 0 else 0
            if diff_AO > 0 and len(remaining_set) >= diff_AO:
                random_numbers = random.sample(remaining_set, diff_AO)
                local_user_data[idx, random_numbers] = 1

    return local_user_data


def perturb_OUE_multi(
    X: np.ndarray,
    epsilon: float,
    domain: int,
    n: int,
    target_set: Set[int],
    ratio: float,
    h_ao: int,
    splits: int,
    num_processes: int = 4
) -> np.ndarray:
    """
    Perturb data using OUE protocol with attack simulation.
    
    Args:
        X: User data array
        epsilon: Privacy parameter
        domain: Domain size
        n: Number of users
        target_set: Set of target items for attack
        ratio: Attacker ratio
        h_ao: Attack optimization parameter
        splits: Number of splits
        num_processes: Parallel processes
    
    Returns:
        Perturbed user data matrix
    """
    q_OUE = 1 / (math.exp(epsilon) + 1)
    
    # Prepare average_1_num_list
    omega_probs = construct_omega(epsilon, domain, 'OUE')
    if h_ao == 1:
        average_1_num_list = np.random.choice(np.arange(domain), size=n, p=omega_probs)
    else:
        average_1_num_list = np.full(n, int(0.5 + (domain - 1) * q_OUE), dtype=int)

    # Prepare ranges for parallel processing
    ranges = []
    for i in range(num_processes):
        start = (i * n) // num_processes
        end = ((i + 1) * n) // num_processes if i < num_processes - 1 else n
        if end <= start:
            continue
        ranges.append((start, end, ratio, target_set, h_ao, splits, average_1_num_list))

    # Create pool with initializer
    with Pool(processes=len(ranges), initializer=_init_worker, initargs=(X, domain, q_OUE)) as pool:
        results = pool.map(_perturb_oue_process, ranges)

    return np.vstack(results)


def build_support_list_1_OUE(
    estimates: np.ndarray,
    n: int,
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """
    Build support list from OUE perturbed data.
    
    Args:
        estimates: Perturbed user data matrix
        n: Number of users
        epsilon: Privacy parameter
    
    Returns:
        Tuple of (support, one_list, ESTIMATE_DIST, None)
    """
    q_OUE = 1 / (math.exp(epsilon) + 1)
    p = 0.5
    
    Results_support = np.array(estimates)
    Estimations = np.sum(Results_support, axis=0)
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = [(i - n * q_OUE) / (p - q_OUE) for i in Estimations]
    
    return Results_support, Results_support_one_list, np.array(Estimations), None


num_samples = 1000000


def uniform_sampling_best_vector(target_set, g, d, m, num_samples):
    best_vector = None
    closest_ones_diff = float('inf')
    max_target_count = 0
    best_score = -float('inf')
    current_target = None
    current_diff = None

    for _ in range(num_samples):
        vector = np.random.binomial(1, 1 / g, size=d)

        # Count the number of 1's in the vector
        ones_count = np.sum(vector)
        ones_diff = abs(ones_count - m)

        target_count = sum(1 for item in target_set if vector[item % d] == 1)

        # Calculate the score: target_count - ones_diff
        score = target_count - ones_diff

        if score > best_score:
            best_score = score
            best_vector = vector
            current_target = target_count
            current_diff = ones_diff

    return best_vector, current_target, current_diff


def calculate_prob_according_sample_size(num_samples, d, g, h, target_set, splits):
    splits_list = random.sample(list(target_set), splits)
    target_set = splits_list
    user_vectors = []
    p = 1 / g

    mu = d * p
    sigma = np.sqrt(d * p * (1 - p))

    lower_bound = max(0, mu - h)
    upper_bound = min(d, mu + h)

    binom_dist = stats.binom(d, p)
    ratio = (binom_dist.cdf(upper_bound) - binom_dist.cdf(lower_bound - 1))
    ratio = ratio / (2 * h + 1)
    # ratio = 1

    N_effective = num_samples * ratio
    print('N_effective: ', N_effective)

    K_min = 1
    K_max = len(target_set)
    for K in range(K_max, K_min - 1, -1):
        prob = (p) ** K * N_effective
        if prob < 1:
            K_max = K
        else:
            break
    K_min = max(K_max, 1)

    K_values = np.arange(K_min, len(target_set) + 1)
    K_probs = []
    for K in K_values:
        prob = (p) ** K * N_effective
        K_probs.append(prob)
    K_probs = np.array(K_probs)

    K_probs = K_probs / np.sum(K_probs)

    return K_values, K_probs


def process_attacker(i, n, ratio, target_set, g, domain, splits, h_ao, e, K_values, K_probs):
    k = np.random.choice(K_values, p=K_probs)
    random.seed()
    averge_project_hash = int(domain / g)
    if splits < averge_project_hash:
        # Split the target set for each user
        splits_list = random.sample(list(target_set), splits)
        # Gap between average mapping
        num_map = averge_project_hash
        # Remaining set (unused in this snippet but kept for completeness)
        remaining_set = set(range(domain)) - set(target_set)
        #h_ao = 0
        if h_ao == 0:
          num_map_AO = random.randint(num_map - int(h_ao), num_map + int(h_ao))
        # num_map_AO num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        else:
          omega_probs = construct_omega(e, domain, 'OLH_User')
          num_map_AO = np.random.choice(range(domain), p=omega_probs)
        non_target_ones = num_map_AO - k
        # Each attacker finds their optimal hash function
        '''best_vector, target_map, diff  = uniform_sampling_best_vector(
            splits_list, g, domain, num_map_AO, num_samples)'''
        target_indices = np.random.choice(list(splits_list), size=k, replace=False)
        non_target_indices = list(set(range(domain)) - set(splits_list))
        non_target_selected = np.random.choice(non_target_indices, size=non_target_ones, replace=False)
        vector = np.zeros(domain, dtype=int)
        vector[target_indices] = 1
        vector[non_target_selected] = 1
    else:
        print('splits > averge_project_hash')
        exit(0)
    # Calculate the index in User_Seed to update
    index = int(n * (1 - ratio) + i)
    #print(f'attacker:{i}, target_map:{k}, diff:{num_map_AO - sum(vector)}, h_ao:{h_ao}, splits:{splits}')
    return index, vector


def process_user_seeds(i, User_Seed_noattack, Y_Nattack, domain, g):
    #print("Processing index" + str(i))
    local_estimate = np.zeros(domain)
    user_seed = User_Seed_noattack[i]
    for v in range(domain):
        if Y_Nattack[i] == (xxhash.xxh3_64(str(v), seed=int(user_seed)).intdigest() % g):
            local_estimate[v] += 1
    # Apply the correction factor
    local_estimate = local_estimate
    return local_estimate


def find_hash_function(seed_list, target_set, domain_eliminate, g, num_map_AO):
    # log the max projection number
    best_score = -np.inf
    # log the best projection seed
    best_seed = -1
    # log the target mapped
    best_target_mapped = None
    # log the best hash value
    best_hash_value = None
    # log the min gap
    best_gap = None
    for seed in seed_list:
        hash_projection_list = np.zeros(g)
        hash_other_projection_list = np.zeros(g)
        hash_result = None
        for item in target_set:
            hash_result = xxhash.xxh3_64(str(item), seed=seed).intdigest() % g
            hash_projection_list[hash_result] += 1
        for item in domain_eliminate:
            hash_result = xxhash.xxh3_64(str(item), seed=seed).intdigest() % g
            hash_other_projection_list[hash_result] += 1
        score = hash_projection_list - np.abs(num_map_AO - hash_projection_list - hash_other_projection_list)
        current_best_score = np.max(score)
        max_indices = np.where(score == current_best_score)[0]
        current_max_target_mapped = hash_projection_list[max_indices]
        current_untarget_mapped = hash_other_projection_list[max_indices]
        current_hash_value = max_indices
        current_gap = np.abs(num_map_AO - current_max_target_mapped - current_untarget_mapped)
        if current_best_score > best_score:
            best_seed = seed
            best_score = current_best_score
            best_hash_value = current_hash_value
            best_gap = current_gap
            best_target_mapped = current_max_target_mapped
    if best_seed == -1:
        return -1, 0.0, None
    return best_seed, best_gap, best_target_mapped, best_hash_value


def process_attacker_User(i, n, ratio, target_set, g, domain, splits, e, h_ao):
    average_project_hash = int(domain / g)
    vector = np.zeros(domain, dtype=int)
    h_ao = int(h_ao)
    if splits < average_project_hash:
        # Split the target set for each user
        splits_list = random.sample(list(target_set), splits)
        # Gap between average mapping
        num_map = average_project_hash
        # Remaining set (unused in this snippet but kept for completeness)
        remaining_set = set(range(domain)) - set(target_set)
        # Adaptive gap between average mapping
        num_map_AO = random.randint(num_map - h_ao, num_map + h_ao)
        # theoretical APA use the omega_list to replace the num_map_AO num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        #num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        seed_list = random.sample(range(1, 10000000), num_samples)
        best_seed, best_gap, current_max_target_mapped, best_hash_value = find_hash_function(seed_list, splits_list,
                                                                                             remaining_set, g,
                                                                                             num_map_AO)
    else:
        print('splits > averge_project_hash')
        exit(0)
    # Calculate the index in User_Seed to update
    index = int(n * (1 - ratio) + i)
    for v in range(domain):
        hashed_value = xxhash.xxh3_64(str(v), seed=int(best_seed)).intdigest() % g
        if hashed_value == best_hash_value:
            vector[v] = 1
   # print(f'attacker:{i}, target_map:{current_max_target_mapped}, diff:{best_gap}, h_ao:{h_ao}, splits:{splits}')
    return index, vector


def build_support_list_1_OLH(domain, Y, n, User_Seed, ratio, g, target_set, p, splits, e, h_ao=0, processor=100):
    #K_values, K_probs = calculate_prob_according_sample_size(num_samples, domain, g, h_ao, target_set, splits)

    # Prepare the partial function with fixed arguments for multiprocessing
    process_attacker_partial = partial(
        process_attacker_User,
        n=n,
        ratio=ratio,
        target_set=target_set,
        g=g,
        domain=domain,
        splits=splits,
        h_ao= 10*h_ao,
        e=e,
    )

    # Calculate the number of attackers
    num_attackers = int(round(n * ratio))

    # Parallel execution of process_attacker using multiprocessing
    with Pool(processes=processor) as pool:
        # Use imap to process in parallel and tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_attacker_partial, range(num_attackers)),
            total=num_attackers,
            desc='Finding optimal seeds'
        ))

    vector_matrix = np.zeros((num_attackers, domain))
    # Update User_Seed with the results from all attackers
    for i, (index, best_vector) in enumerate(results):
        vector_matrix[i, :] = best_vector

    # Create a partial function with fixed arguments for processing user seeds
    process_partial = partial(
        process_user_seeds,
        User_Seed_noattack=User_Seed,
        Y_Nattack=Y,
        domain=domain,
        g=g
    )

    # Process user seeds across multiple processes
    with Pool(processes=processor) as pool:
        estimates = pool.map(process_partial, range(n - num_attackers))
    # input attack's estimate
    '''with Pool(processes=processor) as pool:
        estimates_input = pool.map(process_partial, range(int(n)))'''

    # Combine the results from all attackers
    estimates = np.array(estimates)
    # estimates_input = np.array(estimates_input)
    # estimates_input = estimates_input.reshape(int(n), domain)
    estimates = np.vstack((estimates, vector_matrix))
    estimates = estimates.reshape(int(n), domain)
    Results_support = estimates
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = np.sum(Results_support, axis=0)
    # Estimations_input = np.sum(estimates_input, axis=0)
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations = a * Estimations - b
    # Estimations_input = a * Estimations_input - b
    Estimations_input = None
    return Results_support, Results_support_one_list, Estimations, Estimations_input


def process_attacker_server(i, n, ratio, target_set, g, domain, User_Seed, splits):
    '''
    Function to process each attacker.
    Each attacker finds the best hash value and constructs a vector based on it.

    :param i: Index of the attacker.
    :param n: Total number of users.
    :param ratio: Ratio of attackers to total users.
    :param target_set: Set of target items.
    :param g: Range of hash function outputs (modulo value).
    :param domain: Total domain size.
    :param User_Seed: List of hash seeds for users.
    :return: Tuple of (index in User_Seed, attack_vector).
    '''
    # Calculate the index in User_Seed
    index = int(n * (1 - ratio) + i)
    user_seed = User_Seed[index]

    # Compute hash values for all target items
    target_hashes = {}
    splits_list = random.sample(list(target_set), splits)
    for t in splits_list:
        hashed_value = xxhash.xxh3_64(str(t), seed=int(user_seed)).intdigest() % g
        if hashed_value in target_hashes:
            target_hashes[hashed_value] += 1
        else:
            target_hashes[hashed_value] = 1

    # Find the hash value that maps the most target items
    best_hashed_value = max(target_hashes, key=target_hashes.get)
    max_target_count = target_hashes[best_hashed_value]

    # Construct the attack vector
    attack_vector = np.zeros(domain)
    for v in range(domain):
        hashed_value = xxhash.xxh3_64(str(v), seed=int(user_seed)).intdigest() % g
        if hashed_value == best_hashed_value:
            attack_vector[v] = 1

    print(f'attacker:{i}, best_hashed_value:{best_hashed_value}, max_targets_mapped:{max_target_count}')
    return index, attack_vector


def build_support_list_1_OLH_Server(domain, Y, n, User_Seed, ratio, g, target_set, p, splits, h_ao=0, epsilon=1.0, processor=100):
    '''
    build the support list matrix
    :return:
    '''
    # Prepare the partial function with fixed arguments for multiprocessing
    process_attacker_partial = partial(
        process_attacker_server,
        n=n,
        ratio=ratio,
        target_set=target_set,
        g=g,
        domain=domain,
        User_Seed=User_Seed,
        splits=splits
    )

    # Calculate the number of attackers
    num_attackers = int(round(n * ratio))
    num_normal = int(n - num_attackers)

    # Parallel execution of process_attacker using multiprocessing
    with Pool(processes=processor) as pool:
        # Use imap to process in parallel and tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_attacker_partial, range(num_attackers)),
            total=num_attackers,
            desc='Processing attackers'
        ))

    vector_matrix = np.zeros((num_attackers, domain))
    # Update User_Seed with the results from all attackers
    for i, (index, best_vector) in enumerate(results):
        vector_matrix[i, :] = best_vector

    # Create a partial function with fixed arguments for processing user seeds
    process_partial = partial(
        process_user_seeds,
        User_Seed_noattack=User_Seed,
        Y_Nattack=Y,
        domain=domain,
        g=g
    )

    # Process user seeds across multiple processes
    with Pool(processes=processor) as pool:
        estimates = pool.map(process_partial, range(int(num_normal)))
    # input attack's estimate
    '''with Pool(processes=processor) as pool:
        estimates_input = pool.map(process_partial, range(int(n)))'''

    # Combine the results from all attackers
    estimates = np.array(estimates)
    # estimates_input = np.array(estimates_input)
    # estimates_input = estimates_input.reshape(int(n), domain)
    estimates = np.vstack((estimates, vector_matrix))
    estimates = estimates.reshape(int(n), domain)
    Results_support = estimates
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = np.sum(Results_support, axis=0)
    # Estimations_input = np.sum(estimates_input, axis=0)
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations = a * Estimations - b
    # Estimations_input = a * Estimations_input - b
    Estimations_input = None
    return Results_support, Results_support_one_list, Estimations, Estimations_input


def HST_Server(X, ratio, domain, epsilon, n, target_set, splits):
    '''
    Perform the HST protocol
    :param X: The real values for each users
    :param ratio: fake users ratio
    :param domain: domain size
    :param epsilon: privacy budget
    :param n: number of users
    :param target_set: fake users target set
    :return: support_list, one_list, ESTIMATE_DIST, ESTIMATE_Input
    '''
    c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    s_vectors = np.zeros((n, domain))
    fake_user_num = int(round(n * ratio))
    normal_user_num = n - fake_user_num
    start_idx = n - fake_user_num
    y_values = np.zeros(n)
    for i in range(n):
        # Generate random public vector s_i
        s_i = np.random.choice([-1.0, 1.0], size=domain)
        s_vectors[i, :] = s_i
    for i in range(normal_user_num):
        # User's true data item
        v = X[i]  # Assuming X[i] is in the range [0, domain-1]
        # Generate random public vector s_j
        s_i = s_vectors[i, :]
        # Get s_j[v_b]
        s_i_v = s_i[v]
        # Perturbation process
        if random.random() < math.exp(epsilon) / (math.exp(epsilon) + 1):
            y = c * s_i_v
        else:
            y = -c * s_i_v
        y_values[i] = y
    splits_list = random.sample(list(target_set), splits)
    for i in range(fake_user_num):
        idx = start_idx + i
        s_i = s_vectors[idx, :]
        positive_count = 0
        negative_count = 0
        for v in splits_list:
            if s_i[v] == 1.0:
                positive_count += 1
            elif s_i[v] == -1.0:
                negative_count += 1
        if positive_count >= negative_count:
            y = c
        else:
            y = -c
        y_values[idx] = y
        print(f'Attacker {i}, idx: {idx}, positive_count: {positive_count}, negative_count: {negative_count}, y: {y}')

    support_list = y_values.reshape(-1, 1) * s_vectors
    ESTIMATE_DIST = np.sum(support_list, axis=0)
    Results_support_one_list = np.sum(s_vectors == 1, axis=1)

    return support_list, Results_support_one_list, ESTIMATE_DIST, ESTIMATE_DIST


def HST_Users(X, ratio, domain, epsilon, n, target_set, h_ao, splits):
    '''
    Perform the HST protocol
    :param X: The real values for each users
    :param ratio: fake users ratio
    :param domain: domain size
    :param epsilon: privacy budget
    :param n: number of users
    :param target_set: fake users target set
    :param h_ao: APA's parameter
    :param splits: MGA-A's parameter
    :return: support_list, one_list, ESTIMATE_DIST, ESTIMATE_Input
    '''
    c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    s_vectors = np.zeros((n, domain))
    average_1_num = domain / 2
    fake_user_num = int(round(n * ratio))
    normal_user_num = n - fake_user_num
    start_idx = n - fake_user_num
    h_ao *= 10
    y_values = np.zeros(n)
    # theoretical APA use the omega_list to replace the averge_1_num and set h_ao = 0
    '''if h_ao != 0:
        average_1_num = np.random.choice([i for i in range(domain)], construct_omega(epsilon, domain, 'OUE'))
        h_ao = 0'''
    for i in range(normal_user_num):
        # User's true data item
        v = X[i]  # Assuming X[i] is in the range [0, domain-1]
        # Generate random public vector s_j
        s_i = np.random.choice([-1.0, 1.0], size=domain)
        s_vectors[i, :] = s_i
        # Get s_j[v_b]
        s_i_v = s_i[v]
        # Perturbation process
        if random.random() < math.exp(epsilon) / (math.exp(epsilon) + 1):
            y = c * s_i_v
        else:
            y = -c * s_i_v
        y_values[i] = y
    for i in range(fake_user_num):

        splits_list = random.sample(list(target_set), splits)
        local_user_data = np.full(domain, -1.0)
        local_user_data[list(splits_list)] = 1
        remaining_set = list(set(range(domain)) - set(splits_list))
        diff = int(average_1_num - len(splits_list))
        diff_AO = random.randint(diff - h_ao, diff + h_ao)
        #print(f'attacker:{i}, exp1:{average_1_num}, h_ao:{h_ao}, splits:{splits}')
        if diff_AO > 0 and len(remaining_set) >= diff:
            random_numbers = random.sample(remaining_set, diff_AO)
            local_user_data[random_numbers] = 1
        idx = start_idx + i
        s_vectors[idx,:] = local_user_data
        y = c
        y_values[idx] = y

    support_list = y_values.reshape(-1, 1) * s_vectors
    ESTIMATE_DIST = np.sum(support_list, axis=0)
    Results_support_one_list = np.sum(s_vectors == 1, axis=1)

    return support_list, Results_support_one_list, ESTIMATE_DIST, ESTIMATE_DIST


#def perturb_normal_olh(X, g, p, q):
    Y_normal = np.zeros(len(X))
    for i, v in enumerate(X):
        # Generate hash value
        x = (xxhash.xxh3_64(str(v), seed=i).intdigest() % g)
        y = x
        p_sample = np.random.random_sample()

        # Apply perturbation
        if p_sample <= p:

            # perturb
            y = np.random.randint(0, g)
        Y_normal[i] = y
    return Y_normal


#def build_support_list_normal_olh(Y_normal, n, domain, g, p):
    Results_support_normal = np.zeros((n, domain))
    Estimations_normal_raw = np.zeros(domain)

    for i in range(n):
        user_seed = i  # Use the same seed as in perturbation
        for v in range(domain):
            hashed_value = (xxhash.xxh3_64(str(v), seed=user_seed).intdigest() % g)
            if Y_normal[i] == hashed_value:
                Results_support_normal[i, v] += 1
                Estimations_normal_raw[v] += 1

    Results_support_one_list_normal = np.sum(Results_support_normal, axis=1)

    # Apply the OLH correction factor
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations_normal = a * Estimations_raw - b

    return Results_support_normal, Results_support_one_list_normal, Estimations_normal
