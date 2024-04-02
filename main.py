import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spl
from scipy.sparse import csr_matrix, csc_matrix
from collections import defaultdict
import itertools
import os.path
import time
import os
from scipy.sparse import dok_matrix

def write_to_file(file_path, content):
    with open(file_path, 'a') as output_file:
        output_file.write(content)
    output_file.close()

def cosine_similarity(X, i, j):
    a_norm = spl.norm(X[i])
    b_norm = spl.norm(X[j]) 

    if a_norm == 0 or b_norm == 0:
        return 0.0

    ab = X[i].dot(X[j].T)[0, 0]
    alpha = np.arccos(ab / (a_norm * b_norm))

    if np.isnan(alpha):
        return 0.0

    angle_degrees = np.degrees(alpha)
    cosine_similarity_measure = 1 - (angle_degrees / 180)

    return cosine_similarity_measure

def create_sparse_matrix(input_data):
    user_indices, movie_indices = input_data[:, 0], input_data[:, 1]
    data = np.ones(len(user_indices))
    sparse_matrix = csc_matrix((data, (user_indices, movie_indices)), dtype=np.int8)
    return sparse_matrix

def create_signature_matrix(sparse_matrix, num_signatures, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    num_users, num_items = sparse_matrix.shape
    signature_matrix = np.zeros((num_users, num_signatures))

    for signature_index in range(num_signatures):
        random_indices = np.random.choice(num_items, 500, replace=False)
        selected_items = sparse_matrix[:, random_indices]
        signature_column = np.argmax(selected_items, axis=1)
        signature_matrix[:, signature_index] = signature_column.reshape(-1) 

    return signature_matrix

def create_bands_and_generate_pairs(signature_matrix, num_bands, signature_length, threshold=0.5, max_bucket_size=800):
    num_users, num_hashes = signature_matrix.shape
    band_width = num_hashes // num_bands

    list_of_buckets = []
    unique_pairs = set()

    for band_id in range(num_bands):
        start_col = band_id * band_width
        end_col = (band_id + 1) * band_width
        band_signature_matrix = signature_matrix[:, start_col:end_col]

        buckets = defaultdict(list)

        for user_id in range(num_users):
            signature_tuple = tuple(band_signature_matrix[user_id])
            buckets[signature_tuple].append(user_id)

        non_empty_buckets = {key: value for key, value in buckets.items() if len(value) > 1}
        list_of_buckets.append(non_empty_buckets)

    for band_id, buckets in enumerate(list_of_buckets):
        for bucket in buckets.values():
            if len(bucket) < max_bucket_size:
                pairs = np.array(list(itertools.combinations(bucket, 2)))
                pair_signatures = signature_matrix[pairs[:, 0]] == signature_matrix[pairs[:, 1]]
                similarities = np.sum(pair_signatures, axis=1) / signature_length
                valid_pairs = pairs[similarities > threshold]
                unique_pairs.update(map(tuple, valid_pairs))

    return list_of_buckets, unique_pairs

def jaccard_similarity(unique_pairs, sparse_matrix):
    count = 0
    matrix_array = sparse_matrix.toarray()

    def calculate_jaccard(user1, user2):
        intersection = np.count_nonzero(matrix_array[user1] & matrix_array[user2])
        union = np.count_nonzero(matrix_array[user1] | matrix_array[user2])
        return intersection / union if union != 0 else 0

    for user_pair in unique_pairs:
        user1, user2 = user_pair
        similarity = calculate_jaccard(user1, user2)
        if similarity >= 0.5:
            count += 1
            write_to_file('js.txt', f'{user1},{user2}\n')

    print(f"There are: {count} unique pairs in JS with more than 0.5 similarity")

def process_cosine_similarity(data_matrix, num_bands, num_rows, threshold, random_seed, output_filename):
    np.random.seed(random_seed)
    num_users, num_items = data_matrix.shape

    projection_vectors = np.random.normal(0.0, 1.0, (num_items, num_bands * num_rows))

    random_projections = (data_matrix.dot(projection_vectors) >= 0).reshape((num_users, num_bands, num_rows))
    values = np.power(3, np.repeat([np.repeat([np.arange(num_rows)], num_bands, axis=0)], num_users, axis=0))

    hashed_values = np.multiply(random_projections, values)
    hashed_values = np.sum(hashed_values, axis=2)

    current_directory = os.getcwd()
    print(f"Writing file to: {current_directory}")

    for i in range(num_bands):
        unique_values, counts = np.unique(hashed_values[:, i], return_counts=True)
        non_unique_values = unique_values[counts > 1]

        for val in non_unique_values:
            candidate_pairs = np.where(hashed_values[:, i] == val)[0]
            for pair in itertools.combinations(candidate_pairs, 2):
                similarity = cosine_similarity(data_matrix, pair[0], pair[1])
                if similarity > threshold:
                    with open(output_filename, 'a') as f:  
                        f.write(f'{pair[0]}, {pair[1]}\n')  
                    
                    f.close()

    return None, 0 


def process_js_similarity(input_data, random_seed):
    num_rows = 70
    num_bands = 30
    sparse_matrix = create_sparse_matrix(input_data)
    signature_matrix = create_signature_matrix(sparse_matrix, num_rows, random_seed)
    list_of_buckets, unique_users = create_bands_and_generate_pairs(signature_matrix, num_bands, num_rows)
    jaccard_similarity(unique_users, sparse_matrix)

def process_cs_similarity(input_data, random_seed):
    start_time = time.time()
    row_indices, column_indices, values = zip(*input_data[:, :3])

    cosine_matrix = csr_matrix((values, (row_indices, column_indices)), dtype=int)
    process_cosine_similarity(data_matrix=cosine_matrix, random_seed=random_seed, num_bands=30, num_rows=25, threshold=0.73, output_filename='cs.txt')

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print("Runtime:", elapsed_time_minutes, "minutes")

def process_dcs_similarity(users, random_seed):
    indices_dict = defaultdict(lambda: len(indices_dict))
    indptr, indices, data = [0], [], []

    for user in users:
        user_indices = [indices_dict[rating] for rating in user]
        indices.extend(user_indices)
        data.extend([1] * len(user_indices))
        indptr.append(len(indices))

    movies = list(indices_dict.keys())
    discrete_matrix = csr_matrix((data, indices, indptr), dtype=int)
    process_cosine_similarity(data_matrix=discrete_matrix, random_seed=random_seed, num_bands=35, num_rows=23, threshold=0.73, output_filename='dcs.txt')

def process(input_filepath, random_seed, similarity_measure):
    input_data = np.load(input_filepath)
    users_data = np.split(input_data[:, 1], np.unique(input_data[:, 0], return_index=True)[1][1:])
    random_seed = random_seed

    if similarity_measure == 'js':
        process_js_similarity(input_data, random_seed)
    elif similarity_measure == 'cs':
        process_cs_similarity(input_data, random_seed)
    elif similarity_measure == 'dcs':
        process_dcs_similarity(users_data, random_seed)
    else:
        print("Please enter a valid similarity measure")

    return 0

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-d', type=str, required=True)
    argument_parser.add_argument('-s', type=int, required=True)
    argument_parser.add_argument('-m', type=str, required=True)
    command_line_args = argument_parser.parse_args()

    process(command_line_args.d, command_line_args.s, command_line_args.m)
