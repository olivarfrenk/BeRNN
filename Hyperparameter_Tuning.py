########################################################################################################################
# info: Hyperparameter Tuning
########################################################################################################################
# Random Grid Search of different hyperparameter sets for automated accumulated model training.

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import random
# from sklearn.model_selection import ParameterGrid
import itertools

import Training

########################################################################################################################
# Create HP combinations and randomly choose a selection
########################################################################################################################
def create_param_combinations(param_grid, sample_size):
    # Create all possible combinations of parameters
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    # Randomly sample the specified number of combinations
    sampled_combinations = random.sample(all_combinations, sample_size)

    return sampled_combinations

def create_repeated_param_combinations(param_grid, sample_size):
    # Create the single combination of parameters
    keys, values = zip(*param_grid.items())
    single_combination = dict(zip(keys, [v[0] for v in values]))

    # Return the same combination 'sample_size' times
    repeated_combinations = [single_combination for _ in range(sample_size)]

    return repeated_combinations

# Info: After first HPs the most probable space inheriting the best solution decreased to the following
best_params = {
    'batch_size': [40],  # Wrap the integer in a list
    'in_type': ['normal'],  # Wrap the string in a list
    'rnn_type': ['LeakyGRU'],  # Wrap the string in a list
    'use_separate_input': [False],  # Wrap the boolean in a list
    'loss_type': ['lsq'],  # Wrap the string in a list
    'optimizer': ['adam'],  # Wrap the string in a list
    'activation': ['relu'],  # Wrap the string in a list
    'tau': [100],  # Wrap the integer in a list
    'dt': [20],  # Wrap the integer in a list
    'sigma_rec': [0.05],
    'sigma_x': [0.01],
    'w_rec_init': ['randortho'],  # Wrap the string in a list
    'l1_h': [0.0001,0.00005,0.00001],  # Wrap the integer in a list
    'l2_h': [0.0001,0.00005,0.00001], # .00001
    'l1_weight': [0.0001,0.00005,0.00001],  # Wrap the integer in a list
    'l2_weight': [0.0001,0.00005,0.00001], # .00001
    'l2_weight_init': [0],
    'p_weight_train': [None],  # Wrap None in a list
    'learning_rate': [0.001],  # .001
    'n_rnn': [512], # 512
    'c_mask_responseValue': [5.],
    'monthsConsidered': [['2','3','4'],['4','5','6'],['6','7','8'], ['2','3','4','5','6','7','8']]  # Already a list
}
# # Randomly sample combinations
# sampled_combinations = create_param_combinations(best_params, 20)

# Create one combination and repeat it according to sample_size
sampled_repeated_combinations = create_repeated_param_combinations(best_params, 5)


# Training #############################################################################################################
model_number = 1
# Example iteration through the grid
for params in sampled_repeated_combinations: # info: either sampled_combinations OR sampled_repeated_combinations
    print('START TRAINING FOR NEW MODEL')
    print(params) # Double check with model output files

    # Predefine certain variables
    participant = 'BeRNN_03'
    monthsConsidered = params['monthsConsidered']

    # Data paths for different server
    # preprocessedData_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\Data', participant,'PreprocessedData_wResp_ALL')
    # preprocessedData_path = os.path.join('/data/Data', participant, 'PreprocessedData_wResp_ALL')
    preprocessedData_path = os.path.join('/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main/Data', participant, 'PreprocessedData_wResp_ALL')

    model = 'Model_' + str(model_number) + '_' + participant + '_Month_' + monthsConsidered[0] + '-' + monthsConsidered[-1]
    # Model directories for different server
    # model_dir = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\BeRNN_01_HPT01', model)
    # model_dir = os.path.join('/data/BeRNN_01_HPT04', model)
    model_dir = os.path.join('/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main/BeRNN_03_HPT03', model)

    # Define probability of each task being trained
    rule_prob_map = {"DM": 1, "DM_Anti": 1, "EF": 1, "EF_Anti": 1, "RP": 1, "RP_Anti": 1, "RP_Ctx1": 1, "RP_Ctx2": 1,
                     "WM": 1, "WM_Anti": 1, "WM_Ctx1": 1, "WM_Ctx2": 1}

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Measure the training time
    start_time = time.time()
    print(f'START TRAINING MODEL: {model_number}')

    # Split the data ---------------------------------------------------------------------------------------------------
    # List of the subdirectories
    subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if os.path.isdir(os.path.join(preprocessedData_path, d))]

    # Initialize dictionaries to store training and evaluation data
    train_data = {}
    eval_data = {}

    # Function to split the files
    def split_files(files, split_ratio=0.8):
        # random.seed(42) # info: add seed to always shuffle similiar - would be good for NetworkAnalysis
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)
        return files[:split_index], files[split_index:]

    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_triplets = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
                # III: Exclude files with specific substrings in their names
                # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                #     continue
                # Include only files that contain any of the months in monthsConsidered
                if not any(month in file for month in monthsConsidered):
                    continue
                base_name = file.split('Input')[0]
                # print(base_name)
                input_file = os.path.join(subdir, base_name + 'Input.npy')
                yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                output_file = os.path.join(subdir, base_name + 'Output.npy')
                file_triplets.append((input_file, yloc_file, output_file))

        # Split the file triplets
        train_files, eval_files = split_files(file_triplets)

        # Store the results in the dictionaries
        train_data[subdir] = train_files
        eval_data[subdir] = eval_files

    try:
        Training.train(model_dir=model_dir, hp=params, rule_prob_map=rule_prob_map, train_data = train_data, eval_data = eval_data)

        end_time = time.time()
        elapsed_time_minutes = end_time - start_time / 60
        elapsed_time_hours = elapsed_time_minutes / 60

        print(f"TIME TAKEN TO TRAIN MODEL {model_number}: {elapsed_time_hours:.2f} hours")
    except:
        print("An exception occurred with model number: ", model_number)

    # Count up for next model
    model_number += 1


