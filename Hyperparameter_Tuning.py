# HP Tuning ############################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import random
from sklearn.model_selection import ParameterGrid

import Training

param_grid = {
    'batch_size': [20, 40],
    'in_type': ['normal'],
    'rnn_type': ['LeakyRNN', 'LeakyGRU'],
    'use_separate_input': [False], # todo: True doesn't work
    'loss_type': ['lsq'],
    'optimizer': ['adam'],
    'activation': ['softplus'],
    'tau': [50, 100],
    'dt': [10, 20],
    'sigma_rec': [0.01, 0.05],
    'sigma_x': [0.001, 0.01],
    'w_rec_init': ['diag', 'randortho'],
    'l1_h': [0, 0.0001],
    'l2_h': [0, 0.00001],
    'l1_weight': [0, 0.0001],
    'l2_weight': [0, 0.0001],
    'l2_weight_init': [0, 0.0001],
    'p_weight_train': [None, 0.05],
    'learning_rate': [0.0001, 0.001],
    'n_rnn': [128, 256, 512],
    'c_mask_responseValue': [1.]
}
# Create all possible combinations
grid = list(ParameterGrid(param_grid))
# Randomly sample 100 of them
sampled_grid = random.sample(grid, 100)

# Training #############################################################################################################
model_number = 1
# Example iteration through the grid
for params in sampled_grid:
    print('START TRAINING FOR NEW MODEL')
    print(params) # Double check with model output files
    # Predefine certain variables
    participant = 'BeRNN_03'
    monthsConsidered = ['2', '3', '4', '5', '6', '7']
    # preprocessedData_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\Data', participant,'PreprocessedData_wResp_ALL')
    preprocessedData_path = os.path.join('/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main/Data_070724', participant,'PreprocessedData_wResp_ALL')
    # Define probability of each task being trained
    rule_prob_map = {"DM": 1, "DM_Anti": 1, "EF": 1, "EF_Anti": 1, "RP": 1, "RP_Anti": 1, "RP_Ctx1": 1, "RP_Ctx2": 1,
                     "WM": 1, "WM_Anti": 1, "WM_Ctx1": 1, "WM_Ctx2": 1}
    model = 'Model_' + str(model_number) + '_' + participant + '_Month_' + monthsConsidered[0] + '-' + monthsConsidered[-1]
    # model_dir = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models', model)
    model_dir = os.path.join('/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main/BeRNN_03_HPT01', model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Measure the training time
    start_time = time.time()
    print(f'START TRAINING MODEL: {model_number}')

    # III: Split the data ##############################################################################################
    # List of the subdirectories
    subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if
               os.path.isdir(os.path.join(preprocessedData_path, d))]

    # Initialize dictionaries to store training and evaluation data
    train_data = {}
    eval_data = {}


    # Function to split the files
    def split_files(files, split_ratio=0.8):
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)
        return files[:split_index], files[split_index:]


    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_triplets = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
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
    # III: Split the data ##############################################################################################

    Training.train(model_dir=model_dir, trial_dir=preprocessedData_path, monthsConsidered=monthsConsidered, hp=params, rule_prob_map=rule_prob_map, train_data = train_data, eval_data = eval_data)

    end_time = time.time()
    elapsed_time_minutes = end_time - start_time / 60
    elapsed_time_hours = elapsed_time_minutes / 60

    print(f"TIME TAKEN TO TRAIN MODEL {model_number}: {elapsed_time_hours:.2f} hours")

    # Count up for next model
    model_number += 1

