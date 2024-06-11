########################################################################################################################
# todo: LAB ############################################################################################################
########################################################################################################################
import os
import numpy as np
import random
import glob

# todo: Delete all files from which not all three necessary files exist

# Choose participant data to augment data from
dataFolder = "Data"
participants = ['BeRNN_05']
preprocessedData_folder = 'PreprocessedData_wResp_ALL'
for participant in participants:
    trial_dir = os.path.join("W:\\group_csp\\analyses\\oliver.frank", dataFolder, participant, preprocessedData_folder)
    # trial_dir = os.path.join("/zi/flstorage/group_csp/analyses/oliver.frank", dataFolder, participant, preprocessedData_folder)


    # # Delete all augmented files ###########################################################################################
    # Patterns to search for
    patterns = ['*Rotation*', '*Mirrored*', '*segmentation*', '*Randomization*']
    def delete_files(trial_dir, patterns):
        for pattern in patterns:
            # Search for files containing "rotation" in their filename
            filesToDelete = glob.glob(os.path.join(trial_dir, pattern))
            # Delete each file found
            for file_path in filesToDelete:
                try:
                    os.remove(file_path)
                    print(f'Deleted: {file_path}')
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')

    # tasks = ['RP', 'RP_Anti']
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']
    for task in tasks:
        delete_files(os.path.join(trial_dir,task), patterns)
    # ########################################################################################################################


    ########################################################################################################################
    # todo: Data Augmentation ##############################################################################################
    ########################################################################################################################

    # 1: Segmental Pertubation Between #####################################################################################
    # DM/EF/RP-Tasks: Factor 4
    # 10 batches (segmentSize:10%), 5 batches (segmentSize:20%), 4 batches (segmentSize:25%), 2 batches (segmentSize:50%) creating 60 new batches, respectively
    # tasks = ['RP', 'RP_Anti']
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2']

    # Function to randomly select given number of trials from each batch
    def select_trials(segmentNumb, segmentSize, trial_list):
        selected_trials_x = []
        selected_trials_y = []
        selected_trials_y_loc = []
        filtered_batchList = [f for f in trial_list if len(os.path.basename(f).split('-')) == 6 and os.path.basename(f).split('-')[1] != 'month_1'] # exclude experimental month
        for batch in filtered_batchList:
            indices = list(set(range(0, 40)))
            for i in range(segmentNumb):
                file_stem = batch.split('-')[:-1]
                month = batch.split('-')[1]
                batchNumber = batch.split('-')[2]

                try: # As it is prone to error if one of the file is actually missing
                    input_file = os.path.join(trial_dir, task, '-'.join(file_stem) + '-Input.npy')
                    output_file = os.path.join(trial_dir, task, '-'.join(file_stem) + '-Output.npy')
                    yLoc_file = os.path.join(trial_dir, task, '-'.join(file_stem) + '-yLoc.npy')
                    # Check if all required files exist
                    if os.path.exists(input_file) and os.path.exists(output_file) and os.path.exists(yLoc_file):
                        # Load Input, Output and Response for every batch in defined folder
                        loadedBatch_x = np.load(input_file)  # Trial Activity
                        loadedBatch_y = np.load(output_file)  # Participant Response
                        loadedBatch_y_loc = np.load(yLoc_file)  # Ground Truth
                    else:
                        # If any of the files are missing, delete the existing files for this batch
                        if os.path.exists(input_file):
                            os.remove(input_file)
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        if os.path.exists(yLoc_file):
                            os.remove(yLoc_file)
                        print(f"Files for batch {batch} are incomplete or missing. Skipping this batch.")

                except FileNotFoundError:
                    # If the file doesn't exist, skip this batch
                    print(f"Files for batch {batch} not found. Skipping this batch.")
                    continue

                # Select indices to be shuffled
                drawn_indices = random.sample(indices, segmentSize)
                for index in drawn_indices:
                    indices.remove(index)
                # Select the same indices form all three information components
                selected_trials_x.append(loadedBatch_x[:, drawn_indices, :])
                selected_trials_y.append(loadedBatch_y[:, drawn_indices, :])
                selected_trials_y_loc.append(loadedBatch_y_loc[:, drawn_indices])

        return selected_trials_x, selected_trials_y, selected_trials_y_loc, month, batchNumber

    # SegmentNumber and SegmentSize have to result in total of 40
    segmentNumb_segmentSize = [[10,4], [5,8], [4,10], [2,20]]
    for task in tasks:
        trial_list = glob.glob(os.path.join(trial_dir, task, '*Input.npy'))  # Define list of batches to augment on
        os.chdir(os.path.join(trial_dir, task))  # Define folder to save files in
        try:
            for segmentPair in segmentNumb_segmentSize:
                # Predefine variables
                segmentNumb = segmentPair[0]  # numb of newly created batches
                segmentSize = segmentPair[1] # segment size
                # Get the concatenated selected trials list
                selected_trials_x, selected_trials_y, selected_trials_y_loc, month, batchNumber = select_trials(segmentNumb, segmentSize, trial_list)

                # Number of batches to newly create
                numbOfNewBatches = len(selected_trials_x)/segmentNumb
                for i in range(int(numbOfNewBatches)):
                    newBatch_x = np.concatenate(selected_trials_x[i*segmentNumb:(i*segmentNumb+segmentNumb)], axis=1)
                    newBatch_y = np.concatenate(selected_trials_y[i*segmentNumb:(i*segmentNumb+segmentNumb)], axis=1)
                    newBatch_y_loc = np.concatenate(selected_trials_y_loc[i*segmentNumb:(i*segmentNumb+segmentNumb)], axis=1)
                    # Save all three files
                    input_filename = participant + '-' + month + '-' + batchNumber + '-' + task + '-' + 'SegmentNumber_' + str(segmentNumb) + '-' + 'Segmentation_' + str(i) + '-' + 'Input'
                    np.save(input_filename, newBatch_x)
                    input_filename = participant + '-' + month + '-' + batchNumber + '-' + task + '-' + 'SegmentNumber_' + str(segmentNumb) + '-' + 'Segmentation_' + str(i) + '-' + 'Output'
                    np.save(input_filename, newBatch_y)
                    input_filename = participant + '-' + month + '-' + batchNumber + '-' + task + '-' + 'SegmentNumber_' + str(segmentNumb) + '-' + 'Segmentation_' + str(i) + '-' + 'yLoc'
                    np.save(input_filename, newBatch_y_loc)

                    print(f'Segmented batch augmented for task {task}, segmentNumb {segmentNumb} and segmentSize {segmentSize}')

        except ValueError as err:
            print(f"ValueError occurred: {err.args}")
        except Exception as e:
            print(f"An error occurred: {e}")


    # 2: Geometric Rotation ################################################################################################
    # DM/EF/RP-Tasks: Factor 5
    # 67.5 (6 steps), 135 (12 steps), 202.5 (18 steps), 270 (24 steps) and 337.5 (30 steps) degrees
    # tasks = ['RP', 'RP_Anti']
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2']
    for task in tasks:
        trial_list = glob.glob(os.path.join(trial_dir, task, '*Input.npy'))  # Define list of batches to augment on
        os.chdir(os.path.join(trial_dir, task))  # Define folder to save files in
        try:
            for k in trial_list:
                if k.split('-')[1] != 'month_1':
                    # Split every drawn file from defined folder
                    file_stem = k.split('-')[:-1]
                    # Load Input, Output and Response for every batch in defined folder
                    x = np.load('-'.join(file_stem) + '-Input.npy', mmap_mode='r')  # Trial Activity
                    y = np.load('-'.join(file_stem) + '-Output.npy', mmap_mode='r')  # Participant Response
                    y_loc = np.load('-'.join(file_stem) + '-yLoc.npy', mmap_mode='r')  # Ground Truth
                    # Copy files as they are saved as read-only's
                    new_x = np.copy(x)
                    new_y = np.copy(y)
                    new_y_loc = np.copy(y_loc)
                    # Shift the copied files
                    shiftStepList = [6,12,18,24,30]
                    for shiftSteps in shiftStepList:
                        # Iterate over all trials in the batch
                        for j in range(0,np.size(new_x,0)):
                            for i in range(0,np.size(new_x,1)):
                                # Rotate the modality rings/vectors counter-clockwise
                                rotated_slices_mod1 = np.roll(new_x[j, i, 1:33], shiftSteps)
                                rotated_slices_mod2 = np.roll(new_x[j, i, 33:65], shiftSteps)
                                # Replace the original portion with the rotated slices
                                new_x[j, i, 1:33] = np.array(rotated_slices_mod1)
                                new_x[j, i, 33:65] = np.array(rotated_slices_mod2)
                                # Rotate the output participant response ring counter-clockwise
                                rotated_slices_y = np.roll(new_y[j, i, 1:33], shiftSteps)
                                # Replace the original portion with the rotated slices
                                new_y[j, i, 1:33] = np.array(rotated_slices_y)

                        input_filename = ('-'.join(file_stem) + '-' + 'Rotation_' + str(shiftSteps) + '-Input')
                        output_filename = ('-'.join(file_stem) + '-' + 'Rotation_' + str(shiftSteps) + '-Output')
                        response_filename = ('-'.join(file_stem) + '-' + 'Rotation_' + str(shiftSteps) + '-yLoc')
                        np.save(input_filename, new_x)
                        np.save(output_filename, new_y)
                        np.save(response_filename, new_y_loc)

                        print(f'Rotated batch augmented for task {task} and shift step {shiftSteps}')

        except ValueError as err:
            print(f"ValueError occurred: {err.args}")
        except Exception as e:
            print(f"An error occurred: {e}")


    # WMTask: Factor 32
    # 33.75 (3 steps), 67.5 (6 steps), 101.25 (9 steps), 135 (12 steps), 168.75 (15 steps), 202.5 (18 steps), 236.25 (21 steps),
    # 270 (24 steps), 303.75 (27 steps) and 337.5 (30 steps) degrees
    tasks = ['WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']
    for task in tasks:
        trial_list = glob.glob(os.path.join(trial_dir, task, '*Input.npy'))  # Define list of batches to augment on
        os.chdir(os.path.join(trial_dir, task))  # Define folder to save files in
        try:
            for k in trial_list:
                if k.split('-')[1] != 'month_1':
                    # Split every drawn file from defined folder
                    file_stem = k.split('-')[:-1]
                    # Load Input, Output and Response for every batch in defined folder
                    x = np.load('-'.join(file_stem) + '-Input.npy', mmap_mode='r') # Trial Activity
                    y = np.load('-'.join(file_stem) + '-Output.npy', mmap_mode='r') # Participant Response
                    y_loc = np.load('-'.join(file_stem) + '-yLoc.npy', mmap_mode='r') # Ground Truth
                    # Copy files as they are saved as read-only's
                    new_x = np.copy(x)
                    new_y = np.copy(y)
                    new_y_loc = np.copy(y_loc)
                    # Shift the copied files
                    shiftStepList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
                    for shiftSteps in shiftStepList:
                        # Iterate over all trials in the batch
                        for j in range(0,np.size(new_x,0)):
                            for i in range(0,np.size(new_x,1)):
                                # Rotate the modality rings/vectors counter-clockwise
                                rotated_slices_mod1 = np.roll(new_x[j, i, 1:33], shiftSteps)
                                rotated_slices_mod2 = np.roll(new_x[j, i, 33:65], shiftSteps)
                                # Replace the original portion with the rotated slices
                                new_x[j, i, 1:33] = np.array(rotated_slices_mod1)
                                new_x[j, i, 33:65] = np.array(rotated_slices_mod2)
                                # Rotate the output participant response ring counter-clockwise
                                rotated_slices_y = np.roll(new_y[j, i, 1:33], shiftSteps)
                                # Replace the original portion with the rotated slices
                                new_y[j, i, 1:33] = np.array(rotated_slices_y)

                        input_filename = ('-'.join(file_stem) + '-' + 'Rotation_' + str(shiftSteps) + '-Input')
                        output_filename = ('-'.join(file_stem) + '-' + 'Rotation_' + str(shiftSteps) + '-Output')
                        response_filename = ('-'.join(file_stem) + '-' + 'Rotation_' + str(shiftSteps) + '-yLoc')
                        np.save(input_filename, new_x)
                        np.save(output_filename, new_y)
                        np.save(response_filename, new_y_loc)

                        print(f'Rotated batch augmented for task {task} and shift step {shiftSteps}')

        except ValueError as err:
            print(f"ValueError occurred: {err.args}")
        except Exception as e:
            print(f"An error occurred: {e}")


    # 3: Geometric Mirroring ###############################################################################################
    # AllTask: Factor 2
    # tasks = ['RP', 'RP_Anti']
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']
    for task in tasks:
        trial_list = glob.glob(os.path.join(trial_dir, task, '*Input.npy'))  # Define list of batches to augment on
        os.chdir(os.path.join(trial_dir, task))  # Define folder to save files in
        try:
            for k in trial_list:
                if k.split('-')[1] != 'month_1':
                    # Split every drawn file from defined folder
                    file_stem = k.split('-')[:-1]
                    # Load Input, Output and Response for every batch in defined folder
                    x = np.load('-'.join(file_stem) + '-Input.npy', mmap_mode='r')  # Trial Activity
                    y = np.load('-'.join(file_stem) + '-Output.npy', mmap_mode='r')  # Participant Response
                    y_loc = np.load('-'.join(file_stem) + '-yLoc.npy', mmap_mode='r')  # Ground Truth
                    # Copy files as they are saved as read-only's
                    new_x = np.copy(x)
                    new_y = np.copy(y)
                    new_y_loc = np.copy(y_loc)

                    # Iterate over all trials in the batch
                    for j in range(0, np.size(new_x, 0)):
                        for i in range(0, np.size(new_x, 1)):
                            # Mirror the modality rings/vectors
                            mirrored_slices_mod1 = np.flip(new_x[j, i, 1:33])
                            mirrored_slices_mod2 = np.flip(new_x[j, i, 33:65])
                            # Replace the original portion with the mirrored slices
                            new_x[j, i, 1:33] = np.array(mirrored_slices_mod1)
                            new_x[j, i, 33:65] = np.array(mirrored_slices_mod2)
                            # Mirror the output participant response ring
                            mirrored_slices_y = np.flip(new_y[j, i, 1:33])
                            # Replace the original portion with the mirrored slices
                            new_y[j, i, 1:33] = np.array(mirrored_slices_y)

                        input_filename = ('-'.join(file_stem) + '-' + 'Mirrored' + '-Input')
                        output_filename = ('-'.join(file_stem) + '-' + 'Mirrored' + '-Output')
                        response_filename = ('-'.join(file_stem) + '-' + 'Mirrored' + '-yLoc')
                        np.save(input_filename, new_x)
                        np.save(output_filename, new_y)
                        np.save(response_filename, new_y_loc)

                        print(f'Mirrored batch augmented for task {task} and file {k}')

        except ValueError as err:
            print(f"ValueError occurred: {err.args}")
        except Exception as e:
            print(f"An error occurred: {e}")


    # 4: Randomization of trials within batch ##############################################################################
    # DM/EF/RP-Tasks: Factor 3
    # tasks = ['RP', 'RP_Anti']
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2']
    # Randomize every batch within for three times
    for l in range(3):
        for task in tasks:
            trial_list = glob.glob(os.path.join(trial_dir, task, '*Input.npy')) # Define list of batches to augment on
            os.chdir(os.path.join(trial_dir, task)) # Define folder to save files in
            try:
                for k in trial_list:
                    if k.split('-')[1] != 'month_1':
                        # Split every drawn file from defined folder
                        file_stem = k.split('-')[:-1]
                        # Load Input, Output and Response for every batch in defined folder
                        x = np.load('-'.join(file_stem) + '-Input.npy', mmap_mode='r')  # Trial Activity
                        y = np.load('-'.join(file_stem) + '-Output.npy', mmap_mode='r')  # Participant Response
                        y_loc = np.load('-'.join(file_stem) + '-yLoc.npy', mmap_mode='r')  # Ground Truth
                        # Copy files as they are saved as read-only's
                        new_x = np.copy(x)
                        new_y = np.copy(y)
                        new_y_loc = np.copy(y_loc)
                        # Randomize trials within each batch with a permutation of the original order
                        shape_new_x = new_x.shape
                        permutation = np.random.permutation(shape_new_x[1])
                        # Apply the permutation to the array along the second dimension
                        shuffled_new_x = new_x[:, permutation, :]
                        shuffled_new_y = new_y[:, permutation, :]
                        shuffled_new_y_loc = new_y_loc[:, permutation]

                        input_filename = ('-'.join(file_stem) + '-' + 'Randomization_' + str(l) + '-Input')
                        output_filename = ('-'.join(file_stem) + '-' + 'Randomization_' + str(l) + '-Output')
                        response_filename = ('-'.join(file_stem) + '-' + 'Randomization_' + str(l) + '-yLoc')
                        np.save(input_filename, new_x)
                        np.save(output_filename, new_y)
                        np.save(response_filename, new_y_loc)

                        print(f'Randomized batch augmented for task {task}, file {k} and iteration {l}')

            except ValueError as err:
                print(f"ValueError occurred: {err.args}")
            except Exception as e:
                print(f"An error occurred: {e}")





