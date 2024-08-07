########################################################################################################################
# todo: LAB ############################################################################################################
########################################################################################################################

# Error Comparison #####################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
import glob
import os

import Tools
from Network import Model, popvec, get_perf #, tf_popvec

model = 'Model_146_BeRNN_01_Month_2-6'
model_dir = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models' + '\\' + model + '\\'
trial_dir = 'W:\\group_csp\\analyses\\oliver.frank\\Data' + '\\BeRNN_' + model.split('_')[3] + '\\PreprocessedData_wResp_ALL' # todo: Distinguish between test and training data later

# Filter for months model has used for training
monthsConsidered = list(range(int(model.split('_')[-1].split('-')[0]),int(model.split('_')[-1].split('-')[1])+1))
for i in range(0,len(monthsConsidered)): monthsConsidered[i] = 'month_' + str(monthsConsidered[i])
task = 'WM'
# Should only take the files with monthsConsidered
npy_files_Input = glob.glob(os.path.join(trial_dir, task, '*Input.npy'))
filtered_npy_files_Input = [file for file in npy_files_Input if any(month in file for month in monthsConsidered)]

# Load model
hp = Tools.load_hp(model_dir)
model = Model(model_dir, hp=hp)

# todo: Add to Tools later
def get_perf_errorComparison(modelResponse_machineForm, participantResponse_perfEvalForm):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation). Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(modelResponse_machineForm.shape) != 3:
        raise ValueError('modelResponse_machineForm must have shape (Time, Batch, Unit)')
    # Only look at last time points
    participantResponse_perfEvalForm_compressed = participantResponse_perfEvalForm[-1] # todo: size 40,1
    modelResponse_machineForm = modelResponse_machineForm[-1]

    # Fixation and location of y_hat
    modelResponse_machineForm_fix = modelResponse_machineForm[..., 0]
    modelResponse_machineForm_compressed = popvec(modelResponse_machineForm[..., 1:]) # todo: size 40,1

    # Fixating? Correctly saccading?
    fixating = modelResponse_machineForm_fix > 0.5

    original_dist = participantResponse_perfEvalForm_compressed - modelResponse_machineForm_compressed
    dist = np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
    corr_loc = dist < 0.2 * np.pi  # 35 degreee margin around exact correct respond

    # Should fixate?
    should_fix = participantResponse_perfEvalForm_compressed < 0

    # performance
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    return perf, participantResponse_perfEvalForm_compressed, modelResponse_machineForm_compressed

# Initialize count variables Match and Mismatch
errorMatch = 0
errorMisMatch = 0
correctMatch = 0
correctMisMatch = 0
noCount = 0

# Get networkResponse by opening model and let it process on the current trial
with tf.Session() as sess:
    # if load_dir is not None:
    #     model.restore(load_dir)  # complete restore
    # else:
    # Assume everything is restored
    sess.run(tf.global_variables_initializer())
    # # Load model
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    for file in filtered_npy_files_Input:
        # Get responses for one arbitrary trial particpantResponse, groundTruth, networkResponse
        batchInput = np.load(file, allow_pickle=True)
        participantResponse = np.load(os.path.join(trial_dir, task, '-'.join(file.split('\\')[-1].split('-')[:-1]) + '-Output.npy'),allow_pickle=True)
        participantResponse_perfEvalForm = np.load(os.path.join(trial_dir, task, '-'.join(file.split('\\')[-1].split('-')[:-1]) + '-yLoc.npy'),allow_pickle=True)
        groundTruth = np.load(os.path.join(trial_dir, task, '-'.join(file.split('\\')[-1].split('-')[0:5]) + '-Response.npy'),allow_pickle=True)
        # c_mask actually not needed, as we won't train the network here. ONly defined for calling function
        c_mask = np.zeros((participantResponse.shape[0]*participantResponse.shape[1], participantResponse.shape[2]),dtype='float32')
        # Get model response for current batch
        feed_dict = Tools.gen_feed_dict(model, batchInput, participantResponse, c_mask, hp)  # y: participnt response, that gives the lable for what the network is trained for
        c_lsq, c_reg, modelResponse_machineForm = sess.run([model.cost_lsq, model.cost_reg, model.y_hat], feed_dict=feed_dict)
        print('modelResponse for one batch created')


        # Compare modelResponse_machineForm (y_hat_test: model behavior) with participantResponse_perfEvalForm (participant behavior)
        comparisonResults, participantResponse_perfEvalForm_compressed, modelResponse_machineForm_compressed = get_perf_errorComparison(modelResponse_machineForm, participantResponse_perfEvalForm)


        # These are the 32 possible directions a model or participant can indicate to;
        prefDirections_output = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

        # Transform modelResponse_machineForm_compressed into humanForm
        for i in range(modelResponse_machineForm_compressed.shape[0]):
            closest_num = min(prefDirections_output, key=lambda x: abs(x - modelResponse_machineForm_compressed[i]))
            print(closest_num)

            fieldNumber = np.where(prefDirections_output == closest_num)

            mod1_info = batchInput[-1,i,fieldNumber]
            mod2_info = batchInput[-1,i,33+fieldNumber]

            # III: DM
            if task == 'DM' or task == 'DM_Anti':
                # Define modulation dictionaries for specific columns
                mod1Dict = {'lowest': float(0.25), 'low': float(0.5), 'strong': float(0.75), 'strongest': float(1.0)}
                mod2Dict = {'right.png': float(0.25), 'down.png': float(0.5), 'left.png': float(0.75), 'up.png': float(1.0)}
                # Define an output dictionary for specific response values
                outputDict = {'U': 32, 'R': 8, 'L': 24, 'D': 16}
            # III: EF
            if task == 'EF' or task == 'EF_Anti':
                # Define modulation dictionaries for specific columns
                mod1Dict = {'green': float(0.5), 'red': float(1.0)}
                mod2Dict = {'right.png': float(0.2), 'down.png': float(0.4), 'left.png': float(0.6), 'up.png': float(0.8),'X.png': float(1.0)}
                # Define an output dictionary for specific response values
                outputDict = {'U': 32, 'R': 8, 'L': 24, 'D': 16}
            # III: RP
            if task == 'RP' or task == 'RP_Anti' or task == 'RP_Ctx1' or task == 'RP_Ctx2':
                # Define modulation dictionaries for specific columns
                mod1Dict = {'red': 0.08, 'rust': 0.17, 'orange': 0.25, 'amber': 0.33, 'yellow': 0.42, 'lime': 0.5,
                            'green': 0.58, 'moss': 0.66, 'blue': 0.75, 'violet': 0.83, 'magenta': 0.92, 'purple': 1.0}
                mod2Dict = {'triangle.png': float(0.2), 'pentagon.png': float(0.4), 'heptagon.png': float(0.6),
                            'nonagon.png': float(0.8), 'circle.png': float(1.0)}
                # Define an output dictionary for specific response values
                outputDict = {'Image 2': 2, 'Image 4': 4, 'Image 6': 6, 'Image 8': 8, 'Image 10': 10, 'Image 12': 12,
                              'Image 14': 14, 'Image 16': 16, 'Image 18': 18, 'Image 20': 20, 'Image 22': 22, 'Image 24': 24,
                              'Image 26': 26, 'Image 28': 28, 'Image 30': 30, 'Image 32': 32, \
                              'Image 1': 1, 'Image 3': 3, 'Image 5': 5, 'Image 7': 7, 'Image 9': 9, 'Image 11': 11,
                              'Image 13': 13, 'Image 15': 15, 'Image 17': 17, 'Image 19': 19, 'Image 21': 21,
                              'Image 23': 23, 'Image 25': 25, 'Image 27': 27, 'Image 29': 29, 'Image 31': 31}
            # III: WM
            if task == 'WM' or task == 'WM_Anti' or task == 'WM_Ctx1' or task == 'WM_Ctx2':
                # Define modulation dictionaries for specific columns
                mod1Dict = {'red': 0.08, 'rust': 0.17, 'orange': 0.25, 'amber': 0.33, 'yellow': 0.42, 'lime': 0.5, 'green': 0.58, \
                            'moss': 0.66, 'blue': 0.75, 'violet': 0.83, 'magenta': 0.92, 'purple': 1.0}
                mod2Dict = {'triangle.png': float(0.2), 'pentagon.png': float(0.4), 'heptagon.png': float(0.6),
                            'nonagon.png': float(0.8), 'circle.png': float(1.0)}
                # Define an output dictionary for specific response values
                outputDict_WM = {'Image 1': 1, 'Image 2': 2, 'Image 3': 3, 'Image 4': 4, 'Image 5': 5, 'Image 6': 6, 'Image 7': 7, \
                                 'Image 8': 8, 'Image 9': 9, 'Image 10': 10, 'Image 11': 11, 'Image 12': 12, 'Image 13': 13,\
                                 'Image 14': 14, 'Image 15': 15, 'Image 16': 16, 'Image 17': 17, 'Image 18': 18, 'Image 19': 19,\
                                 'Image 20': 20, 'Image 21': 21, 'Image 22': 22, 'Image 23': 23, 'Image 24': 24, 'Image 25': 25, 'Image 26': 26,\
                                 'Image 27': 27, 'Image 28': 28, 'Image 29': 29, 'Image 30': 30, 'Image 31': 31, 'Image 32': 32}
                outputDict_WM_Ctx = {'object-1591': 8, 'object-1593': 8, 'object-1595': 8, 'object-1597': 8,\
                                     'object-2365': 8, 'object-2313': 8, 'object-2391': 8, 'object-2339': 8,\
                                     'object-1592': 24, 'object-1594': 24, 'object-1596': 24, 'object-1598': 24,\
                                     'object-2366': 24, 'object-2314': 24, 'object-2392': 24, 'object-2340': 24}

                try:

            # Translate batchInput[i][fieldNumber] on both modalities to one humanForm Response
            # modelResponse_machineForm_compressed


        # get_perf
        # concatenate batch with compressed response 40 ndarrays
        # Only look at last time points


        # # for entries in perf:
        # for result in comparisonResults:
        #     if result == 1:
        #         if groundTruth[0] == groundTruth[1]: # if participantResponse == correctResponse
        #             correctMatch += 1
        #         elif groundTruth[0] != groundTruth[1]: # if participantResponse == correctResponse
        #             correctMatch += 1
        #
        #
        # # Helper function to calculate the distance for circular activation ####################################################
        # def get_dist(original_dist):
        #     return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
        #
        # # Helper function to calculate the circular activation
        # def add_x_loc(x_loc, pref):
        #     dist = get_dist(x_loc - pref)
        #     dist /= np.pi / 8
        #     return 0.8 * np.exp(-dist ** 2 / 2)


        # if entry = 1: count matchCorrect or matchError
        # save response in human form

        # if entry = 0: check the concatenated entries for their direction
        # check on which stim they direct on input trial
        # apply error analysis and count mismatchCorrect or mismatchError
        # save response in human form and error type



        # Following vector has the collapsed population vector for every time point in sequence for every trial
        # popVec_modelResponse_machineForm = popvec(modelResponse_machineForm[..., 1:])
        # print(popVec_modelResponse_machineForm)



        # Compare
        # todo: Get the value representing the response direction of the model for one trail (in the complete batch)
        # todo: and get the field representing it in the human form to find in the InputTrial to which response it took



















        # todo: Categorize modelResponse into one discrete response option

        # ##############################################################################################################
        # # todo: Compare the different responses - propably somehow over the evaluation function
        # responseClass_participantResponse = compare(participantResponse, groundTruth)
        # responseClass_networkResponse = compare(networkResponse, groundTruth)
        #
        # todo: Append all the files to a dataframe
        # # Count Match/Mismatch Table
        # if responseClass_participantResponse[0] == 'Error' and responseClass_networkResponse[0] == 'Error' and responseClass_participantResponse[1] \
        #     == networkResponse[1]:
        #     errorMatch += 1
        # elif responseClass_participantResponse[0] == 'Error' and responseClass_networkResponse[0] == 'Error' and responseClass_participantResponse[1] \
        #     != networkResponse[1]:
        #     errorMisMatch += 1
        # elif responseClass_participantResponse[0] == 'Correct' and responseClass_networkResponse[0] == 'Correct' and responseClass_participantResponse[1] \
        #     == networkResponse[1]:
        #     correctMatch += 1
        # elif responseClass_participantResponse[0] == 'Correct' and responseClass_networkResponse[0] == 'Correct' and responseClass_participantResponse[1] \
        #     != networkResponse[1]:
        #     correctMisMatch += 1
        # else:
        #     noCount += 1
