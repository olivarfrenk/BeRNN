import os
from nilearn.glm.first_level import first_level_from_bids
from bids.layout import BIDSLayout
from nilearn.plotting import plot_design_matrix
from nilearn.interfaces.bids import save_glm_to_bids
import gc
import matplotlib.pyplot as plt
import re
import pandas as pd
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
from nilearn.glm import threshold_stats_img

'''
Roles:
- Compute first level models for brain dataset using nilearn
'''

###############################################################################
## Fixed settings for all tasks
###############################################################################
data_dir = '/zi/flstorage/group_csp/in_house_datasets/bernn/mri/rawdata/'
# data_dir = r'W:\group_csp\in_house_datasets\bernn\mri\rawdata'
derivatives_folder = "/zi/flstorage/group_csp/in_house_datasets/bernn/mri/derivatives/fmriprep_25.2.3/"
# derivatives_folder = r"W:\group_csp\in_house_datasets\bernn\mri\derivatives\fmriprep_25.2.3"
space_label = "MNI152NLin2009cAsym"
t_r = 0.8
hrf_model = 'spm'
smoothing_fwhm = 6
n_jobs = 6
verbose = 1
output_dir = r'W:\group_csp\in_house_datasets\bernn\mri\derivatives\firstlevel_nilearn_0.11.1'
output_dir = r'/zi/flstorage/group_csp/in_house_datasets/bernn/mri/derivatives/firstlevel_nilearn_0.11.1'
minimize_memory = False
confounds_selection = ['trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
                       'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
                       'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
                       'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
                       'rot_y','rot_y_derivative1','rot_y_derivative1_power2','rot_y_power2',
                       'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2']
standardize = True

###############################################################################
## Specific settings for this task
###############################################################################

# task_label = "nback"
# contrasts = {"zeroback":"zeroBack",
#              "twoback":"twoBack",
#              "twobackGtzeroback":"twoBack - zeroBack"}

task_label = "reward"
contrasts = {
    # 1. Die einzelnen Entscheidungs-Momente (falls du sie separat ansehen willst)
    "decision_money": "decision_action_money",
    "decision_verbal": "decision_action_verbal",
    "decision_control": "decision_action_control",

    # 2. Deine gewünschten Kontraste für den Moment der Entscheidung vs. Kontrolle:
    # (Das ersetzt dein altes "UCS - no UCS")
    "UCSGnoUCS": "decision_action_money - decision_action_control",
    "noUCSGUCS": "decision_action_control - decision_action_money",

    # 3. Optional: Die Planungsphasen (Antizipation), falls du sie auch auswerten willst:
    "anticipation_money_vs_control": "anticipation_money - anticipation_control"
}

# task_label = "faces"
# contrasts = {"faces":"faces",
#              "forms":"forms",
#              "facesGtforms":"faces - forms"}

# task_label = 'flanker'
# contrasts = {
#     'congruent': '(congruentleft + congruentright) / 2',
#     'incongruent': '(incongruentleft + incongruentright) / 2',
#     'neutral': '(neutralleft + neutralright) / 2',

#     'incongruent_Gt_congruent': '((incongruentleft + incongruentright) / 2) - ((congruentleft + congruentright) / 2)',
#     'congruent_Gt_incongruent': '((congruentleft + congruentright) / 2) - ((incongruentleft + incongruentright) / 2)',

#     'incongruent_Gt_neutral': '((incongruentleft + incongruentright) / 2) - ((neutralleft + neutralright) / 2)',
#     'congruent_Gt_neutral': '((congruentleft + congruentright) / 2) - ((neutralleft + neutralright) / 2)'
# }



###############################################################################
## Get task data for all subjects
###############################################################################

# get only subjects that have completed this task
subjects = BIDSLayout(root=data_dir).get_subjects(task=task_label,suffix='events',extension='tsv')

(
    models,
    run_imgs,
    events,
    confounds,
) = first_level_from_bids(
    dataset_path=data_dir,
    task_label=task_label,
    space_label=space_label,
    sub_labels=subjects,
    derivatives_folder=derivatives_folder,
    t_r=t_r,
    hrf_model=hrf_model,
    smoothing_fwhm=smoothing_fwhm,
    minimize_memory=minimize_memory,
    standardize=standardize,
    n_jobs=n_jobs,
    verbose=verbose,

)

###############################################################################
## Create a function that preprocesses the events.tsv
###############################################################################

def process_events(events_subject,task_label):
    
    # make a copy of the input
    df = events_subject.copy()


    if task_label == 'nback':
        # nilearn doesn't like it when variables start with digits
        df['trial_type'] = df['trial_type'].replace({'0back': 'zeroBack', '2back': 'twoBack'})

        # Temporarily keep only 'zeroBack' and 'twoBack', forward-fill those
        temp = df['trial_type'].where(df['trial_type'].isin(['zeroBack', 'twoBack'])).ffill()

        # Only 'num*' rows that are not 'zeroBack' or 'twoBack'
        mask = df['trial_type'].str.startswith('num', na=False)
        df.loc[mask, 'trial_type'] = temp[mask]


    if task_label == 'faces':
        # Keep only 'MatchForms' and 'MatchFaces' and forward-fill them
        temp = df['trial_type'].where(df['trial_type'].isin(['MatchForms', 'MatchFaces'])).ffill()

        # Mask: intermediate rows starting with 'boy', 'girl', or 'Form'
        mask = df['trial_type'].str.startswith(('boy', 'girl', 'Form'), na=False)

        # Replace only those masked rows
        df.loc[mask, 'trial_type'] = temp[mask]

        # Cosmetics
        df['trial_type'] = df['trial_type'].map({'MatchForms': 'forms', 'MatchFaces': 'faces'})


    if task_label == 'flanker':
        rows = []
        flanker_targets = ['congruentleft', 'incongruentleft', 'congruentright', 'incongruentright', 'neutralleft',
                           'neutralright']

        for i, row in df.iterrows():
            if row['trial_type'] in flanker_targets:

                if i > 0 and df.loc[i - 1, 'trial_type'] == 'crosshair':
                    prev_row = df.loc[i - 1]
                    rows.append({
                        "onset": prev_row['onset'],
                        "duration": row['onset'] - prev_row['onset'],
                        "trial_type": "fixation_crosshair"
                    })

                rows.append({
                    "onset": row['onset'],
                    "duration": row['duration'],
                    "trial_type": row['trial_type']
                })

        df = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


    if task_label == 'reward':
        rows = []

        for i, row in df.iterrows():
            if row['trial_type'] in ['wCSp', 'vCSp', 'CSm']:

                trial = df.loc[i + 1:]

                target_rows = trial[trial['trial_type'].isin(['UCS', 'no UCS'])]
                if target_rows.empty:
                    continue
                target = target_rows.iloc[0]

                anticip_duration = target['onset'] - row['onset']

                if row['trial_type'] == 'wCSp':
                    anticip_type = 'anticipation_money'
                elif row['trial_type'] == 'vCSp':
                    anticip_type = 'anticipation_verbal'
                else:
                    anticip_type = 'anticipation_control'  # CSm

                rows.append({
                    "onset": row['onset'],
                    "duration": anticip_duration,
                    "trial_type": anticip_type
                })

                if target['trial_type'] == 'UCS':
                    if row['trial_type'] == 'wCSp':
                        target_type = 'decision_action_money'
                    elif row['trial_type'] == 'vCSp':
                        target_type = 'decision_action_verbal'
                else:
                    target_type = 'decision_action_control'

                rows.append({
                    "onset": target['onset'],
                    "duration": target['duration'],
                    "trial_type": target_type
                })

        df = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)

    return df

###############################################################################
## Run first level models for this task
###############################################################################

def clear_model(model):
    """Delete all model instance attributes and force garbage collection."""
    for key in list(vars(model).keys()):
        try:
            delattr(model, key)
        except Exception:
            pass  # some descriptors or read-only attrs
    gc.collect()

for idx in range(len(subjects)):

    # get everything for this subject
    model_subject = models[idx]
    run_imgs_subject = run_imgs[idx][0]
    events_subject = events[idx][0]
    confounds_subject = confounds[idx][0]
    
    # Optional: set brain mask from frmiprep instead of auto-computation
    # of brain mask 
    mask_img = run_imgs_subject.replace('preproc_bold','brain_mask')
    model_subject.mask_img = mask_img
    
    # process events dataframe for this task
    events_subject_preproc = process_events(events_subject, task_label)
    
    # reduce to confounds of choice and fill NaNs for the first time point with 0s
    # https://github.com/nilearn/nilearn/issues/5738
    confounds_subject = confounds_subject.loc[:,confounds_selection]
    confounds_subject.iloc[0] = confounds_subject.iloc[0].fillna(0)
    
    # fit model
    model_subject.fit(run_imgs_subject,events_subject_preproc,confounds_subject)
    
    # compute contrasts and save results (will also create html report)
    subject_label = model_subject.subject_label
    output_dir_subject = os.path.join(output_dir,task_label,subject_label)
    os.makedirs(output_dir_subject)
    save_glm_to_bids(model_subject,contrasts,out_dir=output_dir_subject,prefix=subject_label)
    
    # clear memory so we don't get RAM overflow
    clear_model(model_subject)
    plt.close('all')


