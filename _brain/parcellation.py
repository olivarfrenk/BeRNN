import nibabel as nib
import pandas as pd
import numpy as np
from nilearn.image import resample_to_img, new_img_like
from nilearn.maskers import NiftiLabelsMasker
import os
import re

from _analysis import clustering

'''
Roles:
- Computes brain parcellations of global and subnetworks
'''

# Prepare atlas ******************************************************************
# Define paths to your already existing local atlas files
schaefer_path = r'C:\Users\oliver.frank\nilearn_data\schaefer_2018\Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
ho_path = r'C:\Users\oliver.frank\nilearn_data\harvard_oxford\HarvardOxford-sub-maxprob-thr50-2mm.nii.gz'

# Load images directly via Nibabel instead of fetching them online
schaefer_img = nib.load(schaefer_path)
ho_img = nib.load(ho_path)

# Geometry adjustment (Resample Harvard-Oxford to Schaefer geometry)
ho_resampled = resample_to_img(ho_img, schaefer_img, interpolation='nearest')
schaefer_data = schaefer_img.get_fdata().astype(int)
ho_data = ho_resampled.get_fdata().astype(int)

# Isolate subcortical regions and shift their ID indices
subcortical_mask = (ho_data > 0) & (schaefer_data == 0)
combined_data = schaefer_data.copy()
combined_data[subcortical_mask] = ho_data[subcortical_mask] + 1000

# Create NIfTI image object and the labels masker
combined_atlas_img = new_img_like(schaefer_img, combined_data)
masker = NiftiLabelsMasker(labels_img=combined_atlas_img, resampling_target='labels', standardize=False)
# Prepare atlas ******************************************************************


taskList = ['faces', 'flanker', 'nback', 'reward']

for task in taskList:
    directory_task = rf'W:\group_csp\in_house_datasets\bernn\mri\derivatives\firstlevel_nilearn_0.11.1\{task}'
    subjectList = os.listdir(directory_task)

    for subject in subjectList:
        directory_subject = os.path.join(directory_task, subject)
        contrast_dict = {
            'faces_contrast': f'{subject}_contrast-facesgtforms_stat-z_statmap.nii.gz',

            # flanker_contrast: f'{subject}_contrast-congruentGtIncongruent_stat-z_statmap.nii.gz',
            # flanker_contrast: f'{subject}_contrast-congruentGtNeutral_stat-z_statmap.nii.gz',
            'flanker_contrast': f'{subject}_contrast-incongruentGtCongruent_stat-z_statmap.nii.gz',
            # flanker_contrast: f'{subject}_contrast-incongruentGtNeutral_stat-z_statmap.nii.gz',

            'nback_contrast': f'{subject}_contrast-twobackgtzeroback_stat-z_statmap.nii.gz',

            'reward_contrast': f'{subject}_contrast-anticipationMoneyVsControl_stat-z_statmap.nii.gz',
            # reward_contrast: f'{subject}_contrast-noucsgucs_stat-z_statmap.nii.gz',
            # reward_contrast: f'{subject}_contrast-ucsgnoucs_stat-z_statmap.nii.gz'
        }

        print(subject)
        print(task)

        contrast_file = contrast_dict[f'{task}_contrast']
        directory_contrast = os.path.join(directory_subject, contrast_file)

        parcellated_array = masker.fit_transform(directory_contrast).flatten()
        np.save(os.path.join(directory_subject, f'{subject}_{task}_contrast_parcellation.npy'), parcellated_array)

        print(f"Parcellation sucessful. Final number of regions: {len(parcellated_array)}")



# info. Concatenate the parcellated task vectors along each other ***************************
schaefer_path = r'C:\Users\oliver.frank\nilearn_data\schaefer_2018\Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
schaefer_txt_file = r'C:\Users\oliver.frank\nilearn_data\schaefer_2018\Schaefer2018_1000Parcels_7Networks_order.txt'
ho_path = r'C:\Users\oliver.frank\nilearn_data\harvard_oxford\HarvardOxford-sub-maxprob-thr50-2mm.nii.gz'

# Load the raw 3D images directly via nibabel
schaefer_img = nib.load(schaefer_path)
ho_img = nib.load(ho_path)

# Extract voxel matrices directly without resampling, since grids already match
schaefer_data = schaefer_img.get_fdata().astype(int)
ho_data = ho_img.get_fdata().astype(int)

# Combine them: shift HO IDs by +1000 where Schaefer has no cortical labels (0)
subcortical_mask = (ho_data > 0) & (schaefer_data == 0)
combined_data = schaefer_data.copy()
combined_data[subcortical_mask] = ho_data[subcortical_mask] + 1000

print("Building atlas label lookups...")
# Load the official Schaefer region names to map IDs to networks
schaefer_labels = pd.read_csv(schaefer_txt_file, sep='\t', header=None, names=['label_name'])
schaefer_labels['atlas_id'] = schaefer_labels.index + 1

# Complete ordered list for HarvardOxford-sub-maxprob-thr50-2mm.nii.gz (21 regions)
ho_labels_ordered = [
    'Left-Cerebral-White-Matter',  # ID: 1 (1001) -> Ignore
    'Left-Cerebral-Cortex',        # ID: 2 (1002) -> Ignore
    'Left-Lateral-Ventricle',      # ID: 3 (1003) -> Ignore
    'Left-Thalamus',               # ID: 4 (1004)
    'Left-Caudate',                # ID: 5 (1005)
    'Left-Putamen',                # ID: 6 (1006)
    'Left-Pallidum',               # ID: 7 (1007)
    'Brain-Stem',                  # ID: 8 (1008)
    'Left-Hippocampus',            # ID: 9 (1009)
    'Left-Amygdala',               # ID: 10 (1010)
    'Left-Accumbens',              # ID: 11 (1011)
    'Right-Cerebral-White-Matter', # ID: 12 (1012) -> Ignore
    'Right-Cerebral-Cortex',       # ID: 13 (1013) -> Ignore
    'Right-Lateral-Ventricle',     # ID: 14 (1014) -> Ignore
    'Right-Thalamus',              # ID: 15 (1015)
    'Right-Caudate',               # ID: 16 (1016)
    'Right-Putamen',               # ID: 17 (1017)
    'Right-Pallidum',              # ID: 18 (1018)
    'Right-Hippocampus',           # ID: 19 (1019)
    'Right-Amygdala',              # ID: 20 (1020)
    'Right-Accumbens'              # ID: 21 (1021)
]

# Map the local HO labels to IDs starting from 1001
ho_mapping = [{'label_name': name, 'atlas_id': idx + 1001} for idx, name in enumerate(ho_labels_ordered)]
ho_labels_df = pd.DataFrame(ho_mapping)

# Combine both into a single master lookup table (Exactly 1021 rows)
master_labels = pd.concat([schaefer_labels, ho_labels_df], ignore_index=True)

# info. define subnetwork **********************************************************************************************
# Options: '', 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'
# SubCort: 'Thalamus|Caudate|Putamen|Pallidum|Brain-Stem|Hippocampus|Amygdala|Accumbens'
subNetwork_string = 'Default'

# Find out exactly which IDs are actually present in your NiftiLabelsMasker via combined_data
actual_atlas_ids = sorted([int(x) for x in np.unique(combined_data) if x > 0])
print(f"Total active parcels detected in your generated 3D NIfTI image: {len(actual_atlas_ids)}")

# Numerical list of Harvard-Oxford IDs we want to IGNORE (White matter, Cortex, Ventricles)
ho_ignore_ids = [1001, 1002, 1003, 1012, 1013, 1014]

network_indices = []
if subNetwork_string == '':
    save_suffix = 'FullBrain'
elif subNetwork_string == 'Thalamus|Caudate|Putamen|Pallidum|Brain-Stem|Hippocampus|Amygdala|Accumbens':
    save_suffix = 'SubCort'
else:
    save_suffix = subNetwork_string

# Loop through the actual array layout using pure numbers for total stability
for array_position_idx, atlas_id in enumerate(actual_atlas_ids):

    if subNetwork_string == '':
        # Full Brain: Take everything
        network_indices.append(array_position_idx)

    elif subNetwork_string == 'Thalamus|Caudate|Putamen|Pallidum|Brain-Stem|Hippocampus|Amygdala|Accumbens':
        # Subcortical only: Must be a shifted HO ID (> 1000) AND not in our ignore list
        if atlas_id > 1000 and atlas_id not in ho_ignore_ids:
            network_indices.append(array_position_idx)

    else:
        # Fallback for standard cortical networks ('Vis', 'Default', etc.) via string matching
        matching_row = master_labels[master_labels['atlas_id'] == atlas_id]
        if not matching_row.empty:
            label_name = str(matching_row['label_name'].values[0])
            if subNetwork_string.lower() in label_name.lower():
                network_indices.append(array_position_idx)

print(f"Active network: '{save_suffix}' | Safely selected: {len(network_indices)} indices without out-of-bounds risks.")

taskList = ['reward', 'flanker', 'faces', 'nback']  # same order as for ANNs
directory_rdm = r'W:\group_csp\analyses\oliver.frank\_brainModels\functional_matrices_rdm'
os.makedirs(directory_rdm, exist_ok=True)

directory_task_list = rf'W:\group_csp\in_house_datasets\bernn\mri\derivatives\firstlevel_nilearn_0.11.1\faces'  # always the same list of participants
subjectList = os.listdir(directory_task_list)

for subject in subjectList:
    task_vector_list = []
    skip_subject = False

    for task in taskList:
        directory_task = rf'W:\group_csp\in_house_datasets\bernn\mri\derivatives\firstlevel_nilearn_0.11.1\{task}'
        directory_subject = os.path.join(directory_task, subject)
        file_path = os.path.join(directory_subject, f'{subject}_{task}_contrast_parcellation.npy')

        # Check if file exists before trying to load it
        if not os.path.exists(file_path):
            print(f"Warning: Missing data for {subject} in task {task}. Skipping subject.")
            skip_subject = True
            break

        task_vector = np.load(file_path)

        # Apply network masking logic to the vector using our dynamically safe indices
        subnetwork_vector = task_vector[network_indices]
        task_vector_list.append(subnetwork_vector)

    # Safety break to skip RDM calculations if a subject is missing a task
    if skip_subject or len(task_vector_list) != len(taskList):
        continue

    # Stack your 4 tasks together vertically: Shape is (4, Number_of_Selected_Parcels)
    averageVector_list_stacked = np.stack(task_vector_list, axis=0)

    # Compute RDM matrix
    rdm_metric = 'cosine'
    rdm, rdm_vector = clustering.compute_rdm(averageVector_list_stacked.T, rdm_metric)

    # Save the final matrix using the correct network suffix
    output_filename = f'{subject}_rdm_{rdm_metric}_{save_suffix}_contrast.npy'
    np.save(os.path.join(directory_rdm, output_filename), rdm)

print("\nRDM processing loop completed successfully!")
# info. Concatenate the parcellated task vectors along each other ***************************


# Define the path to your RDM directory
directory_rdm = r'W:\group_csp\analyses\oliver.frank\_brainModels\functional_matrices_rdm'

if not os.path.exists(directory_rdm):
    print(f"Error: The directory does not exist: {directory_rdm}")
else:
    print(f"Scanning folder: {directory_rdm}\n")

    rename_count = 0

    # Loop through all files in the directory
    for filename in os.listdir(directory_rdm):
        # Only process files that contain "SNIP"
        if "SNIP" in filename:

            # Regex to capture:
            # Group 1: Subject ID (9 characters starting with SNIP)
            # Group 2: Session/Recording number (2 digits)
            # Group 3: Metric (e.g., cosine)
            # Group 4: The custom subnetwork and contrast suffix
            match = re.match(r"^(SNIP[A-Z0-9]{5})(\d{2})_rdm_([a-z]+)_(.*)\.npy$", filename)

            if match:
                subject_id = match.group(1)  # e.g., SNIP6IECX
                session_num = match.group(2)  # e.g., 01
                metric = match.group(3)  # e.g., cosine
                network_suffix = match.group(4)  # e.g., Cont_contrast or SubCort_contrast

                # Reconstruct name: sub-SNIP6IECX_ses-01_rdm_cosine_Cont_contrast.npy
                new_filename = f"sub-{subject_id}_ses-{session_num}_rdm_{metric}_{network_suffix}.npy"

                # Full paths for renaming
                old_file_path = os.path.join(directory_rdm, filename)
                new_file_path = os.path.join(directory_rdm, new_filename)

                # Rename the file
                os.rename(old_file_path, new_file_path)

                print(f"Renamed: {filename} -> {new_filename}")
                rename_count += 1
            else:
                print(f"Skipped (Pattern mismatch): {filename}")

    print(f"\nDone! Successfully renamed {rename_count} files.")


