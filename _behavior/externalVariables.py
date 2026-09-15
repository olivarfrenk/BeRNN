import numpy as np
import json
import os
# import pickle
from scipy import stats

'''
- Manually created lists of external subject variables 
- Executes explorative correlation analysis with external variables and topological markers over all subjects
'''

# **********************************************************************************************************************
# head: Manually created lists of external variables
# **********************************************************************************************************************
# Read vertically
# Calculate correlation over time points; alternatively: repeated measure correlation
# Demographic variables
age = [22, 23, 21, 31, 33]
sex = [0, 0, 0, 0, 1] # 0: female ; 1: male
profession = [2, 2, 2, 3, 1] # 0: unemployed ; 1: hospital care ; 2: student ; 3: employed ; 4: self-employed
education = [0, 0, 1, 1, 0] # 0: high school ; 1: undergraduate ; 2: graduate ; 3: postgraduate

# info: 1. assesment (01, 02, 03, 04, 05) ##############################################################################
bcats_trail = [80, 37, 55, 101, 56]
bcats_symbol = [61, 95, 100, 94, 90]
bcats_animals = [35, 49, 27, 36, 25]
# z-score for each test
z_trail = (bcats_trail[4] - np.mean(bcats_trail)) / np.std(bcats_trail)
z_symbol = (bcats_symbol[4] - np.mean(bcats_symbol)) / np.std(bcats_symbol)
z_animals = (bcats_animals[4] - np.mean(bcats_animals)) / np.std(bcats_animals)
# more time means worse performance
z_trail *= -1
# final individual z-score
z_score_cognition_bcats = np.mean([z_trail, z_symbol, z_animals])
print(np.round(z_score_cognition_bcats, 3))

# MDD: -0.838
# ASD: 1.173
# HC1: 0.159
# HC2: -0.319
# SCZ: -0.175

# MRT
nback = [58.33, 87.5, 87.5, 72.92, 43.75]
reward = [6/20, 0/20, 8/20, 4/20, 4/20] # fix create accuracy with 20 max
faces = [57/58, 57/58, 57/58, 56/58, 57/58]
flanker = [109/145, 116/145, 114/145, 112/145, 114/145]
# z-score for each test
z_nback = (nback - np.mean(nback)) / np.std(nback)
z_reward = (reward - np.mean(reward)) / np.std(reward)
z_faces = (faces - np.mean(faces)) / np.std(faces)
z_flanker = (flanker - np.mean(flanker)) / np.std(flanker)
# final individual z-score
z_score_cognition_mrt = np.mean([z_nback[0], z_reward[0], z_faces[0], z_flanker[0]])


# info: 2. assesment (01, 02, 03, 04, 05) ##############################################################################
# MRT
nback = [95.83, 91.66, 87.5, 83.33, 83.33]
reward = [4/20, 8/20, 6/20, 6/20, 8/20]
faces = [1, 53/58, 1, 57/58, 1]
flanker = [121/145, 114/145, 113/145, 113/145, 114/145]

# info: 3. assesment (01, 02, 03, 04, 05) ##############################################################################
# MRT
nback = [97.92, 91.66, 89.58, 87.5, 68.75]
reward = [10/20, 6/20, 10/20, 10/20, 4/20]
faces = [1, 54/58, 57/58, 56/58, 57/58]
flanker = [107/145, 115/145, 112/145, 111/145, 128/145]

# info: 4. assesment (01, 02, 03, 05) ##################################################################################
# MRT
nback = [85.42, 97.92, 100, 87.5]
reward = [8/20, 4/20, 10/20, 2/20]
faces = [57/58, 53/58, 57/58, 1]
flanker = [115/145, 116/145, 111/145, 115/145]

# info: 5. assesment (01, 02, 03, 05) ##################################################################################
# BCATS
bcats_trail = [72, 38, 39, 48]
bcats_symbol = [76, 93, 93, 93]
bcats_animals = [47, 36, 37, 32]
# z-score for each test
z_trail = (bcats_trail[3] - np.mean(bcats_trail)) / np.std(bcats_trail)
z_symbol = (bcats_symbol[3] - np.mean(bcats_symbol)) / np.std(bcats_symbol)
z_animals = (bcats_animals[3] - np.mean(bcats_animals)) / np.std(bcats_animals)
# more time means worse performance
z_trail *= -1
# final individual z-score
z_score_cognition_bcats = np.mean([z_trail, z_symbol, z_animals])
print(np.round(z_score_cognition_bcats, 3))

# MDD: -0.588
# ASD: 0.345
# HC1: 0.381
# HC2: NA
# SCZ: -0.139

# MRT
nback = [79.17, 83.33, 97.92, 66.66]
reward = [6/20, 2/20, 8/20, 8/20]
faces = [34/58, 34/58, 1, 1]
flanker = [83/145, 63/145, 113/145, 113/145]
# z-score for each test
z_nback = (nback - np.mean(nback)) / np.std(nback)
z_reward = (reward - np.mean(reward)) / np.std(reward)
z_faces = (faces - np.mean(faces)) / np.std(faces)
z_flanker = (flanker - np.mean(flanker)) / np.std(flanker)
# final individual z-score
z_score_cognition_mrt = np.mean([z_nback[3], z_reward[3], z_faces[3], z_flanker[3]])


# **********************************************************************************************************************
# head: Explorative correlation analysis
# **********************************************************************************************************************
# hitop_general_end = [0.5, 0.395, 0.076, 0.192]
hitop_official_end = [0.574, 0.553, 0.191, 0.447]

age = [22, 23, 21, 33]

z_score_cognition_mrt = [-0.588, 0.345, 0.381, -0.139]
z_score_cognition_bcats = [-0.426, -0.977, 1.052, 0.351]

predictors = {
    # "HiTOP General": hitop_general_end,
    "HiTOP Official": hitop_official_end,
    "Age": age,
    "Z-Score Cognition MRT": z_score_cognition_mrt,
    "Z-Score Cognition BCATS": z_score_cognition_bcats,
}

participants = ["beRNN_01", "beRNN_02", "beRNN_03", "beRNN_05"]
tasks = "4task"
dataType = "highDim_correctOnly"
density_tresholds = ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]

# Initialize a dictionary to store all json data
json_output_data = {}

for density_treshold in density_tresholds:

    clustering_list = []
    modularity_list = []
    # n_modules_list = []
    participation_list = []
    # efficiency_list = []

    for participant in participants:
        meta_dict_path = os.path.join(
            r"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists",
            f"topologicalMarker_dict_beRNN_compare_4task_beRNN_01_highDim_128_hp8_{density_treshold}.json",
        )

        with open(meta_dict_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        clustering_list.append(meta_dict[participant]["avg_clustering"][-1])
        modularity_list.append(meta_dict[participant]["mod_value_sparse"][-1])
        # n_modules_list.append(meta_dict[participant]["n_modules_list"][-1][0])
        participation_list.append(meta_dict[participant]["participation_coefficient"][-1])
        # efficiency_list.append(meta_dict[participant]["global_efficiency_list"][-1][0])

    print("****************************************************")
    print("Explorative correlations for density threshold: ", density_treshold)
    print("****************************************************")

    graph_metrics = {
        "Clustering": clustering_list,
        "Modularity": modularity_list,
        # "N-Modules": n_modules_list,
        "Participation": participation_list,
        # "Efficiency": efficiency_list,
    }

    # Initialize sub-dictionary for the current density threshold
    json_output_data[density_treshold] = {}

    for metric_name, metric_values in graph_metrics.items():
        print(f"\n--- Metrik: {metric_name} ---")

        # Initialize sub-dictionary for the current graph metric
        json_output_data[density_treshold][metric_name] = {}

        for pred_name, pred_values in predictors.items():
            r_val, p_val = stats.pearsonr(metric_values, pred_values)
            print(f"  vs {pred_name:25} -> r = {r_val:6.3f}, p = {p_val:5.3f}")

            # # Save data points (converted to standard float for JSON serialization)
            # json_output_data[density_treshold][metric_name][pred_name] = {
            #     "r": float(r_val),
            #     "p": float(p_val),
            # }

# # save everything to json file
# output_json_path = "../data/correlation_results_12tasks_correctOnly.json"
# with open(output_json_path, "w", encoding="utf-8") as json_file:
#     json.dump(json_output_data, json_file, indent=4)
#
# print(f"\n[SUCCESS] All correlation data saved to: {output_json_path}")


