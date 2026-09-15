# ======================================================================================
# Linear Mixed Models - Separate Pairwise Modelset Comparisons
# ======================================================================================
if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dplyr, tidyr, lme4, lmerTest, emmeans, ggplot2, performance, patchwork, scales)

# --- Configuration & Path Setup ---
topMarker_list <- c('avg_clustering', 'mod_value_sparse', 'participation_coefficient')
densities <- c('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0')
folder <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/__topologicalMarker_pValue_lists"

modelsets <- c('topologicalMarker_dict_beRNN_compare_4task_beRNN_01_highDim_128_hp8_', 
               'topologicalMarker_dict_beRNN_compare_4task_beRNN_01_highDim_correctOnly_128_hp8',
               'topologicalMarker_dict_brain_')

# Initialize separate data frames for Analysis A (highDim vs brain)
q1_res_A <- data.frame(); q2_res_A <- data.frame(); posthoc_A <- data.frame()

# Initialize separate data frames for Analysis B (correctOnly vs brain)
q1_res_B <- data.frame(); q2_res_B <- data.frame(); posthoc_B <- data.frame()

# --- JSON Parsing Function ---
parse_json_to_df <- function(file_path, group_label) {
  if (!file.exists(file_path)) return(NULL)
  data <- fromJSON(file_path)
  rows <- list()
  for (p_id in names(data)) {
    for (marker_raw in names(data[[p_id]])) {
      if (marker_raw %in% topMarker_list) {
        values <- data[[p_id]][[marker_raw]][1:5]
        for (m_idx in seq_along(values)) {
          rows[[length(rows) + 1]] <- data.frame(
            Participant = p_id, Group = group_label, Pair_ID = as.factor(m_idx), 
            Marker_Type = marker_raw, Marker_Value = values[m_idx]
          )
        }
      }
    }
  }
  return(bind_rows(rows))
}


# --- Shared Processing Engine ---
run_lmm_analysis <- function(df_paired, density) {
  df_paired <- df_paired %>% mutate(across(c(Participant, Group, Marker_Type, Pair_ID), as.factor))
  
  m_q1 <- lmer(Marker_Value ~ Group * Marker_Type + (1 | Participant/Pair_ID), data = df_paired)
  is_singular <- isSingular(m_q1)
  
  anova_df <- as.data.frame(anova(m_q1, type = "3"))
  p_col <- grep("Pr", names(anova_df), value = TRUE)
  
  p_group <- if("Group" %in% rownames(anova_df)) anova_df["Group", p_col] else NA
  p_inter <- if("Group:Marker_Type" %in% rownames(anova_df)) anova_df["Group:Marker_Type", p_col] else NA
  
  curr_icc <- 0
  if (!is_singular) { 
    icc_obj <- performance::icc(m_q1)
    curr_icc <- as.numeric(icc_obj$ICC_adjusted) 
  }
  
  q1_out <- data.frame(Density = as.numeric(density), P_Value_Group = p_group, P_Value_Interaction = p_inter, ICC = curr_icc, Singular = is_singular)
  ph_out <- as.data.frame(emmeans(m_q1, pairwise ~ Group | Marker_Type)$contrasts) %>% mutate(Density = as.numeric(density))

  ranova_res <- as.data.frame(ranova(m_q1))
  # extract p value for random effect 
  p_re <- if(nrow(ranova_res) > 1) ranova_res[2, "Pr(>Chisq)"] else NA 
  q2_out <- data.frame(Density = as.numeric(density), P_Random_Effect = p_re)
  
  return(list(q1 = q1_out, ph = ph_out, q2 = q2_out))
}


# --- Main Iteration Loop ---
for (density in densities) {
  all_files <- list.files(folder, full.names = TRUE)
  
  file_1 <- all_files[grep(paste0(modelsets[1], ".*", density, "\\.json$"), all_files)]
  file_2 <- all_files[grep(paste0(modelsets[2], ".*", density, "\\.json$"), all_files)]
  file_3 <- all_files[grep(paste0(modelsets[3], ".*", density, "\\.json$"), all_files)]
  
  df_1 <- if(length(file_1) > 0) parse_json_to_df(file_1[1], "highDim") else NULL
  df_2 <- if(length(file_2) > 0) parse_json_to_df(file_2[1], "highDim_correctOnly") else NULL
  df_3 <- if(length(file_3) > 0) parse_json_to_df(file_3[1], "brain") else NULL
  
  # --- Analysis A: highDim vs brain ---
  if (!is.null(df_1) && !is.null(df_3)) {
    res_A <- run_lmm_analysis(bind_rows(df_1, df_3), density)
    q1_res_A <- rbind(q1_res_A, res_A$q1); posthoc_A <- rbind(posthoc_A, res_A$ph); q2_res_A <- rbind(q2_res_A, res_A$q2)
  }
  
  # --- Analysis B: highDim_correctOnly vs brain ---
  if (!is.null(df_2) && !is.null(df_3)) {
    res_B <- run_lmm_analysis(bind_rows(df_2, df_3), density)
    q1_res_B <- rbind(q1_res_B, res_B$q1); posthoc_B <- rbind(posthoc_B, res_B$ph); q2_res_B <- rbind(q2_res_B, res_B$q2)
  }
  print(paste("Density", density, "completed."))
}

# ======================================================================================
# Reusable Plotting Function for Separated Results
# ======================================================================================
generate_plots <- function(q1_res, posthoc_res, q2_res, analysis_title) {
  
  # 1. Fixed Effects Line Plot
  q1_long <- q1_res %>%
    pivot_longer(cols = c(P_Value_Group, P_Value_Interaction), names_to = "Effect", values_to = "P") %>%
    mutate(Effect = recode(Effect, "P_Value_Group" = "Group Main Effect", "P_Value_Interaction" = "Interaction"))
  
  p1 <- ggplot(q1_long, aes(x = Density, y = P, color = Effect, group = Effect)) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", alpha = 0.6) +
    geom_line(size = 1.2) + geom_point(size = 3) +
    scale_y_log10(breaks = 10^seq(0, -40, by = -10), labels = label_log()) +
    scale_x_continuous(breaks = seq(0.1, 1.0, by = 0.1)) +
    coord_cartesian(ylim = c(1e-40, 1)) +
    labs(title = paste("Significance Profile:", analysis_title), x = "Density", y = "p-value (Log Scale)") +
    theme_bw() + theme(legend.position = "bottom", aspect.ratio = 0.6)
  
  # 2. Combined Mean Diff & ICC Plot
  sub_theme <- theme_bw() + theme(aspect.ratio = 0.3, panel.grid.minor = element_blank(), axis.title = element_text(size = 9))
  
  p_diff <- ggplot(posthoc_res, aes(x = Density, y = estimate, color = Marker_Type)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    geom_line(size = 1) + geom_point(size = 2) +
    scale_x_continuous(breaks = seq(0.1, 1, 0.1), labels = NULL) +
    labs(title = paste("Mean Differences:", analysis_title), y = "Estimate", x = NULL) + sub_theme
  
  p_icc <- ggplot(q1_res, aes(x = Density, y = ICC)) + 
    geom_area(fill = "steelblue", alpha = 0.2) + geom_line(color = "steelblue", size = 1) +
    scale_y_continuous(limits = c(0, 1)) + scale_x_continuous(breaks = seq(0.1, 1, 0.1)) +
    labs(title = "Within-Participant ICC Consistency", y = "ICC", x = "Density") + sub_theme
  
  p2 <- (p_diff / p_icc) + plot_layout(guides = "collect")
  
  # 3. Random Effects
  p3 <- ggplot(q2_res, aes(x = Density, y = P_Random_Effect)) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
    geom_line(size = 1.2, color = "royalblue") + geom_point(size = 3, color = "royalblue") +
    scale_y_log10(breaks = 10^seq(0, -15, by = -5), labels = label_log()) +
    scale_x_continuous(breaks = seq(0.1, 1.0, by = 0.1)) + coord_cartesian(ylim = c(1e-15, 1)) +
    labs(title = paste("Random Effects (Ranova):", analysis_title), x = "Density", y = "p-value (Log Scale)") +
    theme_bw() + theme(aspect.ratio = 0.6)
  
  print(p1); print(p2); print(p3)
}

# --- Generate All Visualizations Individually ---
message("\n--- Plotting Analysis A (highDim vs brain) ---")
generate_plots(q1_res_A, posthoc_A, q2_res_A, "highDim vs Brain")

message("\n--- Plotting Analysis B (highDim_correctOnly vs brain) ---")
generate_plots(q1_res_B, posthoc_B, q2_res_B, "highDim_correctOnly vs Brain")


# ======================================================================================
# Console-export for table
# ======================================================================================
print_table_data <- function(q1_res, q2_res, analysis_title) {
  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  cat("  TABELLEN-DATEN FÜR:", toupper(analysis_title), "\n")
  cat(paste(rep("=", 60), collapse = ""), "\n\n")

  table_df <- q1_res %>%
    left_join(q2_res, by = "Density") %>%
    select(Density, P_Value_Group, P_Value_Interaction, P_Random_Effect, ICC, Singular) %>%
    mutate(
      # Runden für schönere Darstellung in Tabellen
      P_Value_Group       = sprintf("%.4e", P_Value_Group),
      P_Value_Interaction = sprintf("%.4e", P_Value_Interaction),
      P_Random_Effect     = sprintf("%.4e", P_Random_Effect),
      ICC                 = round(ICC, 3)
    )

  # Print results
  cat("--- OPTION 1: Für Excel (Tab-getrennt) ---\n")
  cat("Density\tP_Group_Main\tP_Interaction\tP_Random_Effect\tICC\tSingular\n")
  for(i in 1:nrow(table_df)) {
    cat(paste(table_df[i, ], collapse = "\t"), "\n")
  }
  


