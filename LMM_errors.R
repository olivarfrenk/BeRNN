library(dplyr)
library(nnet)       # For Fixed-Effects Multinomial
library(brms)       # For Bayesian Mixed-Effects Multinomial (Optional)
library(car)        # For Wald Anova tests

# Read raw data (keeping counts intact)
data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts.csv"
all_data <- read.csv(data_path, stringsAsFactors = FALSE)

# Ensure subject and category are factors
all_data$subject <- as.factor(all_data$subject)
all_data$error_category <- as.factor(all_data$error_category)

unique_tasks <- unique(all_data$task)

for (current_task in unique_tasks) {
  cat("\n========================================================\n")
  cat("ANALYSIS FOR TASK:", current_task, "\n")
  cat("========================================================\n")
  
  # Filter data for current task
  task_data <- all_data %>% filter(task == current_task)
  
  # Exclude categories with absolute zero counts across all subjects
  task_data <- task_data %>%
    group_by(error_category) %>%
    filter(sum(count) > 0) %>%
    ungroup()
  
  # To feed an aggregated table into a multinomial model, 
  # we reconstruct it into a subject-by-category matrix
  wide_data <- task_data %>%
    tidyr::pivot_wider(id_cols = subject, names_from = error_category, values_from = count, values_fill = 0)
  
  # Create a matrix of counts for the dependent variable
  error_matrix <- as.matrix(wide_data[, -1])
  
  # ----------------------------------------------------
  # PATH A: FIXED-EFFECTS MULTINOMIAL 
  # ----------------------------------------------------
  cat("\n--- Running Fixed-Effects Multinomial Regression ---\n")
  
  model_fixed <- tryCatch({
    multinom(error_matrix ~ subject, data = wide_data, trace = FALSE)
  }, error = function(e) {
    cat("Warning: Model fitting failed for this task. Check for extreme data sparsity.\n")
    return(NULL)
  })
  
  if (!is.null(model_fixed)) {
    # Type II Anova provides the definitive Chi-Square test for the "subject" effect
    cat("\nANOVA Results (Does error distribution differ across subjects?):\n")
    print(Anova(model_fixed, type = "II"))
  }
  
  # ----------------------------------------------------
  # PATH B: BAYESIAN MIXED-EFFECTS MULTINOMIAL 
  # ----------------------------------------------------
  # cat("\n--- Running Bayesian Mixed-Effects Multinomial ---\n")
  # # Note: brms requires a long format where counts are treated as weights
  # model_mixed <- brm(
  #   error_category | weights(count) ~ 1 + (1 | subject),
  #   data = task_data,
  #   family = categorical(link = "logit"),
  #   chains = 2, cores = 2, iter = 2000, refresh = 0
  # )
  # print(summary(model_mixed))
  
}
