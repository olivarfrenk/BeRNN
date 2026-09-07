# ======================================================================================
# Linear Mixed Model to compare model internal representations over time between subjects
# ======================================================================================
library(tidyverse)
library(lme4)
library(lmerTest)

target_dir <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNN_v1/data/"

file_list <- list.files(
  path = target_dir, 
  pattern = "trajectory_data_.*highDim_original.*\\.csv$", 
  full.names = TRUE
)

full_dataset <- file_list %>% 
  map_df(~read_csv(.)) %>% 
  mutate(
    Subject = as.factor(Subject),
    Month = as.factor(Month),
    Model_ID = as.factor(Model_ID)
  )

# Preview the gathered data
head(full_dataset)


# Test: Significant differences between months within subjects *****************
# Test: Significant differences within months between subjects *****************
global_month_m1 <- lmer(Marker_1 ~ Month + Subject + (1 | Model_ID), data = full_dataset)
global_month_m2 <- lmer(Marker_2 ~ Month + Subject + (1 | Model_ID), data = full_dataset)
global_month_m3 <- lmer(Marker_3 ~ Month + Subject + (1 | Model_ID), data = full_dataset)

anova(global_month_m1)
anova(global_month_m2)
anova(global_month_m3)


