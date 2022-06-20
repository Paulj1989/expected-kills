#!/usr/bin/Rscript

# Setup ----

# import packages
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(parsnip)
  library(yardstick)
  library(workflowsets)
  library(discrim)
  
})

# import train/test data
load(here::here("data", "intermediate", "splits.RData"))

# import recipe
model_rec <- readr::read_rds(here::here("data", "intermediate", "recipe.rds"))

# set ggplot theme
theme_set(theme_minimal(base_size = 14))

# specify that tidymodels output needs to work on a dark theme
options(tidymodels.dark = TRUE)

# Metrics ----

eval_metrics <-
  metric_set(roc_auc, accuracy, f_meas, j_index)

# Models ----

# logistic regression
log_mod <- 
  logistic_reg(
    penalty = 0.2
  ) %>%
  set_mode('classification') %>%
  set_engine('glm')

# naive bayes
nb_mod <-
  naive_Bayes() %>%
  set_mode('classification') %>%
  set_engine('klaR')

# knn
knn_mod <- 
  nearest_neighbor() %>% 
  set_mode("classification") %>%
  set_engine("kknn")

# nnet
nnet_mod <- 
  mlp() %>% 
  set_mode("classification") %>%
  set_engine("keras",
             verbose = FALSE)

# random forest
rf_mod <-
  rand_forest(
    trees = 1000
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# xgboost
xgb_mod <-
  boost_tree(
    trees = 1000,
    stop_iter = 10
  ) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

# Workflow ----

# specify the workflow set
model_pipeline <- 
   workflow_set(
     # preprocessing steps
      preproc = list(model_rec),
      # models to be trained
      models = 
        list(
          log = log_mod,
          nb = nb_mod,
          knn = knn_mod,
          nnet = nnet_mod,
          rf = rf_mod,
          xgb = xgb_mod)
   ) %>%
  # simplify workflow id for each model
  mutate(wflow_id = gsub("(recipe_)", "", wflow_id))

# Fit ----

# fit models on resamples
train_models <-
  model_pipeline %>%
  # map across all preprocessing steps and models in workflow set
   workflow_map(
     "fit_resamples",
     # set seed for reproducibility
     seed = 456,
     # identify resamples
     resamples = train_folds,
     # metrics for evaluating performance
     metrics = eval_metrics,
     # log results throughout training process
     verbose = TRUE)

# Save Models ----

readr::write_rds(train_models, here::here("outputs", "models", "trained_models.rds"))
