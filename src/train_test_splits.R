#!/usr/bin/Rscript

# Setup ----

# import packages
suppressPackageStartupMessages({
  library(rsample)
})

# load data
df <- readr::read_rds(here::here("data", "intermediate", "cleaned_data.rds"))

# Split Data ----

# set seed for reproducibility
set.seed(456)

# split train/test data
train_test_split <-
  initial_split(
    df,
    strata = evaluation_code,
    prop = 0.6
  )

# set train/test data as df objects
train_df <-
  training(train_test_split)

test_df <-
  testing(train_test_split)

# set cross-validation folds
train_folds <-
  vfold_cv(
    train_df,
    v = 10,
    strata = evaluation_code
  )

# Save ----

save(
  # splits
  train_test_split,
  # train/test data
  train_df, test_df,
  # folds
  train_folds,
  file = here::here(
    "data", "intermediate", "splits.RData"
  )
)