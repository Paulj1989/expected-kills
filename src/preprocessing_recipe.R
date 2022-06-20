#!/usr/bin/Rscript

# Setup ----

# import packages
suppressPackageStartupMessages({
  library(dplyr)
  library(recipes)
})

# import train/test data
load(here::here("data", "intermediate", "splits.RData"))

# Preprocessing Recipe ----

model_rec <-
  # model formula
  recipe(
    evaluation_code ~
      skill_type + skill_subtype + blockers +
      team_setter_position + opponent_setter_position +
      start_coordinate_x + start_coordinate_y +
      end_coordinate_x + end_coordinate_y,
    data = train_df
  ) %>%
  # convert team setter position to 2HP and 3HP groups
  step_mutate(
    team_setter_position =
      case_when(
        team_setter_position %in% c(1, 6, 5) ~ "3HP",
        team_setter_position %in% c(2, 3, 4) ~ "2HP"
      ),
    # convert opponent setter position to 2HP and 3HP groups
    opponent_setter_position =
      case_when(
        opponent_setter_position %in% c(1, 6, 5) ~ "3HP",
        opponent_setter_position %in% c(2, 3, 4) ~ "2HP"
      )
  ) %>%
  # convert setter position variables to factors
  step_string2factor(
    team_setter_position,
    opponent_setter_position
  ) %>%
  # combine low frequency factor levels
  step_other(all_nominal(), threshold = 0.05) %>%
  # convert categorical variables to factors
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # specify interactions between variables
  step_interact(
    terms = ~ end_coordinate_x:end_coordinate_y +
      start_coordinate_x:start_coordinate_y
  ) %>%
  # remove variables that are heavily correlated with others
  step_corr(threshold = 0.8) %>%
  # remove no variance predictors which provide no predictive information
  step_nzv(all_predictors())

# Save ----

readr::write_rds(model_rec, here::here("data", "intermediate", "recipe.rds"))