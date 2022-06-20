#!/usr/bin/Rscript

# Setup ----

# import packages
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tune)
  library(workflowsets)
  library(yardstick)
})

# import trained models
train_models <- 
  readr::read_rds(here::here("outputs", "models", "trained_models.rds"))

# import function for saving metrics
source(here::here("src", "functions", "save_metrics.R"))

# set ggplot theme
theme_set(theme_minimal(base_size = 14))

# Evaluate Performance ----

## Metrics ----

# apply save_metrics function across each model
eval_sets <-
  purrr::map(
    train_models$wflow_id,
    save_metrics,
    data = train_models
  ) %>%
  purrr::set_names(
    train_models$wflow_id
  )

# save each metric set in a yaml file
for (i in seq_along(eval_sets)) {
  yaml::write_yaml(
    eval_sets[[i]],
    file =
      here::here("outputs", "metrics", paste0(names(eval_sets[i]), ".yaml")
      )
  )
}

## Plots ----

# visualize performance across all metrics
eval_plot <-
  autoplot(train_models) +
  scale_color_viridis_d()

ggsave(
  here::here("outputs", "plots", "train_eval.png"),
  eval_plot, width = 12, height = 6
)