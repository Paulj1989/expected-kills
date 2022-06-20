#!/usr/bin/Rscript

# Setup ----

# import packages
suppressPackageStartupMessages({
  library(dplyr)
})

# load raw data
big12_raw <- readr::read_csv(here::here("data", "raw", "big_12.csv"))

# Process Data ----

df <- big12_raw %>%
  # filter rows to return relevant subset
  filter(skill == "Attack") %>%
  mutate(
    # convert evaluation code to named variables for easier interpretation
    evaluation_code =
      recode(evaluation_code,
        "#" = "kill",
        "/" = "error",
        "=" = "error",
        "+" = "play",
        "-" = "play"
      ),
    # convert hole in the block (4) to 1.5 to better reflect linear effectiveness of the block
    blockers =
      recode(
        num_players_numeric,
        `0` = 0,
        `1` = 1,
        `2` = 2,
        `3` = 3,
        `4` = 1.5
      ),
    # create team setter position for easier representation of setter position effect
    team_setter_position =
      case_when(
        homeaway == "Home" ~ home_setter_position,
        homeaway == "Away" ~ visiting_setter_position
      ),
    # create opponent setter position for easier representation of setter position effect
    opponent_setter_position =
      case_when(
        homeaway == "Home" ~ visiting_setter_position,
        homeaway == "Away" ~ home_setter_position
      )
  ) %>%
  # filter non-terminal plays
  filter(evaluation_code != "play" &
    phase == "Reception") %>%
  # select columns that are relevant to the model
  select(
    "evaluation_code",
    "skill_type",
    "start_zone",
    "end_zone",
    "end_subzone",
    "skill_subtype",
    "blockers",
    "team_setter_position",
    "opponent_setter_position",
    "start_coordinate_x",
    "start_coordinate_y",
    "end_coordinate_x",
    "end_coordinate_y",
    "contact",
    "rotation"
  ) %>%
  # drop na's in the data
  tidyr::drop_na()

# Save ----

readr::write_rds(df, here::here("data", "intermediate", "cleaned_data.rds"))