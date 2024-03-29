Process Raw Data and Split into Test/Train Sets
================
Paul Johnson
2022-09-07

- <a href="#import-data" id="toc-import-data">Import Data</a>
- <a href="#clean-and-process-data-and-select-relevant-columns"
  id="toc-clean-and-process-data-and-select-relevant-columns">Clean and
  Process Data and Select Relevant Columns</a>
- <a href="#split-data-in-to-traintest-sets-and-resamples"
  id="toc-split-data-in-to-traintest-sets-and-resamples">Split Data in to
  Train/Test Sets and Resamples</a>
- <a href="#save-data" id="toc-save-data">Save Data</a>

# Import Data

``` r
# load raw data
big12_raw <- readr::read_csv(here::here("data", "raw", "big_12.csv"))
```

# Clean and Process Data and Select Relevant Columns

``` r
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
```

# Split Data in to Train/Test Sets and Resamples

Using a relatively large training set because there’s plenty of data and
I’m a little bit concerned about the potential for overfitting.

The data is split into a training set which is 60% of the full dataset,
and a testing set which is the remaining 40%. The training set is then
split into ten different resamples (split into analysis and assessment
sets) using k-fold cross-validation.

``` r
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
```

# Save Data

We will save the processed dataset, the initial train/test split, the
train/test sets, and the resamples, so that they can be imported in the
notebooks that follow.

``` r
save(
  # data
  df,
  # splits
  train_test_split,
  # train/test data
  train_df, test_df,
  # folds
  train_folds,
  file = here::here(
    "data", "vb_data.RData"
  )
)
```
