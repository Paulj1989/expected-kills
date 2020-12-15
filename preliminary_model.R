pacman::p_load(tidyverse, tidymodels)

# load data
power5 <- read_csv("power5_2018.csv")

# rename, filter, and select relevant columns
df <- power5 %>%
  # rename all columns to lowercase
  rename_all(tolower) %>%
  rename(
    "prev_contact" = previouscontact,
    "prevprev_contact" = previouspreviouscontact
  ) %>%
  # filter rows to return model subset
  filter(gender == "women" & skill == "A") %>%
  filter(skillquality == "#" | skillquality == "=") %>%
  # remove grade from contact columns
  mutate(
    prev_contact = str_sub(prev_contact, 1, 1),
    prevprev_contact = str_sub(prevprev_contact, 1, 1)
  ) %>%
  # select columns that are relevant to the model
  select(
    "skillquality", "skilltype", "prev_contact",
    "prevprev_contact", "blockers", "extra",
    "set_angle", "firstx", "firsty", "thirdx", "thirdy"
  )

# split train/test data
set.seed(456)

train_test_split <-
  initial_split(df,
    prob = 0.8,
    strata = skillquality
  )

power5_train <- training(train_test_split)
power5_test <- testing(train_test_split)

# preprocessing recipe
power5_recipe <-
  # identify the outcome and predictor variables
  recipe(skillquality ~ ., data = power5_train) %>%
  # dummy nominal variables (except for outcome)
  step_dummy(all_nominal(), -skillquality) %>%
  # apply BoxCox Transformation to all predictor variables
  step_BoxCox(all_predictors()) %>%
  # normalize predictor variables
  step_normalize(all_predictors())

# xgboost model
xg_model <-
  boost_tree() %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# model workflow (combine model + recipe)
xg_wflow <-
  workflow() %>%
  add_model(xg_model) %>%
  add_recipe(power5_recipe)

# fit model
xg_fit <- fit(xg_wflow, power5_train)

# predict test results
predict(xg_fit, new_data = power5_test)

# model accuracy
pred_accuracy <-
  power5_test$skillquality %>%
  bind_cols(
    predict(xg_fit, new_data = power5_test),
    predict(xg_fit, new_data = power5_test, type = "prob")
  )