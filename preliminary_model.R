pacman::p_load(tidyverse, tidymodels, bbplot)

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

# model predictions
xg_preds <-
  xg_fit %>%
  predict(new_data = power5_test) %>%
  bind_cols(power5_test %>% select(skillquality))

xg_preds$skillquality <- factor(xg_preds$skillquality)

# plotting confusion matrix
xg_preds %>%
  conf_mat(skillquality, .pred_class) %>%
  pluck(1) %>%
  as_tibble() %>%
  ggplot(aes(Prediction, Truth, alpha = n)) +
  geom_tile(show.legend = FALSE) +
  geom_text(aes(label = n), colour = "white", alpha = 1, size = 8) +
  bbc_style() +
  labs(title = "Confusion Matrix Plotting Model Predictions & Actual Outcomes") +
  theme(
    plot.title = element_text(size = 20),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank()
  )

ggsave("conf_mat.png", width = 10, height = 8)

# model performance metrics
# (accuracy, precision, recall, f1 score)

tibble(
  "Accuracy" =
    metrics(xg_preds, skillquality, .pred_class) %>%
      filter(.metric == "accuracy") %>%
      select(.estimate),
  "Precision" =
    precision(xg_preds, skillquality, .pred_class) %>%
      select(.estimate),
  "Recall" =
    recall(xg_preds, skillquality, .pred_class) %>%
      select(.estimate),
  "F1 Score" =
    f_meas(xg_preds, skillquality, .pred_class) %>%
      select(.estimate)
) %>%
  unnest(cols = c(Accuracy, Precision, Recall, `F1 Score`))
