## imports ----------

pacman::p_load(tidymodels, tidyverse, patchwork, doParallel, yaml)

## data ---------

# load data
power5 <- read_csv("power5_2018.csv")

# set random seed
set.seed(456)

# rename, filter, and select relevant columns
df <- power5 %>%
  # rename all columns to lowercase
  rename_all(tolower) %>%
  rename(
    "atk_set_code" = "atk/set code",
    "prev_contact" = previouscontact,
    "prevprev_contact" = previouspreviouscontact
  ) %>%
  # filter rows to return model subset
  filter(gender == "women" & skill == "A") %>%
  # remove grade from contact columns
  mutate(
    skillquality=recode(skillquality, "#"='kill', '/'='error', '='='error',
                                      '+'='play', '-'='play'),
    prev_contact = str_sub(prev_contact, 1, 1),
    prevprev_contact = str_sub(prevprev_contact, 1, 1)
  ) %>%
  # filter non-terminal plays 
  filter(skillquality != "play") %>%
  # shuffle rows
  sample_n(nrow(.)) %>%
  # select columns that are relevant to the model
  select(
    "skillquality", "skilltype", "atk_set_code", "prev_contact",
    "prevprev_contact", "blockers", "extra", "setid", 
    "set_angle", "rallycount", "maxposs", "contactcount",
    "firstx", "firsty", "thirdx", "thirdy"
  )

# split train/test data
train_test_split <-
  initial_split(df,
                strata = skillquality,
                prop = 0.8
  )

train_df <- training(train_test_split)
test_df <- testing(train_test_split)

## preprocessing ----------

# preprocessing recipe
preprocessing <- 
  recipe(skillquality ~ ., data = train_df) %>%
  themis::step_upsample(skillquality, over_ratio = 0.8) %>%
  # convert categorical variables to factors
  step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  step_nzv(all_nominal()) %>%
  # normalize
  step_normalize(all_numeric()) %>% 
  prep()

# cross validation 
cv_folds <- 
  bake(
    preprocessing, 
    new_data = train_df
  ) %>%  
  vfold_cv(v = 5)

## model specification ----------

# xgboost model
xgboost_model <- 
  boost_tree(
    mode = "classification",
    trees = 2000,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),
    stop_iter = 10
  ) %>%
  # binary classification task
  set_engine("xgboost", objective = "binary:logistic",
             lambda=1, alpha=1, verbose=1)

## hyperparameter tuning ----------

# specify parameters
xgboost_params <- 
  parameters(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction()
  )

# grid specification
xgboost_grid <- 
  grid_max_entropy(
    xgboost_params, 
    size = 60
  )

# model workflow 
xgboost_wflow <- 
  workflow() %>%
  add_model(xgboost_model) %>% 
  add_formula(skillquality ~ .)

# set parallel processing to boost speed of model tuning
# model is quite expensive, so this should be done when nothing
# else is running on computer (if processing power is limited)
all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# hyperparameter tuning
xgboost_tuned <- tune_grid(
  object = xgboost_wflow,
  resamples = cv_folds,
  grid = xgboost_grid,
  metrics = metric_set(f_meas),
  control = control_grid(verbose = TRUE)
)

write_yaml(xgboost_tuned, "xgboost_tuned.yml")

# xgboost_tuned <- read_yaml("xgboost.yml")
# xgboost_tuned <- tidypredict::as_parsed_model(xgboost_tuned)

## specifying final model ----------

# show best model
xgboost_tuned %>%
  show_best(metric = "f_meas", event_level="second", n = 10) %>%
  knitr::kable()

# select best model
xgboost_best_params <- xgboost_tuned %>%
  select_best("f_meas", event_level="second")

knitr::kable(xgboost_best_params)

write_yaml(xgboost_best_params, "xgboost_best_params.yml")

# finalize model
xgboost_model_final <- xgboost_model %>% 
  finalize_model(xgboost_best_params)

# fit final model
train_processed <- bake(preprocessing,  new_data = train_df)
xgboost_fit <- xgboost_model_final %>% fit(skillquality~., data=train_processed)

## training ----------

# predictions
xg_preds <- xgboost_fit %>% predict(new_data=train_processed)

#probabilities
xg_probs <- xgboost_fit %>%
  predict_classprob.model_fit(new_data=train_processed)

# preds & probs
predictions <- bind_cols(factor(train_df$skillquality), xg_preds,xg_probs)
knitr::kable(head(predictions))

predictions %>%
  f_meas(truth=...1,estimate=.pred_class, event_level="second")

write_yaml(predictions, "train_predictions.yml")

#confidence matrix
confmat<-predictions %>% yardstick::conf_mat(truth=...1,estimate=.pred_class) 

p1 <- autoplot(confmat, type = "heatmap") +
  theme_minimal(base_family = "Fira Code", base_size = 14) +
  labs(title = "Confidence Matrix") +
  theme(
    legend.position = "none",
    plot.title = element_text(family = "Montserrat", color = "grey30", size = 14),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12, color = "grey30"),
    panel.grid.major.x = element_line(color = "#F3F5F6", size = 0.4),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank())

# pr curve
pr <- pr_curve(predictions, ...1, error)

p2 <- autoplot(pr) +
  labs(title = "Precision-Recall Curve",
    y = "Precision", x = "Recall") +
  theme_minimal(base_family = "Fira Code", base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(family = "Montserrat", color = "grey30", size = 14),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12, color = "grey30"),
    panel.grid.major.x = element_line(color = "#F3F5F6", size = 0.4),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank())

#combine plots
patchwork <- p1 + p2

patchwork + plot_annotation(
  title = 'Evaluating XGBoost Model Performance Predicting Kills & Errors',
  subtitle = 'Model Accuracy Predicting Kills & Errors in Training Data',
  theme = theme(plot.title = element_text(family = "Montserrat", color = "grey10", size = 22),
                plot.subtitle = element_text(family = "Montserrat", color = "grey30", size = 20)
))

ggsave(here::here("train_plots.png"), dpi = 320, width = 16, height = 9)


## testing ----------

## test data
test_processed  <- bake(preprocessing, new_data = test_df)
xg_test_preds <- xgboost_fit %>% predict(new_data=test_processed)

#probabilities
xg_test_probs <- xgboost_fit %>%
  predict_classprob.model_fit(new_data=test_processed)

# preds & probs
test_predictions <- bind_cols(factor(test_df$skillquality), xg_test_preds, xg_test_probs)
knitr::kable(head(test_predictions))

test_predictions %>%
  f_meas(truth=...1,estimate=.pred_class, event_level="second")

write_yaml(test_predictions, "test_predictions.yml")

#confidence matrix
confmat<-test_predictions %>% yardstick::conf_mat(truth=...1,estimate=.pred_class) 

p1 <- autoplot(confmat, type = "heatmap", family="Fira Code") +
  theme_minimal(base_family = "Fira Code", base_size = 14) +
  labs(title = "Confidence Matrix") +
  theme(
    legend.position = "none",
    plot.title = element_text(family = "Montserrat", color = "grey30", size = 14),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12, color = "grey30"),
    panel.grid.major.x = element_line(color = "#F3F5F6", size = 0.4),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank())

# pr curve
pr <- pr_curve(test_predictions, ...1, error)

p2 <- autoplot(pr) +
  labs(title = "Precision-Recall Curve",
       y = "Precision", x = "Recall") +
  theme_minimal(base_family = "Fira Code", base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(family = "Montserrat", color = "grey30", size = 14),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12, color = "grey30"),
    panel.grid.major.x = element_line(color = "#F3F5F6", size = 0.4),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank())

#combine plots
patchwork <- p1 + p2

patchwork + plot_annotation(
  title = 'Evaluating XGBoost Model Performance Predicting Kills & Errors',
  subtitle = 'Model Accuracy Predicting Kills & Errors in Testing Data',
  theme = theme(plot.title = element_text(family = "Montserrat", color = "grey10", size = 22),
                plot.subtitle = element_text(family = "Montserrat", color = "grey30", size = 20)
  ))

ggsave(here::here("test_plots.png"), dpi = 320, width = 16, height = 9)

## feature importance ----------

# variable importance plots

vip::vip(xgboost_fit) +
  annotate("text", 3, 0.09,
           label = "XGBoost Feature Importance",
           family = "IBM Plex Sans", color = "grey10",
           hjust = 0, size = 15, lineheight = 0.5) +
  annotate("text", 1.8, 0.09,
           label = "The Features with the Greatest Influence\non Expected Kills (xK) Model Predictions",
           family = "IBM Plex Sans", color = "grey30",
           hjust = 0, size = 9, lineheight = 1.2) +
  scale_x_discrete(labels = c("Contact Count", "Rally Count", "Set Angle",
                              "Set ID", "First Y", "Max Poss", "First X",
                              "Previous, Previous Contact", "Third Y",
                              "Third X")) +
  labs(x = NULL, y = "Importance") +
  theme_minimal(base_family = "Fira Code", base_size = 14) +
  theme(
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
    axis.text = element_text(size = 10),
    axis.title.x = element_text(size = 12, color = "grey30"),
    axis.title.y = element_blank(),
    panel.grid.major.x = element_line(color = "#F3F5F6", size = 0.4),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank())

ggsave(here::here("feature_importance.png"), dpi = 320, width = 16, height = 9)
