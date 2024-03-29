Specifying Preprocessing Recipe and Evaluation Metrics
================
Paul Johnson
2022-09-07

- <a href="#preprocessing-recipe"
  id="toc-preprocessing-recipe">Preprocessing Recipe</a>
- <a href="#specify-evaluation-metrics"
  id="toc-specify-evaluation-metrics">Specify Evaluation Metrics</a>
- <a href="#save-recipe-and-metrics" id="toc-save-recipe-and-metrics">Save
  Recipe and Metrics</a>

We can use everything we’ve learned from the EDA to specify a
preprocessing recipe that processes the data ready for the model
training.

# Preprocessing Recipe

Having carried out some exploration of the data in the [previous
notebook](./src/eda.Rmd) the following features were identified for the
model:

- Skill Type
- Skill Subtype
- Blockers
- Team/Opponent Setter Position (Grouped 1-3 and 4-6)
- Start and End X/Y Coordinates (plus interactions)

``` r
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
      start_coordinate_x:start_coordinate_y) %>%
  # remove variables that are heavily correlated with others
  step_corr(threshold = 0.8) %>%
  # remove no variance predictors which provide no predictive information
  step_nzv(all_predictors())
  
# store the baked recipe df
baked_df <-
  model_rec %>%
  prep() %>%
  bake(new_data=NULL)
```

Comments:

- I don’t think we need to normalize the numeric variables because all
  are between 0 - 25, and I think these are suitably similar scales.

We can have a look at the data after all the preprocessing steps have
been carried out:

``` r
# check everything looks good
glimpse(baked_df)
```

    Rows: 11,165
    Columns: 19
    $ blockers                                <dbl> 3.0, 1.0, 2.0, 2.0, 2.0, 1.5, …
    $ start_coordinate_x                      <dbl> 1.68125, 1.49375, 0.70625, 3.2…
    $ start_coordinate_y                      <dbl> 3.129630, 3.055556, 2.981481, …
    $ end_coordinate_x                        <dbl> 0.55625, 1.60625, 1.64375, 1.1…
    $ end_coordinate_y                        <dbl> 6.981481, 3.722222, 5.648148, …
    $ evaluation_code                         <fct> error, error, error, error, er…
    $ skill_type_Half.ball.attack             <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    $ skill_type_Head.ball.attack             <dbl> 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, …
    $ skill_type_High.ball.attack             <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    $ skill_type_Quick.ball.attack            <dbl> 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, …
    $ skill_type_Slide.ball.attack            <dbl> 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, …
    $ skill_subtype_Hard.spike                <dbl> 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, …
    $ skill_subtype_Tip                       <dbl> 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, …
    $ team_setter_position_X2HP               <dbl> 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, …
    $ team_setter_position_X3HP               <dbl> 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, …
    $ opponent_setter_position_X2HP           <dbl> 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, …
    $ opponent_setter_position_X3HP           <dbl> 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, …
    $ end_coordinate_x_x_end_coordinate_y     <dbl> 3.883449, 5.978819, 9.284144, …
    $ start_coordinate_x_x_start_coordinate_y <dbl> 5.261690, 4.564236, 2.105671, …

Everything looks good!

# Specify Evaluation Metrics

We can also set the evaluation metrics for the training process. We will
use ROC-AUC as the primary evaluation metric but also checking on the
Accuracy, F1-Score, and J-Index.

``` r
eval_metrics <-
  metric_set(roc_auc, accuracy, f_meas, j_index)
```

This isn’t really a preprocessing step but the same evaluation metrics
are reused throughout the remaining notebooks, so it makes sense to set
the metrics now and save on code elsewhere.

# Save Recipe and Metrics

We will save the preprocessing recipe and metric set so that they can be
imported in the notebooks that follow.

``` r
save(
  # recipe
  model_rec,
  # metrics
  eval_metrics,
  file = here::here(
    "data", "preprocessed.RData"
  )
)
```
