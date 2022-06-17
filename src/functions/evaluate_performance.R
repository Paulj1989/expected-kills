# function to evaluate model performances from a workflow set
evaluate_performance <- function(models, metric, select_best = FALSE){
  models %>% 
    rank_results(
      {{metric}},
      select_best = select_best) %>% 
    filter(.metric == {{metric}})
}
