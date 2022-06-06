
# function to evalute model performances from a workflow set
evaluate_performance <- function(models, metric){
  models %>% 
    rank_results(metric) %>% 
    filter(.metric == metric) %>% 
    select(model, !!metric := mean, rank)
}

# function to generate frequency table (from janitor) of a variable in the data
freq_tbl <- function(df, var){
  df %>%
    janitor::tabyl(var)
}


