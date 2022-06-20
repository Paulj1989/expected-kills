# function to save metrics
save_metrics <- function (data, model) {
    data %>%
    collect_metrics() %>%
    filter(wflow_id == {{model}}) %>%
    select(.metric, mean) %>%
    tidyr::pivot_wider(
      names_from = .metric,
      values_from = mean
    )
}
