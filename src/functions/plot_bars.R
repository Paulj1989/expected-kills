# function for plotting bar charts
plot_bars <- function(var) {
  train_df %>%
    ggplot(aes(x = evaluation_code, fill = {{var}})) +
    geom_bar(position = "dodge") +
    scale_fill_viridis_d() +
    labs(x = NULL, y = NULL, fill = NULL)
}
