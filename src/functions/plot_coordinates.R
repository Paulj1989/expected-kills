# function for plotting x,y coordinate plots
plot_coordinates <- function(x, y){
  train_df %>%
    ggplot(aes(x = {{x}}, y = {{y}})) +
    geom_hex() +
    scale_fill_viridis_c() +
    labs(fill = NULL) +
    facet_wrap(~evaluation_code)
}
