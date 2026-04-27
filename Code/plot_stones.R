library(tidyverse)
library(gganimate)

shuffle_data <- read_csv("Documents/Shuffleboard/Data/tracking_data_cleaned.csv")

shuffle_data %>% 
  filter(!is.na(toss_id)) %>% 
  View()

shuffle_data_2 <- shuffle_data %>% 
  group_by(track_id) %>% 
  arrange(frame) %>% 
  mutate(x_lag = lag(x),
         y_lag = lag(y),
         change = sqrt((x - x_lag)^2 + (y - y_lag)^2))

quantile(shuffle_data_2$change, c(0.01, 0.25, 0.5, 0.75, 0.9), na.rm = T)

anim <- ggplot() +
  geom_point(data = shuffle_data,
             aes(x = x, y = y, color = class_name, group = track_id, size = 0.5)) +
  scale_radius() +
  scale_color_manual(values = c("black", "gray")) +
  transition_time(frame) +
  ease_aes('linear') +
  coord_fixed() +
  theme_minimal()


animate(anim, fps = 30, nframes = max(shuffle_data$frame), width = 1000, end_pause = 5, renderer = gifski_renderer())

anim_save("Documents/Shuffleboard/Data/test_save.gif")


test_points <- data.frame(x = c(3, 3, 3, 3, 23, 23, 23, 23),
                          y = c(94, 18, 12, 6, 94, 18, 12, 6))

test_points %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() +
  coord_fixed() +
  xlim(c(0, 26)) +
  ylim(c(0, 94))
