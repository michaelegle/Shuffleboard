library(tidyverse)
library(gganimate)

shuffle_data <- read_csv("Documents/Shuffleboard/Data/tracking_data_cleaned.csv")

shuffle_data %>% 
  filter(!is.na(toss_id)) %>% 
  View()

shuffle_data_end <- shuffle_data %>% 
  group_by(toss_id, track_id) %>% 
  arrange(frame) %>% 
  slice_tail()

unique(shuffle_data$toss_id)

test <- shuffle_data %>% 
  filter(toss_id == 11)

# Tosses should be:
# - initial board setup (3 stones) | TOSS ID 1
# - first actual toss all the way. gray stone hits black stone and gray stone falls off board | TOSS ID 3
# - second actual toss. black stone makes slight contact with gray stone and stops on board | TOSS ID 7
# - third actual toss. gray stone knocks off black stone
# - fourth actual toss. black stone lightly taps gray stone (this one may cause issues) | TOSS ID 15
# - fifth actual toss. gray stone hits black stone and both fall off board
# - sixth actual toss. black stone hits black and gray stones, black stone that was just tossed and gray stone fall off board | TOSS ID 21
# - seventh actual toss. gray stone hits black stone, which knocks another gray stone off of board | TOSS ID 25
# - eighth actual toss. black stone hits gray stone off of board
# - ninth actual toss. gray stone knocks off black stone and sticks on edge of the board | TOSS ID 31

# should be 10 total unique non-NA toss IDs

shuffle_data %>%
  filter(toss_id == 15) %>% 
  filter(stone_settled == 0) %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() +
  coord_fixed() +
  xlim(c(0, 26)) +
  ylim(c(0, 188))

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





