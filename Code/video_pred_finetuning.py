# finetune the parameters for video prediction

from pbp_functions import *

# predict_new_video(video_path = "Film/Edge_Case_Game/IMG_0408.MOV", window = 2) # looks fine
predict_new_video(video_path = "Film/Andy_Mike/IMG_9868.MOV", window = 2) # perfect except for last issue with stone that disappears

# I think the parameters are fine as they are, the model just needs to see more stones that are directly touching

# Mini model retrain before release:
# - 75 more green stone frames
# - 125 more directly touching frames (50 black, 50 gray, 25 green)

