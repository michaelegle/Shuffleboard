import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import pandas as pd
import numpy as np

from test_build_pbp import clean_tracking_data

predictions = pd.read_csv("Data/predictions.csv")

predictions = clean_tracking_data(predictions)

print(predictions)

def build_court_design():

    fig, ax = plt.subplots()

    ax.add_patch(patches.Rectangle((0, 0), 26, 188, facecolor = "#465337"))
    ax.add_patch(patches.Rectangle((3, 6), 20, 176, facecolor = "#CAB498"))
    for y in [6, 12, 18, 94, 170, 176, 182]:
        ax.plot([3, 23], [y, y], color = "black", lw = 1)

    ax.text(x = 13, y = 9, s = "3", color = "black", ha = "center", va = "center", fontsize = 5, zorder=1)
    ax.text(x = 13, y = 15, s = "2", color = "black", ha = "center", va = "center", fontsize = 5, zorder=1)
    ax.text(x = 13, y = 21, s = "1", color = "black", ha = "center", va = "center", fontsize = 5, zorder=1)
    ax.text(x = 13, y = 167, s = "1", color = "black", ha = "center", va = "center", fontsize = 5, rotation = 180, zorder=1)
    ax.text(x = 13, y = 173, s = "2", color = "black", ha = "center", va = "center", fontsize = 5, rotation = 180, zorder=1)
    ax.text(x = 13, y = 179, s = "3", color = "black", ha = "center", va = "center", fontsize = 5, rotation = 180, zorder=1)

    ax.set_aspect('equal')

    plt.xlim(0, 26)
    plt.ylim(0, 188)
    plt.axis("off")

    plt.show()



#build_court_design()


fig, ax = plt.subplots()

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.margins(0)

ax.add_patch(patches.Rectangle((0, 0), 26, 188, facecolor = "#465337"))
ax.add_patch(patches.Rectangle((3, 6), 20, 176, facecolor = "#CAB498"))
for y in [6, 12, 18, 94, 170, 176, 182]:
    ax.plot([3, 23], [y, y], color = "black", lw = 1)

ax.text(x = 13, y = 9, s = "3", color = "black", ha = "center", va = "center", fontsize = 5)
ax.text(x = 13, y = 15, s = "2", color = "black", ha = "center", va = "center", fontsize = 5)
ax.text(x = 13, y = 21, s = "1", color = "black", ha = "center", va = "center", fontsize = 5)
ax.text(x = 13, y = 167, s = "1", color = "black", ha = "center", va = "center", fontsize = 5, rotation = 180)
ax.text(x = 13, y = 173, s = "2", color = "black", ha = "center", va = "center", fontsize = 5, rotation = 180)
ax.text(x = 13, y = 179, s = "3", color = "black", ha = "center", va = "center", fontsize = 5, rotation = 180)

ax.set_aspect('equal')

plt.xlim(0, 26)
plt.ylim(0, 188)
plt.axis("off")

ax.set_xlim(0, 26)
ax.set_ylim(0, 188)


COLOR_MAP = {
    "black_stone": "#1a1a1a",
    "gray_stone":  "#a0a0a0",
    "green_stone": "#93d645"
}



predictions['frame'] = predictions['frame'].astype(int)

colors = plt.cm.tab10.colors
num_frames = max(predictions['frame'])

scatters = {}
tracks = {}
for _, row in predictions.iterrows():
    tid = row["track_id"]
    if tid not in tracks:
        tracks[tid] = {
            "positions": [],
            "color": COLOR_MAP.get(row["final_class_name"], "#ffffff")
        }
    tracks[tid]['positions'].append((int(row["frame"]), row["x"], row["y"]))

    
MAX_FRAMES_SINCE_SEEN = 0  # hide stone if not detected in last 5 frames

def get_pos_at_frame(tid, frame):
    positions = [(f, x, y) for f, x, y in tracks[tid]["positions"] if f <= frame]
    if not positions:
        return None
    best = max(positions, key=lambda p: p[0])
    # Hide if last detection was too long ago
    if frame - best[0] > MAX_FRAMES_SINCE_SEEN:
        return None
    return best

def update(frame):
    for tid in tracks:
        pos = get_pos_at_frame(tid, frame)
        if pos is None:
            scatters[tid].set_data([], [])
            continue

        _, x, y = pos
        scatters[tid].set_data([x], [y])

    ax.set_xlabel(f"Frame {frame}/{num_frames}")
    return list(scatters.values())

# And simplify scatter creation — remove trails dict entirely
scatters = {}
for tid in tracks:
    color = tracks[tid]["color"]
    scatters[tid], = ax.plot([], [], "o", color=color, markersize=4,
                              markeredgecolor="white", markeredgewidth=1,
                              label=f"Stone {tid}", zorder = 2)

ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=1000/30,
    blit=False
)

plt.tight_layout()
#plt.show()

ani.save("stone_tracking.mp4", writer="ffmpeg", fps=30)

