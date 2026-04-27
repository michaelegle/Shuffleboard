import cv2
import random
import os

def save_random_frames_from_clip(video_folder, output_folder, frames_per_video = 1):
    video_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith("test_clip.mov")
    ]

    print(f"Found {len(video_files)} video(s). Extracting {frames_per_video} frames each...")

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  Could not open: {video_file}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < frames_per_video:
            print(f"  Warning: {video_file} has only {total_frames} frames.")
            frame_indices = list(range(total_frames))
        else:
            # Pick random unique frame indices
            frame_indices = sorted(random.sample(range(total_frames), frames_per_video))
        
        print(f"  Processing: {video_file} ({total_frames} total frames)")

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                output_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, frame)
                print(f"    Saved frame {frame_idx} → {output_filename}")
            else:
                print(f"    Could not read frame {frame_idx}")
        
        cap.release()
    
    print("\nDone!")



save_random_frames_from_clip("../Film", "../Training Images")

