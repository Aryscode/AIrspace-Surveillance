# extract_frames.py
import os
import cv2
import random
import pandas as pd

def extract_frames(video_path, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{prefix}_frame{frame_idx:04}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        frame_list.append((prefix, frame_idx))
        frame_idx += 1

    cap.release()
    print(f"[{prefix}] Extracted {frame_idx} frames.")
    return frame_list

def process_all_videos(video_folder, output_folder, split_ratio=0.8):
    all_samples = []

    for fname in os.listdir(video_folder):
        if fname.endswith(".mp4") and fname.startswith("V_"):
            prefix = os.path.splitext(fname)[0]
            video_path = os.path.join(video_folder, fname)
            out_dir = os.path.join(output_folder, prefix)
            frames = extract_frames(video_path, out_dir, prefix=prefix)
            all_samples.extend(frames)

    # Shuffle and split
    random.shuffle(all_samples)
    train_cut = int(len(all_samples) * split_ratio)
    train_samples = all_samples[:train_cut]
    test_samples = all_samples[train_cut:]

    pd.DataFrame(train_samples, columns=["video", "frame"]).to_csv("train_index.csv", index=False)
    pd.DataFrame(test_samples, columns=["video", "frame"]).to_csv("test_index.csv", index=False)
    print(f"Train/Test split complete: {len(train_samples)} train frames, {len(test_samples)} test frames.")

if __name__ == "__main__":
    process_all_videos("videos", "images", split_ratio=0.8)
