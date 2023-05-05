import argparse
import logging
import random
import copy
import jsonlines
import numpy as np
import requests
from datasets import load_dataset
from PIL import Image
import mediapipe as mp
import os

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Example of a data preprocessing script."
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="The directory to store the dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="The directory to store cache",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="number of examples in the dataset",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="number of processors to use in `dataset.map()`",
    )
    parser.add_argument(
        "--source_data_loc",
        type=str,
        required=True,
        help="The directory or path to local or huggingface dataset",
    )
    args = parser.parse_args()
    return args


def filter_dataset(dataset):
    small_dataset = dataset.shuffle(seed=0).select(range(args.max_train_samples))
    return small_dataset

if __name__ == "__main__":
    args = parse_args()

    # load hand grid dataset
    dataset = load_dataset(
        args.source_data_loc,
        cache_dir=args.cache_dir,
        split="train" # change for different splits
    )
    if not os.path.exists(args.train_data_dir):
        os.makedirs(f"{args.train_data_dir}/images")
        os.makedirs(f"{args.train_data_dir}/processed_images")

    id_list = range(len(dataset))
    dataset = dataset.add_column("id", id_list)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    def draw_landmarks_on_image(rgb_image, detection_result):
        annotated_image = np.zeros_like(rgb_image)

        mp_drawing.draw_landmarks(annotated_image, detection_result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        
        # Right hand
        mp_drawing.draw_landmarks(annotated_image, detection_result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Left Hand
        mp_drawing.draw_landmarks(annotated_image, detection_result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        mp_drawing.draw_landmarks(annotated_image, detection_result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        return annotated_image

    def preprocess_and_save(example, idx):
        try:

            image = example["image"]
            image_path = f"{args.train_data_dir}/images/{idx}.png"
            image.save(image_path)

            # generate and save mediapipe landmarks
            image = np.array(example["image"])
            holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)
            detection_result = holistic.process(processed_image)

            processed_image = draw_landmarks_on_image(image, detection_result)
            processed_image_path = (
                f"{args.train_data_dir}/processed_images/{idx}.png"
            )
            processed_image.save(processed_image_path)

            caption = example['text']

            # write to meta.jsonl
            meta = {
                "image": image_path,
                "conditioning_image": processed_image_path,
                "caption": caption,
            }
            # Doesn't write if 0 hands are detected
            if not (detection_result.left_hand_landmarks == None and detection_result.right_hand_landmarks == None):
                with jsonlines.open(
                    f"{args.train_data_dir}/meta_filtered.jsonl", "a"
                ) as writer:  # for writing
                    writer.write(meta)

        except Exception as e:
            logger.error(f"Failed to process image {idx}: {str(e)}")

    dataset.map(preprocess_and_save, num_proc=args.num_proc, with_indices=True)

    print(f"created data folder at: {args.train_data_dir}")