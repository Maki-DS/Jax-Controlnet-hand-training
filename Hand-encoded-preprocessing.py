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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
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

    right_style_lm = copy.deepcopy(solutions.drawing_styles.get_default_hand_landmarks_style())
    left_style_lm = copy.deepcopy(solutions.drawing_styles.get_default_hand_landmarks_style())
    right_style_lm[0].color=(251, 206, 177)
    left_style_lm[0].color=(255, 255, 225)

    def draw_landmarks_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.zeros_like(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            if handedness[0].category_name == "Left":
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    left_style_lm,
                    solutions.drawing_styles.get_default_hand_connections_style())
            if handedness[0].category_name == "Right":
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    right_style_lm,
                    solutions.drawing_styles.get_default_hand_connections_style())
        return Image.fromarray(annotated_image)

    def preprocess_and_save(example, idx):
        try:
            image = example["image"]
            image_path = f"{args.train_data_dir}/images/{idx}.png"
            image.save(image_path)

            # generate and save mediapipe landmarks
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(example["image"]))
            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
            detector = vision.HandLandmarker.create_from_options(options)
            detection_result = detector.detect(image)

            processed_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
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
            if len(detection_result.hand_landmarks) > 0:
                with jsonlines.open(
                    f"{args.train_data_dir}/meta_filtered.jsonl", "a"
                ) as writer:  # for writing
                    writer.write(meta)

        except Exception as e:
            logger.error(f"Failed to process image {idx}: {str(e)}")

    dataset.map(preprocess_and_save, num_proc=args.num_proc, with_indices=True)

    print(f"created data folder at: {args.train_data_dir}")