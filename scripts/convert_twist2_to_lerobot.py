"""
Convert TWIST2 dataset format to LeRobot format compatible with GR00T N1.5.

TWIST2 format:
- Episodes in episode_XXXX/ folders
- Each episode has data.json with frames containing:
  - state_body (35-dim), state_hand_left (7-dim), state_hand_right (7-dim), state_neck (2-dim)
  - action_body (35-dim), action_hand_left (7-dim), action_hand_right (7-dim), action_neck (2-dim)
  - rgb_left, rgb_right image paths

Usage:
    python scripts/convert_twist2_to_lerobot.py --input_dir /path/to/twist2/dataset --output_name my_dataset

The resulting dataset will be saved to $HF_LEROBOT_HOME/output_name
"""

import json
import shutil
from pathlib import Path

import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


# Dimensions for each state/action component
STATE_ACTION_DIMS = {
    "body": 35,
    "hand_left": 7,
    "hand_right": 7,
    "neck": 2,
}
TOTAL_DIM = sum(STATE_ACTION_DIMS.values())  # 51


def find_episodes(dataset_path: Path) -> list[Path]:
    """Find all episode directories in the dataset."""
    episodes = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and item.name.startswith("episode_"):
            data_json = item / "data.json"
            if data_json.exists():
                episodes.append(item)
    return episodes


def load_episode_data(episode_path: Path) -> dict:
    """Load the data.json file for an episode."""
    with open(episode_path / "data.json", "r") as f:
        return json.load(f)


def concatenate_state(frame: dict) -> np.ndarray:
    """Concatenate state components into a single array.
    
    Order: body (35) + hand_left (7) + hand_right (7) + neck (2) = 51 dims
    """
    state = np.concatenate([
        np.array(frame["state_body"], dtype=np.float32),
        np.array(frame.get("state_hand_left", [0.0] * 7), dtype=np.float32),
        np.array(frame.get("state_hand_right", [0.0] * 7), dtype=np.float32),
        np.array(frame.get("state_neck", [0.0] * 2), dtype=np.float32),
    ])
    return state


def concatenate_action(frame: dict) -> np.ndarray:
    """Concatenate action components into a single array.
    
    Order: body (35) + hand_left (7) + hand_right (7) + neck (2) = 51 dims
    """
    action = np.concatenate([
        np.array(frame["action_body"], dtype=np.float32),
        np.array(frame.get("action_hand_left", [0.0] * 7), dtype=np.float32),
        np.array(frame.get("action_hand_right", [0.0] * 7), dtype=np.float32),
        np.array(frame.get("action_neck", [0.0] * 2), dtype=np.float32),
    ])
    return action


def load_image(episode_path: Path, image_rel_path: str) -> np.ndarray:
    """Load an image from the episode directory."""
    image_path = episode_path / image_rel_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def create_modality_json(output_path: Path):
    """Create the modality.json file for GR00T N1.5 compatibility.
    
    Always stores the full semantic breakdown:
    - state_body[0:6] contains: root_vel(2), root_height(1), root_orientation(2), root_ang_vel(1)
    - state_body[6:35] contains: joint positions (29 dims)
    - Total body: 35 dims
    - hand_left: 7 dims, hand_right: 7 dims, neck: 2 dims
    - Grand total: 51 dims
    """
    modality = {
        "state": {
            "root_vel": {"start": 0, "end": 2},
            "root_height": {"start": 2, "end": 3},
            "root_orientation": {"start": 3, "end": 5},
            "root_ang_vel": {"start": 5, "end": 6},
            "body": {"start": 6, "end": 35},
            "hand_left": {"start": 35, "end": 42},
            "hand_right": {"start": 42, "end": 49},
            "neck": {"start": 49, "end": 51},
        },
        "action": {
            "root_vel": {"start": 0, "end": 2},
            "root_height": {"start": 2, "end": 3},
            "root_orientation": {"start": 3, "end": 5},
            "root_ang_vel": {"start": 5, "end": 6},
            "body": {"start": 6, "end": 35},
            "hand_left": {"start": 35, "end": 42},
            "hand_right": {"start": 42, "end": 49},
            "neck": {"start": 49, "end": 51},
        },
        "video": {
            "rgb_left": {"original_key": "observation.images.rgb_left"},
            "rgb_right": {"original_key": "observation.images.rgb_right"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }
    
    meta_path = output_path / "meta"
    meta_path.mkdir(parents=True, exist_ok=True)
    with open(meta_path / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)
    print(f"Created modality.json at {meta_path / 'modality.json'}")


def main(
    input_dir: str,
    output_name: str = "twist2_dataset",
    output_dir: str | None = None,
    fps: int = 30,
    image_size: tuple[int, int] = (256, 256),
    push_to_hub: bool = False,
):
    """
    Convert TWIST2 dataset to LeRobot format.
    
    Always stores the full 51-dim state/action:
    - root_vel (2) + root_height (1) + root_orientation (2) + root_ang_vel (1) + body_joints (29) = 35
    - hand_left (7) + hand_right (7) + neck (2) = 16
    - Total: 51 dims
    
    The data_config handles which subset to use for training.
    
    Args:
        input_dir: Path to the TWIST2 dataset directory containing episode_XXXX folders
        output_name: Name for the output dataset
        output_dir: Custom output directory. If provided, dataset will be saved to output_dir/output_name.
                   If not provided, defaults to HF_LEROBOT_HOME/output_name (~/.cache/huggingface/lerobot/)
        fps: Frames per second of the dataset
        image_size: Target size for images (height, width)
        push_to_hub: Whether to push to Hugging Face Hub
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")
    
    # Find all episodes
    episodes = find_episodes(input_path)
    if not episodes:
        raise ValueError(f"No episodes found in {input_path}")
    print(f"Found {len(episodes)} episodes")
    
    # Always store full 51-dim state/action
    state_dim = TOTAL_DIM  # 51
    action_dim = TOTAL_DIM  # 51
    
    # Determine output path
    if output_dir is not None:
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        output_path = base_path / output_name
    else:
        output_path = HF_LEROBOT_HOME / output_name
    
    # Clean up any existing dataset
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset with features
    print(f"Creating LeRobot dataset at {output_path}")
    create_kwargs = {
        "repo_id": output_name,
        "robot_type": "humanoid",
        "fps": fps,
        "features": {
            "observation.images.rgb_left": {
                "dtype": "video",
                "shape": (image_size[0], image_size[1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.rgb_right": {
                "dtype": "video",
                "shape": (image_size[0], image_size[1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["action"],
            },
        },
        "image_writer_threads": 4,
        "image_writer_processes": 2,
    }
    
    # Add custom root path if output_dir is specified
    if output_dir is not None:
        create_kwargs["root"] = output_path
    
    dataset = LeRobotDataset.create(**create_kwargs)
    
    # Collect all unique tasks for task_index mapping
    all_tasks = []
    for episode_path in episodes:
        episode_data = load_episode_data(episode_path)
        task = episode_data.get("text", {}).get("goal", "default task")
        if task not in all_tasks:
            all_tasks.append(task)
    print(f"Found {len(all_tasks)} unique tasks")
    
    # Process each episode
    for episode_path in tqdm(episodes, desc="Converting episodes"):
        episode_data = load_episode_data(episode_path)
        
        # Get task description
        task = episode_data.get("text", {}).get("goal", "default task")
        
        # Process each frame
        frames = episode_data.get("data", [])
        for frame in frames:
            # Load images
            try:
                rgb_left = load_image(episode_path, frame["rgb_left"])
                rgb_right = load_image(episode_path, frame["rgb_right"])
            except FileNotFoundError as e:
                print(f"Warning: {e}, skipping frame")
                continue
            
            # Resize images if needed
            if rgb_left.shape[:2] != image_size:
                rgb_left = np.array(Image.fromarray(rgb_left).resize(
                    (image_size[1], image_size[0]), Image.BILINEAR
                ))
            if rgb_right.shape[:2] != image_size:
                rgb_right = np.array(Image.fromarray(rgb_right).resize(
                    (image_size[1], image_size[0]), Image.BILINEAR
                ))
            
            # Get state and action (always full 51 dims)
            state = concatenate_state(frame)
            action = concatenate_action(frame)
            
            # Add frame to dataset
            dataset.add_frame(
                {
                    "observation.images.rgb_left": rgb_left,
                    "observation.images.rgb_right": rgb_right,
                    "observation.state": state,
                    "action": action,
                },
                task=task,
            )
        
        # Save episode
        dataset.save_episode()
    
    # Create modality.json for GR00T compatibility (always stores full schema)
    create_modality_json(output_path)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Total episodes: {len(episodes)}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Optionally push to hub
    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["twist2", "humanoid", "gr00t"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Push complete!")


if __name__ == "__main__":
    tyro.cli(main)

# USAGE: Convert the Twist2 dataset to LeRobot format
# python scripts/convert_twist2_to_lerobot.py \
#     --input_dir ./datasets/1022_charlie_pick_brick \
#     --output_name charlie_pick_brick_lerobot \
#     --fps 30 \
#     --image_size 256 256
#
# With custom output directory:
# python scripts/convert_twist2_to_lerobot.py \
#     --input_dir ./datasets/1022_charlie_pick_brick \
#     --output_name charlie_pick_brick_lerobot \
#     --output_dir ./datasets \
#     --fps 30 \
#     --image_size 256 256