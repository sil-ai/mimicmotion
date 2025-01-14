import os
import sys
import argparse
import logging
import math
import boto3
import yaml
import torch.jit
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
from clearml import Dataset, Task, Logger
import numpy as np
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
from dotenv import load_dotenv
from mimicmotion.utils.geglu_patch import patch_geglu_inplace
from constants import ASPECT_RATIO
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

patch_geglu_inplace()

load_dotenv()

def download_file_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3',
            region_name='us-east-1',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    logger.info(f"Downloading {s3_key} from the bucket {bucket_name} to {local_path}")
    s3.download_file(bucket_name, s3_key, local_path)
    return local_path


def update_yaml_file(yaml_path, local_video_path, local_image_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    config['test_case'][0]['ref_video_path'] = local_video_path
    config['test_case'][0]['ref_image_path'] = local_image_path

    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    logger.info(f"Updated YAML file: {yaml_path}")


def set_up_media_logging():
    logger = Logger.current_logger()
    logger.set_default_upload_destination(uri=f"s3://sil-mimicmotion")
    return logger

def get_clearml_paths():
    dataset = Dataset.get(dataset_id="47cf215eb8e54f099b21cc2d17f3460d")
    dataset.get_mutable_local_copy(target_folder="./models", overwrite=True)
    return os.getcwd()



def preprocess(video_path, image_path, resolution=576, sample_stride=2):
    """Preprocess ref image pose and video pose."""
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels)  # (c, h, w)
    h, w = image_pixels.shape[-2:]
    if h > w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1


def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    """Run the MimicMotion pipeline."""
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    frames = pipeline(
        image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale,
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        _video_frames = video_frames[vid_idx, 1:]  # Omitir el primer frame

    return _video_frames


@torch.no_grad()
def main(args):
    """Pipeline principal."""
    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    pipeline = create_pipeline(infer_config, device)

    for task in infer_config.test_case:
        print("Task FPS: ", str(task.fps))
        if args.fps != "None":
            task.fps = task.fps * (float(args.fps) / 30) # normalize frame rate, with original default being 30fps
            print("Updated Task FPS: " + str(task.fps))
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path, task.ref_image_path,
            resolution=task.resolution, sample_stride=task.sample_stride
        )
        _video_frames = run_pipeline(
            pipeline, image_pixels, pose_pixels, device, task
        )
        save_path_pose = f"{args.output_dir}{os.path.basename(task.ref_video_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        save_to_mp4(
            _video_frames,
            save_path_pose,
            fps=task.fps,
        )
        base = os.getcwd()
        media_logger.report_media(
            local_path=os.path.join(base, save_path_pose),
            title=f"{os.path.basename(task.ref_video_path).split('.')[0]}",
            iteration=task.id,
            series="Inference"
        )


def set_logger(log_file=None, log_level=logging.INFO):
    """Configura el logger."""
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_file", type=str, default=None, help="Path to the log file")
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml", help="Path to the test.yaml configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Path to output directory")
    parser.add_argument("--no_use_float16", action="store_true", help="Whether to use float16 to speed up inference")
    parser.add_argument("--s3_video_key", type=str, required=True, help="S3 key for the video file")
    parser.add_argument("--s3_image_key", type=str, required=True, help="S3 key for the image file")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the S3 bucket")
    parser.add_argument("--fps", type=str, required=True, help="Input Video Frames per second")

    args = parser.parse_args()
    args_dict = vars(args)

    task_clearml = Task.init(project_name="MimicMotion", task_name="Inferencev3", task_type=Task.TaskTypes.inference)
    params = task_clearml.connect(args_dict)

    if task_clearml.is_main_task():
        task_clearml.set_base_docker(docker_image="alejandroquinterosil/clearml-image:mimicmotion")
        task_clearml.set_system_tags(["allow_vault_secrets"])
        task_clearml.execute_remotely(queue_name="production", exit_process=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    media_logger = set_up_media_logging()
    mimic_path = get_clearml_paths()

    args.log_file = params.get("log_file", args.log_file)
    args.inference_config = params.get("inference_config", args.inference_config)
    args.output_dir = params.get("output_dir", args.output_dir)
    args.no_use_float16 = params.get("no_use_float16", args.no_use_float16)
    args.s3_video_key = params.get("s3_video_key", args.s3_video_key)
    args.s3_image_key = params.get("s3_image_key", args.s3_image_key)
    args.bucket_name = params.get("bucket_name", args.bucket_name)
    args.fps = params.get("fps", args.fps)

    if not args.s3_video_key or not args.s3_image_key or not args.bucket_name:
        logger.error("Missing required arguments!")
        sys.exit(1)

    local_video_path = "assets/example_data/videos/" + os.path.basename(args.s3_video_key)
    local_image_path = "assets/example_data/images/" + os.path.basename(args.s3_image_key)

    download_file_from_s3(args.bucket_name, args.s3_video_key, local_video_path)
    download_file_from_s3(args.bucket_name, args.s3_image_key, local_image_path)

    yaml_path = args.inference_config
    update_yaml_file(yaml_path, local_video_path, local_image_path)

    print("---------------------------------------------------------------")
    print("Args: ", args)
    print("---------------------------------------------------------------")
    absolute_output_dir = mimic_path + "/" + args.output_dir
    Path(absolute_output_dir).mkdir(parents=False, exist_ok=True)
    set_logger(args.log_file \
               if args.log_file is not None else f"{absolute_output_dir}{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    main(args)
    logger.info(f"--- Finished ---")


