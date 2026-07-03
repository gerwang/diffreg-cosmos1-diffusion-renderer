# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import torch

from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import GBUFFER_INDEX_MAPPING
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataset_inference import VideoFramesDataset
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import dict_collation_fn, dict_collation_fn_concat, sample_continuous_keys

from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.io import save_video, save_image_or_video
from cosmos_predict1.diffusion.inference.variable_batch_sampler import VariableBatchSampler

torch.enable_grad(False)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add Diffusion Renderer specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Diffusion_Renderer_Inverse_Cosmos_7B",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Diffusion_Renderer_Inverse_Cosmos_7B",
        ],
    )

    parser.add_argument(
        "--inference_passes",
        type=str,
        default=["basecolor", "normal", "depth", "roughness", "metallic"],
        nargs="+",
        help=(
            "List of G-buffer passes to infer. "
            "Options typically include: basecolor, normal, depth, roughness, metallic. "
            "Default: basecolor normal depth roughness metallic"
        )
    )
    parser.add_argument(
        "--normalize_normal",
        type=str2bool,
        default=False,
        help="If True, normal maps are normalized to unit vectors before saving. Default: False."
    )
    parser.add_argument(
        "--save_image",
        type=str2bool,
        default=True,
        help="If True, saves each output frame as an image file. Default: True."
    )
    parser.add_argument(
        "--save_video",
        type=str2bool,
        default=True,
        help="If True, saves the output as a video file. Default: True."
    )
    parser.add_argument(
        "--skip_existing",
        type=str2bool,
        default=False,
        help="If True, skips samples whose output files already exist for a pass."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the input data. Can be a directory containing video frames or images, or a dataset name."
    )
    parser.add_argument(
        "--overlap_n_frames",
        type=int,
        default=0,
        help="Number of overlapping frames between consecutive video chunks. Useful for sliding window processing of videos."
    )
    parser.add_argument(
        "--chunk_mode",
        type=str,
        default='first',
        choices=['first', 'all', 'drop_last', 'all_fill_last'],
        help="How to select video chunks from the dataset: 'first' uses only the first chunk, 'all' uses all possible chunks, 'drop_last' drops the last incomplete chunk."
    )
    parser.add_argument(
        "--group_mode",
        type=str,
        default='folder',
        choices=['folder', 'webdataset', 'image'],
        help=(
            "How to group input frames: 'folder' loads frames from a directory, "
            "'webdataset' groups frames following the convention of webdataset, "
            "'image' treats each image file as a separate sample."
        )
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=None,
        nargs="+",
        help="List of allowed image file extensions to load (e.g., jpg, png, jpeg). If not set, uses default supported types."
    )
    parser.add_argument(
        "--resize_resolution",
        type=int,
        default=None,
        nargs="+",
        help="Resize input images to this resolution before other processing, e.g. center crop. Provide as two integers: height width. If not set, uses original image size."
    )
    parser.add_argument(
        "--clip_name_prefix",
        type=str,
        default=None,
        help="Optional prefix to prepend to clip names for output paths (useful when dataset_path points to a subdirectory)."
    )

    return parser.parse_args()


def _outputs_exist_for_pass(args: argparse.Namespace, data_batch: dict, gbuffer_pass: str) -> bool:
    if not args.save_image and not args.save_video:
        return False

    for j in range(len(data_batch["clip_name"])):
        clip_name = data_batch["clip_name"][j]
        chunk_ind_str = data_batch["chunk_index"][j] if "chunk_index" in data_batch else "0000"
        if args.save_image:
            for ind in range(args.num_video_frames):
                save_path = os.path.join(
                    args.video_save_folder,
                    "gbuffer_frames",
                    f"{clip_name}/{chunk_ind_str}.{ind:04d}.{gbuffer_pass}.png",
                )
                if not os.path.exists(save_path):
                    return False
        if args.save_video:
            clip_name_flat = clip_name.replace("/", "__")
            video_save_path = os.path.join(
                args.video_save_folder, f"{clip_name_flat}.{gbuffer_pass}.mp4"
            )
            if not os.path.exists(video_save_path):
                return False
    return True


def demo(args: argparse.Namespace):
    """Run diffusion renderer inference.

    Args:
        args: Command line arguments
    """
    misc.set_random_seed(args.seed)

    # Initialize renderer pipeline
    pipeline = DiffusionRendererPipeline(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
    )

    # Prepare input data
    dataset = VideoFramesDataset(
        root_dir=args.dataset_path,
        sample_n_frames=args.num_video_frames,
        overlap_n_frames=args.overlap_n_frames,
        chunk_mode=args.chunk_mode,
        group_mode=args.group_mode,
        image_extensions=args.image_extensions,
        resolution=[args.height, args.width],
        resize_resolution=args.resize_resolution,
        clip_name_prefix=args.clip_name_prefix,
    )
    batch_sizes = [dataset.batch_num_chunks_per_video(
        i) for i in range(len(dataset.video_groups))]
    sampler = VariableBatchSampler(len(dataset), batch_sizes)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=0, pin_memory=True, collate_fn=dict_collation_fn)

    # Create output directory
    os.makedirs(args.video_save_folder, exist_ok=True)

    # Generate output
    n_test = len(dataloader)
    iter_dataloader = iter(dataloader)
    for i in range(n_test):
        data_batch = next(iter_dataloader)

        for gbuffer_pass in args.inference_passes:
            if args.skip_existing and _outputs_exist_for_pass(args, data_batch, gbuffer_pass):
                clip_names = ", ".join(data_batch["clip_name"])
                log.info(
                    f"Skipping pass {gbuffer_pass} for clip(s) {clip_names} (outputs exist)."
                )
                continue
            # overwrite specified G-buffer passes
            context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
            data_batch["context_index"].fill_(context_index)

            outputs = pipeline.generate_video(
                data_batch=data_batch,
                normalize_normal=(
                    gbuffer_pass == 'normal' and args.normalize_normal),
            )

            for j in range(outputs.shape[0]):
                output = outputs[j]  # (T, H, W, C)
                # Save output as individual frames
                if args.save_image:
                    video_relative_base_name = data_batch['clip_name'][j]
                    chunk_ind_str = data_batch['chunk_index'][j] if 'chunk_index' in data_batch else '0000'
                    for ind in range(output.shape[0]):  # (T, H, W, C)
                        save_path = os.path.join(args.video_save_folder, "gbuffer_frames",
                                                 f"{video_relative_base_name}/{chunk_ind_str}.{ind:04d}.{gbuffer_pass}.png")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        save_image_or_video(
                            video_save_path=save_path,
                            video=output[ind:ind + 1, ...],
                            H=args.height,
                            W=args.width,
                        )
                # Save output as video
                if args.save_video:
                    clip_name = data_batch['clip_name'][j].replace("/", "__")
                    video_save_path = os.path.join(
                        args.video_save_folder, f"{clip_name}.{gbuffer_pass}.mp4")
                    save_image_or_video(
                        video_save_path=video_save_path,
                        video=output,
                        fps=args.fps,
                        H=args.height,
                        W=args.width,
                        video_save_quality=5,
                    )
                    log.info(f"Saved video to {video_save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
