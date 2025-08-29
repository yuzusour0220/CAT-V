import os
import sys
import argparse
import json
from typing import List, Tuple, Optional, Callable

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModel, AutoTokenizer


# ImageNet normalization (RGB)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[Tuple[int, int]],
                                width: int, height: int, image_size: int) -> Tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12,
                       image_size: int = 448, use_thumbnail: bool = False) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if (i * j) <= max_num and (i * j) >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def _uniform_frame_indices(bound: Optional[Tuple[float, float]], fps: float, max_frame: int,
                            first_idx: int = 0, num_segments: int = 32) -> np.ndarray:
    if bound:
        start, end = float(bound[0]), float(bound[1])
    else:
        start, end = -100000.0, 100000.0
    start_idx = max(first_idx, int(start * fps))
    end_idx = min(int(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / max(1, num_segments)
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def _fps_frame_indices(video_fps: float, max_frame: int, target_fps: float, max_frames_limit: int) -> np.ndarray:
    if target_fps <= 0:
        # fallback to uniform if invalid
        return np.linspace(0, max_frame, num=max_frames_limit, dtype=int)
    step = max(int(round(video_fps / target_fps)), 1)
    idx = np.arange(0, max_frame + 1, step, dtype=int)
    if len(idx) > max_frames_limit:
        # downsample uniformly to match limit
        idx = np.linspace(0, max_frame, num=max_frames_limit, dtype=int)
    return idx


def load_video_as_patches(
    video_path: str,
    bound: Optional[Tuple[float, float]] = None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 16,
    strategy: str = 'uniform',
    target_fps: float = 0.0,
    use_thumbnail: bool = True,
) -> Tuple[torch.Tensor, List[int]]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    transform = build_transform(input_size=input_size)
    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []

    # Choose sampling strategy
    if strategy == 'fps' and target_fps > 0:
        frame_indices = _fps_frame_indices(fps, max_frame, target_fps, max_frames_limit=num_segments)
    else:
        frame_indices = _uniform_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def build_prompt(num_frames: int, lang: str = 'ja', for_video: bool = False) -> str:
    if for_video:
        prefix = ''
    else:
        prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])

    instr = (
        'This is one video. Provide a concise yet detailed caption '
        'describing the main actions, objects, and scene changes.'
    )
    return prefix + instr


def try_chat_video(model, tokenizer, video_path: str, prompt: str, generation_config: dict,
                   verbose: bool = False) -> Optional[str]:
    if verbose:
        try:
            attrs = [a for a in dir(model) if 'video' in a.lower() or 'videos' in a.lower()]
            print(f"[try_chat_video] Model video-related attrs: {attrs}")
        except Exception as e:
            print(f"[try_chat_video] Could not inspect model attrs: {e}")
    attempts: List[Tuple[str, Callable[[], str]]] = [
        (
            'chat_video(tokenizer, video_path, prompt, generation_config)',
            lambda: model.chat_video(tokenizer, video_path, prompt, generation_config),
        ),
        (
            'chat_video(tokenizer, videos=..., question=..., generation_config=...)',
            lambda: model.chat_video(tokenizer, videos=video_path, question=prompt, generation_config=generation_config),
        ),
        (
            'chat_video(tokenizer, video=..., question=..., generation_config=...)',
            lambda: model.chat_video(tokenizer, video=video_path, question=prompt, generation_config=generation_config),
        ),
        (
            'chat_video(tokenizer, videos=[...], question=..., generation_config=...)',
            lambda: model.chat_video(tokenizer, videos=[video_path], question=prompt, generation_config=generation_config),
        ),
        (
            'chat_video(tokenizer, video=[...], question=..., generation_config=...)',
            lambda: model.chat_video(tokenizer, video=[video_path], question=prompt, generation_config=generation_config),
        ),
        (
            'chat(tokenizer, video_path, prompt, generation_config)',
            lambda: model.chat(tokenizer, video_path, prompt, generation_config),
        ),
        (
            'chat(tokenizer, videos=..., question=..., generation_config=...)',
            lambda: model.chat(tokenizer, videos=video_path, question=prompt, generation_config=generation_config),
        ),
        (
            'chat(tokenizer=..., videos=..., question=..., generation_config=...)',
            lambda: model.chat(tokenizer=tokenizer, videos=video_path, question=prompt, generation_config=generation_config),
        ),
        (
            'chat_video(tokenizer=..., videos=..., question=..., generation_config=...)',
            lambda: model.chat_video(tokenizer=tokenizer, videos=video_path, question=prompt, generation_config=generation_config),
        ),
        (
            'chat(tokenizer, video=..., question=..., generation_config=...)',
            lambda: model.chat(tokenizer, video=video_path, question=prompt, generation_config=generation_config),
        ),
        (
            'chat(tokenizer, videos=[...], question=..., generation_config=...)',
            lambda: model.chat(tokenizer, videos=[video_path], question=prompt, generation_config=generation_config),
        ),
        (
            'chat(tokenizer, video=[...], question=..., generation_config=...)',
            lambda: model.chat(tokenizer, video=[video_path], question=prompt, generation_config=generation_config),
        ),
    ]
    for label, fn in attempts:
        try:
            if verbose:
                print(f'[try_chat_video] Attempt: {label}')
            return fn()
        except Exception as e:
            if verbose:
                print(f'[try_chat_video] Failed: {label}: {type(e).__name__}: {e}')
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description='InternVL video captioning. Use direct video when supported.')
    parser.add_argument('--model_path', type=str, required=True, help='Hugging Face model path, e.g., OpenGVLab/InternVL2-8B')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--max_frames', type=int, default=16, help='Number of frames to sample (higher = better but heavier)')
    parser.add_argument('--input_size', type=int, default=448, help='Input resolution per tile (square)')
    parser.add_argument('--lang', type=str, default='ja', choices=['ja', 'en'], help='Output language for caption')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt. If set, overrides default prompt builder')
    parser.add_argument('--output_json', type=str, default=None, help='Optional path to save result JSON')
    parser.add_argument('--method', type=str, default='auto', choices=['video', 'frames', 'auto'],
                        help='video: force direct video API; frames: force frame sampling; auto: try video then fallback to frames')
    parser.add_argument('--verbose', action='store_true', help='Print verbose logs about method selection and attempts')
    parser.add_argument('--local_files_only', action='store_true', help='Load models/tokenizers from local cache only')
    parser.add_argument('--hf_home', type=str, default=None, help='Override HF_HOME for model cache inside container')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Override device selection')
    parser.add_argument('--dtype', type=str, default=None, choices=['bf16', 'fp16', 'fp32'], help='Override torch dtype')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens for generation')
    parser.add_argument('--quality', type=str, default=None, choices=['high', 'balanced', 'fast'],
                        help='Preset for max_frames/max_new_tokens: high(48/512), balanced(24/256), fast(16/128)')
    parser.add_argument('--device_map', type=str, default='auto',
                        choices=['auto', 'balanced', 'balanced_low_0', 'sequential', 'off'],
                        help='Use HF accelerate to shard model across GPUs. Use off to disable and rely on --device.')
    parser.add_argument('--frames_strategy', type=str, default='uniform', choices=['uniform', 'fps'],
                        help='How to sample frames from video when using frames method')
    parser.add_argument('--fps', type=float, default=0.0, help='When frames_strategy=fps, target frames-per-second')
    parser.add_argument('--tiles_max_num', type=int, default=1, help='Max number of tiles per frame (dynamic_preprocess)')
    parser.add_argument('--no_thumbnail', action='store_true', help='Disable appending thumbnail tile when tiling > 1')
    # CUDA environment controls
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='Set CUDA_VISIBLE_DEVICES (e.g., "0,2")')
    parser.add_argument('--cuda_device_order', type=str, default=None, help='Set CUDA_DEVICE_ORDER (e.g., "PCI_BUS_ID")')

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f'Video not found: {args.video_path}')

    # Configure HF cache
    if args.hf_home:
        os.environ['HF_HOME'] = args.hf_home
    # Configure CUDA env if requested (must be before any CUDA context usage)
    if args.cuda_visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
        if args.verbose:
            print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if args.cuda_device_order:
        os.environ['CUDA_DEVICE_ORDER'] = args.cuda_device_order
        if args.verbose:
            print(f"[env] CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}")
    # Select device and dtype
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dtype == 'bf16':
        preferred_dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        preferred_dtype = torch.float16
    elif args.dtype == 'fp32':
        preferred_dtype = torch.float32
    else:
        preferred_dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    # Apply quality preset if requested
    if args.quality is not None:
        if args.quality == 'high':
            args.max_frames = 48
            args.max_new_tokens = max(args.max_new_tokens, 512)
        elif args.quality == 'balanced':
            args.max_frames = 24
            args.max_new_tokens = max(args.max_new_tokens, 256)
        elif args.quality == 'fast':
            args.max_frames = 16
            args.max_new_tokens = min(args.max_new_tokens, 128) if args.max_new_tokens else 128

    # Load model/tokenizer
    if args.device_map != 'off':
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=preferred_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            device_map=args.device_map,
        ).eval()
    else:
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=preferred_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        ).eval()
        if device == 'cuda':
            model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=args.local_files_only,
    )

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    # Choose method: prefer direct video when requested/available
    response: Optional[str] = None
    if args.method in ('video', 'auto'):
        # Build a video-style prompt (no Frame markers)
        prompt_text = args.prompt.strip() if args.prompt else build_prompt(num_frames=0, lang=args.lang, for_video=True)
        with torch.inference_mode():
            response = try_chat_video(model, tokenizer, args.video_path, prompt_text, generation_config, verbose=args.verbose)
        if response is not None:
            result = {
                'video': os.path.basename(args.video_path),
                'model': args.model_path,
                'method': 'video',
                'frames': None,
                'prompt': prompt_text,
                'caption': response,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.output_json:
                os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            return

        if args.method == 'video':
            # user forced video; do not fallback silently
            raise RuntimeError('Direct video API is not available for this model (or signature mismatch). Try --method frames.')

    # Fallback to frame-based approach
    pixel_values, num_patches_list = load_video_as_patches(
        args.video_path,
        bound=None,
        input_size=args.input_size,
        max_num=args.tiles_max_num,
        num_segments=args.max_frames,
        strategy=args.frames_strategy,
        target_fps=args.fps,
        use_thumbnail=not args.no_thumbnail,
    )
    # Place inputs on appropriate device/dtype
    if args.device_map != 'off' and torch.cuda.is_available():
        input_device = torch.device('cuda:0')
    else:
        input_device = torch.device(device)
    pixel_values = pixel_values.to(dtype=preferred_dtype, device=input_device)

    if args.prompt is not None and len(args.prompt.strip()) > 0:
        prompt = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]) + args.prompt.strip()
    else:
        prompt = build_prompt(num_frames=len(num_patches_list), lang=args.lang, for_video=False)

    with torch.inference_mode():
        try:
            response = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                generation_config,
                num_patches_list=num_patches_list,
            )
        except TypeError:
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    result = {
        'video': os.path.basename(args.video_path),
        'model': args.model_path,
        'method': 'frames',
        'frames': len(num_patches_list),
        'prompt': prompt,
        'caption': response,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
