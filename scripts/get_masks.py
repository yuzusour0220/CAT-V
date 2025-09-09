import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
from tqdm import tqdm
import sys
sys.path.append("./")
from sam2.build_sam import build_sam2_video_predictor

# カラー定義（マスク描画に使用する色リスト）
color = [(255, 0, 0)]

def load_txt(gt_path):
    # ground truth テキストを読み込み、各行の bbox を辞書形式で返す
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x_min, y_min, x_max, y_max = line.strip().split(",")
        # x, y, w, h = int(x), int(y), int(w), int(h)
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        # prompts は {frame_index: ((x_min,y_min,x_max,y_max), class_id)} の形を返す
        prompts[fid] = ((x_min, y_min, x_max, y_max), 0)
    return prompts

def determine_model_cfg(model_path):
    # モデルのパスからモデルサイズを判定し、対応する設定ファイルを返す
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    # 入力が mp4 ファイルか、フレームディレクトリかを検証してそのまま返す
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    # メイン処理:
    # 1) モデル設定を決定して predictor を構築
    # 2) 入力を準備（動画フレームまたはフレームディレクトリ）
    # 3) 指定された初期 bbox を使ってマスクを生成し、ビデオへ書き出す（必要なら）
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)
    print(prompts)

    # decide fps for writing/display
    fps = float(args.fps)
    if isinstance(frames_or_path, str) and frames_or_path.endswith(".mp4"):
        cap_meta = cv2.VideoCapture(frames_or_path)
        vfps = cap_meta.get(cv2.CAP_PROP_FPS)
        if vfps and vfps > 1e-3:
            fps = float(vfps)
        cap_meta.release()

    # 動画に書き出す場合は全フレームを読み込み、出力ビデオの幅高さを決める
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.lower().endswith(".jpg")])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            if len(loaded_frames) == 0 or loaded_frames[0] is None:
                raise ValueError("No frames were loaded from the frame directory.")
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            if len(loaded_frames) == 0 or loaded_frames[0] is None:
                raise ValueError("No frames were loaded from the video.")
            height, width = loaded_frames[0].shape[:2]

    # 出力用 VideoWriter を準備（mp4, 固定フレームレート 30fps）
    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = args.video_output_path + f"/{osp.basename(args.video_path).split('.')[0]}_mask.mp4"
        out = cv2.VideoWriter(
            out_path,
            fourcc,
            fps,
            (width, height),
        )

    # 推論は autocast と inference_mode で高速化/メモリ効率化
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # predictor の状態を初期化（動画を読み込ませる）
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)

        # 総フレーム数（開始インデックス決定用）
        num_frames = state["num_frames"] if isinstance(state, dict) and "num_frames" in state else None
        if num_frames is None:
            # fallback: 推定
            num_frames = len(loaded_frames) if args.save_to_video else 0

        # 開始フレーム決定
        start_idx = 0
        if args.start_frame_idx is not None:
            start_idx = int(args.start_frame_idx)
        elif args.start_time is not None:
            start_idx = int(round(float(args.start_time) * fps))
        start_idx = max(0, min(start_idx, max(0, num_frames - 1)))

        # 開始フレームに対応する bbox を選ぶ（無ければ 0 行目や先頭行をフォールバック）
        if start_idx in prompts:
            init_box, _ = prompts[start_idx]
        elif 0 in prompts:
            init_box, _ = prompts[0]
        else:
            # prompts の最初の要素
            init_box, _ = next(iter(prompts.values()))

        # 開始フレームでボックスを与える
        _ = predictor.add_new_points_or_box(state, box=init_box, frame_idx=start_idx, obj_id=0)

        # 収集用ストレージ（あとで時間順に書き出し）
        collected = {}

        def collect_one(frame_idx, object_ids, masks):
            mask_to_vis = {}
            bbox_to_vis = {}
            for obj_id, mask in zip(object_ids, masks):
                mask_np = mask[0].detach().cpu().numpy()
                mask_np = mask_np > 0.0
                non_zero_indices = np.argwhere(mask_np)
                if len(non_zero_indices) == 0:
                    bb = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bb = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bb
                mask_to_vis[obj_id] = mask_np
            collected[frame_idx] = (mask_to_vis, bbox_to_vis)

        # 前方（>= start_idx）へ伝播
        for fidx, object_ids, masks in tqdm(
            predictor.propagate_in_video(state, start_frame_idx=start_idx, reverse=False)
        ):
            collect_one(fidx, object_ids, masks)

        # 後方（< start_idx）へ伝播（重複を避けるため start_idx-1 から開始）
        if getattr(args, "bidirectional", False) and start_idx > 0:
            for fidx, object_ids, masks in tqdm(
                predictor.propagate_in_video(state, start_frame_idx=start_idx - 1, reverse=True)
            ):
                collect_one(fidx, object_ids, masks)

        # ビデオ保存: フレーム0→N-1の順でオーバレイ
        if args.save_to_video:
            warned_oob = False
            total_frames_to_write = len(loaded_frames)
            for fidx in range(total_frames_to_write):
                img = loaded_frames[fidx]
                if fidx in collected:
                    mask_to_vis, bbox_to_vis = collected[fidx]
                    for obj_id, mask_np in mask_to_vis.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask_np] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)
                    for obj_id, bb in bbox_to_vis.items():
                        cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color[obj_id % len(color)], 2)
                else:
                    # 未推論フレーム（通常は起きないが fps ズレ等に備えてそのまま書く）
                    if not warned_oob:
                        print("[INFO] Some frames may not have masks; writing raw frames for them.")
                        warned_oob = True
                out.write(img)

            out.release()

    # 後片付け：メモリ解放
    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # コマンドライン引数の定義（デフォルトはデモ用のパス）
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./assets/demo.mp4", help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", default="./assets/demo.txt", help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="./checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="./results/", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    parser.add_argument("--start_time", type=float, default=None, help="Start time in seconds to place the initial bbox.")
    parser.add_argument("--start_frame_idx", type=int, default=None, help="Start frame index (0-based) for the initial bbox.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for frame-directory inputs; for mp4, fps is auto-detected if possible.")
    parser.add_argument("--bidirectional", action="store_true", help="Propagate segmentation in both time directions.")
    args = parser.parse_args()
    main(args)
