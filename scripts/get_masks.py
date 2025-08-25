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

    # 動画に書き出す場合は全フレームを読み込み、出力ビデオの幅高さを決める
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(".jpg")])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
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
            height, width = loaded_frames[0].shape[:2]
            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    # 出力用 VideoWriter を準備（mp4, 固定フレームレート 30fps）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path+f"/{osp.basename(args.video_path).split('.')[0]}_mask.mp4", fourcc, 30, (width, height))

    # 推論は autocast と inference_mode で高速化/メモリ効率化
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # predictor の状態を初期化（動画を読み込ませる）
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        # 初期フレームの bbox を与えて対象オブジェクトを登録
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        
        # propagate_in_video は (frame_idx, object_ids, masks) を逐次返すジェネレータ
        for frame_idx, object_ids, masks in tqdm(predictor.propagate_in_video(state)):
            # マスクと bbox を可視化する準備
            mask_to_vis = {}
            bbox_to_vis = {}

            # 各オブジェクトについてマスクを numpy に変換し bbox を計算
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            # ビデオ保存フラグが立っていればフレームにマスクと bbox を描画して書き出す
            if args.save_to_video:
                img = loaded_frames[frame_idx]
                # マスクを色付きで重ねる（alpha は 0.2）
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)
            
                # bbox を描画
                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

               # add the text to the bottom of EACH frame
                # The text depends on the current frame's position in the video, with one decimal place reserved for seconds
                # The font color is red, and the text is centered (so you should calculate the len of text) on the bottom like subtitle, occupying 1/5 of the frame height.
                # The text is displayed in the format "103.5s"
                # time = frame_idx / 30
                # time_text = f"{time:.1f}s"
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 4
                # font_thickness = 8
                # font_color = (0, 0, 255)
                # text_size = cv2.getTextSize(time_text, font, font_scale, font_thickness)[0]
                # text_x = (width - text_size[0]) // 2
                # text_y = height - 5
                # cv2.putText(img, time_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

                out.write(img)

        if args.save_to_video:
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
    args = parser.parse_args()
    main(args)
