import os
import sys
# 実行ディレクトリに依存せずローカルモジュールを import できるように、
# プロジェクト直下を Python のモジュール検索パスに追加する設定
script_dir = os.path.dirname(os.path.abspath(__file__))
sys_paths = [
    os.path.abspath(os.path.join(script_dir, '../'))
]
print(sys_paths)
for sys_path in sys_paths:
    if sys_path not in sys.path:
        sys.path.append(sys_path)
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as T
import json
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from eval_utils import get_answers_output_path, gen_QAs_dataloader, unpack_QAs, basic_parser

# 画像前処理で用いる ImageNet の平均・標準偏差（RGB）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

""""""
def build_transform(input_size):
    # 入力画像をモデルに渡すための標準的な前処理を構築
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        # RGB 以外のモードで来た場合は RGB に変換
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        # 正方形にリサイズ（バイキュービック補間）
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        # Tensor 化（[0,1] にスケーリング）
        T.ToTensor(),
        # ImageNet の統計量で正規化
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

""""""
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    # 入力画像のアスペクト比に最も近い (cols, rows) の分割パターンを選ぶ
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
            # 同差の場合は、元画像面積と image_size をもとに閾値で優先度を微調整
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

""""""
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    # 指定の時間範囲 bound=[start,end] 秒（なければ全体）から、
    # 均等に `num_segments` 個のフレームインデックスをサンプリング
    if bound:
        start, end = float(bound[0]), float(bound[1])
    else:
        start, end = -100000, 100000
    # 秒→フレームに変換し、範囲と上限でクリップ
    start_idx = max(first_idx, int(start * fps))
    end_idx = min(int(end * fps), max_frame)
    # セグメント幅（フレーム数）
    seg_size = float(end_idx - start_idx) / num_segments
    # 各セグメントの中心付近のフレームを選択
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

""""""
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    # 画像のアスペクト比に応じてグリッド分割し、`image_size` のタイル群を作る
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    # 総タイル数が [min_num, max_num] を満たす (cols, rows) の候補集合を作成
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    # 入力画像のアスペクト比に最も近い分割（cols, rows）を選ぶ
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    # 選んだ分割数に合わせて、リサイズ後の全体サイズを計算
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    # まず画像全体を (target_width, target_height) にリサイズ
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        # i 番目のタイルに対応する領域（左上原点のグリッド）を計算
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        # 指定領域をクロップし、個別タイル画像を作成
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        # タイルが複数ある場合のみ、元画像のサムネイル（image_size 四方）を末尾に追加
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

""""""
def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, save_dir=None):
    # 動画を読み込み、サンプリングした各フレームをタイル分割＋前処理して結合
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    # optional: prepare saving sampled frames
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i, frame_index in enumerate(frame_indices, start=1):
        # decord のフレームを numpy → PIL に変換し、RGB 化
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        # save the exact frames fed to LLM (before tiling/transform)
        if save_dir is not None:
            fname = f"F{i:04d}_idx{frame_index:06d}.jpg"
            img.save(os.path.join(save_dir, fname), quality=95)
        # タイル分割（必要に応じてサムネイルも追加）
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        # タイルごとに前処理を適用してテンソル化
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list
if __name__ == "__main__":
    # 引数定義の取得とパース
    parser = basic_parser()
    # 追加引数: キャプションで使用したフレームの保存先
    parser.add_argument('--save_caption_frames_dir', type=str, default=None,
                        help='If set, saves only the frames actually fed into the LLM for captioning.')
    args = parser.parse_args()
    # 結果 JSON の保存先パスを決定
    answers_output_path = get_answers_output_path(args)
    # QA 入力の DataLoader を作成
    dataloader = gen_QAs_dataloader(args)
    # 事前学習済みのビジョン言語モデルをロード（GPU / bfloat16）
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    # 同じパスからトークナイザをロード
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    # 生成設定：貪欲探索、最大 1024 トークン
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    answers = []
    event_list = []
    # duration = 0
    for idx, (QA_data, _) in enumerate(tqdm(dataloader)):
        # DataLoader から 1 サンプル分を展開
        final_question, video_path, short_answer, video, segment, question = unpack_QAs(QA_data)
        # モデルに HO（ハイライト対象オブジェクト）中心の詳細な段落キャプション生成を指示
        final_question = """ Please pay attention to the object highlighted (HO) by colored bounding box and blue mask in the video frames, and generate accurate object-centric caption for the HO. Please make sure in object-centric paragraph caption, the sentences should be detailed and specific, and the subjects of all sentences **MUST be "HO"**. Please follow the format:



 HO: ...



 HO's itself attributes: ...



 All actions done by HO: ...



 All statuses of HO: ...



 All other objects interacted with HO: ...



 All environments/backgrounds of HO: ...



 All events related to HO: ...



 Final object-centric paragraph caption: The HO is [attributes], [environment]. From ... to ...s, the HO [status], [any action], [any status/attribute/environment changes]. From ... to ...s, the HO [status], [any action], [any status/attribute/environment changes]. From ... to ...s, the HO [status], [any action], [any status/attribute/environment changes]. The OH's [final status]. Crate a caption as detailed as possible by describing every single action."""
        # event = f'From {segment[0]} to {segment[1]}s, {short_answer}'
        # TAからのモデルからのshort answerを含まない
        # event = f'From {segment[0]} to {segment[1]}s, event{idx}'
        # セグメント情報からイベント文を入れる設計だが、ここでは空のまま
        event = ""
        event_list.append(event)
        # duration = max(duration, float(segment[1]))

    # print(event_list)
    # 指定フレーム数（max_frames_num）だけ均等サンプリングして前処理
    pixel_values, num_patches_list = load_video(
        video_path,
        num_segments=args.max_frames_num,
        max_num=1,
        save_dir=args.save_caption_frames_dir,
    )
    n_frames = len(num_patches_list)
    # 型変換と GPU 転送
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    # LLM にフレーム画像列を知らせるためのテキストプレフィックス
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    # 追加イベント（現状は空文字）を連結
    video_prefix += ''.join([f'{event}\n' for event in event_list])
    # プレフィックスと質問文を結合して最終プロンプトを作成
    question_tmp = video_prefix + final_question
    print(question_tmp)
    try:
        # 画像テンソルとテキストを入力して応答を生成
        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values, question_tmp, generation_config)
        print(response)
        # このサンプルの出力を集約
        answers.append({
            "video": video,
            "segment": segment,
            "question": question,
            # "options": options, 
            # "task_class": task_class, 
            "short_answer": short_answer,
            "model_answer": response           
        })
    except Exception as e:
        print(f"Error encountered at idx {idx}: {e}")
        # エラーが起きた場合も形式を揃えて記録
        answers.append({
            "video": video,
            "segment": segment,
            "question": question,
            # "options": options, 
            # "task_class": task_class, 
            "short_answer": short_answer,
            "model_answer": '<error_processing>'           
        })
    # exit()



    # for idx, (QA_data, _) in enumerate(tqdm(dataloader)):
    #     # print(QA_data)
    #     # continue
    #     try:
    #         final_question, video_path, short_answer, \
    #             video, segment, question = unpack_QAs(QA_data)
    #         # print(video)
    #         # print(segment, question)
    #         print(f'From {segment[0]} to {segment[1]}s,', short_answer)
    #         print(final_question)
    #         if args.provide_boundaries:
    #             pixel_values, num_patches_list = load_video(video_path, bound=segment, num_segments=args.max_frames_num, max_num=1)
    #         else:
    #             pixel_values, num_patches_list = load_video(video_path, num_segments=args.max_frames_num, max_num=1)
    #         # exit()
    #         print(num_patches_list)
    #         exit()
    #         pixel_values = pixel_values.to(torch.bfloat16).cuda()
    #         video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    #         question_tmp = video_prefix + final_question
    #         # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
    #         with torch.inference_mode():
    #             response = model.chat(tokenizer, pixel_values, question_tmp, generation_config,
    #                             num_patches_list=num_patches_list)
    #         print(response)
    #         answers.append({
    #             "video": video,
    #             "segment": segment,
    #             "question": question,
    #             # "options": options, 
    #             # "task_class": task_class, 
    #             "short_answer": short_answer,
    #             "model_answer": response           
    #         })

    #     except Exception as e:
    #         print(f"Error encountered at idx {idx}: {e}")
    #         answers.append({
    #             "video": video,
    #             "segment": segment,
    #             "question": question,
    #             # "options": options, 
    #             # "task_class": task_class, 
    #             "short_answer": short_answer,
    #             "model_answer": '<error_processing>'           
    #         })
    # exit()
    # 結果を 2 箇所に保存（評価用の標準出力先と、明示的パス）
    with open(answers_output_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)
    with open(args.final_json_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)
