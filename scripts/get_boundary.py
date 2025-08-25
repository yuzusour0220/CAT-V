import torch
import transformers
import json
import sys
import os
import argparse

# sys.path に親ディレクトリを追加してローカルモジュールを import 可能にする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trace.conversation import conv_templates, SeparatorStyle
from trace.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from trace.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token_all, process_video, process_image, KeywordsStoppingCriteria
from trace.model.builder import load_pretrained_model


def inference(args):
    # Video Inference のための引数を準備（複数に対応しているがここでは先頭を使用）
    paths = args.video_paths
    questions = args.questions
    modal_list = ['video']

    # 1. モデルの初期化：パスからモデル名を取得し、トークナイザー・モデル・プロセッサを読み込む
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, device='cuda', device_map={'': 'cuda:0'})
    conv_mode = 'llama_2'

    # 2. ビジュアル前処理（動画または画像を読み込み、モデル入力用に変換）
    if modal_list[0] == 'video':
        # 動画をフレーム列とタイムスタンプに変換する関数を呼ぶ
        tensor, video_timestamps = process_video(paths[0], processor, model.config.image_aspect_ratio, num_frames=64)
        # モデルへ投入する際のデータ型とデバイスを指定
        tensor = tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        # 画像として処理する場合（ここでは video の分岐が優先される）
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]

    # モデルが期待する形に合わせてリストに包む
    tensor = [tensor]
    video_timestamps = [video_timestamps]
    heads = [1]

    # 3. テキスト前処理（マルチモーダルトークンを付与し、会話テンプレートに挿入してプロンプトを生成）
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # モデルが同期トークンでタイムスタンプ等を返す想定で、末尾に <sync> を追加
    prompt += '<sync>'
    print(prompt)
    # マルチモーダル用のトークナイズ（特殊トークンを全て処理）
    input_ids = tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to('cuda')
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    # 停止条件（会話の区切り文字）を設定
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    do_sample = True

    # 4. モデルによる生成（画像/動画とテキストを渡して応答を生成）
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            video_timestamps=video_timestamps,
            heads=heads
        )

    # 5. 出力の解析：生成された token id 列を走査してタイムスタンプ・スコア・キャプションを抽出
    outputs = {
        'timestamps': [],
        'scores': [],
        'captions': [],
    }
    cur_timestamps = []
    cur_timestamp = []
    cur_scores = []
    cur_score = []
    cur_caption = []
    # 出力 token id は特定の範囲で意味を持つ（値域で判別）
    for idx in output_ids[0]:
        if idx <= 32000:
            # 32000 はキャプション区切り（文の終端）を示す特別な id
            if idx == 32000:
                new_caption = tokenizer.decode(cur_caption, skip_special_tokens=True)
                outputs['captions'].append(new_caption)
                cur_caption = []
            else:
                # 通常のテキスト token id はそのままキャプションのバッファへ
                cur_caption.append(idx)
        elif idx <= 32013: # 32001..32013 はタイムスタンプ関連のトークン範囲
            # ここでは時間に関するトークンをデコードして数値（文字列の連結）として保持
            if idx == 32001:
                # 32001 はタイムスタンプの区切り（セグメント終了）を示す
                if len(cur_timestamp) > 0:
                    cur_timestamps.append(float(''.join(cur_timestamp)))
                outputs['timestamps'].append(cur_timestamps)
                cur_timestamps = []
                cur_timestamp = []
            elif idx == 32002:
                # 32002 はタイムスタンプ内の区切り（開始と終了の間）を示す
                if len(cur_timestamp) > 0:
                    cur_timestamps.append(float(''.join(cur_timestamp)))
                cur_timestamp = []
            else:
                # それ以外は時間トークンをデコードして文字列として蓄積
                cur_timestamp.append(model.get_model().time_tokenizer.decode(idx - 32001))
        else: # 32014 以降はスコア関連のトークン範囲
            if idx == 32014:
                # スコアの区切り（セグメント終了）を示す
                if len(cur_score) > 0:
                    cur_scores.append(float(''.join(cur_score)))
                outputs['scores'].append(cur_scores)
                cur_scores = []
                cur_score = []
            elif idx == 32015:
                # スコア内の区切り（複数スコアの区切り）を示す
                if len(cur_score) > 0:
                    cur_scores.append(float(''.join(cur_score)))
                cur_score = []
            else:
                # スコアトークンをデコードして文字列として蓄積
                cur_score.append(model.get_model().score_tokenizer.decode(idx - 32014))
    # ループ終了後、残ったキャプションがあれば追加
    if len(cur_caption):
        outputs['captions'].append(tokenizer.decode(cur_caption, skip_special_tokens=True))

    # 6. 出力を JSON 形式で保存（タイムスタンプと対応するキャプションを results に格納）
    try:
        results = []
        for i in range(len(outputs['timestamps'])):
            output = {
                'video': paths[0].split("/")[-1][:-4] + "_mask.mp4",
                'segment': f"{outputs['timestamps'][i][0]}_{outputs['timestamps'][i][1]}",
                'question': "",
                'answer': outputs['captions'][i],
            }
            results.append(output)

        with open(f'./results/{paths[0].split("/")[-1].split(".")[0]}_boundary.json', 'w') as f:
            json.dump(results, f)
    
    except Exception as e:
        # 保存に失敗した場合はエラーメッセージを出力し、代替の空結果を保存
        print(e)
        print("Failed to save the output to a json file.")
        with open(f'./results/{paths[0].split("/")[-1].split(".")[0]}_boundary.json', 'w') as f:
            json.dump([{"video": paths[0].split("/")[-1], "segment": f"0.0_{video_timestamps[0][1]}", "question": "", "answer": ""}], f)


if __name__ == "__main__":
    # コマンドライン引数のパース：動画パス、質問、モデルパスを受け取る
    parser = argparse.ArgumentParser(description="Inference script for boundary detection.")
    parser.add_argument("--video_paths", nargs='+', required=True, help="Paths to the input video files.")
    parser.add_argument("--questions", nargs='+', required=True, help="Questions for video inference.")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model.")
    args = parser.parse_args()

    inference(args)
