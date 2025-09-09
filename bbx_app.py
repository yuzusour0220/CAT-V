# Video Bounding Box Selector (fit-to-screen, back-scale coords)
# conda:
#   conda create -n bboxapp python=3.10 -y && conda activate bboxapp
#   conda install -c conda-forge streamlit opencv pillow numpy -y
#   pip install streamlit-drawable-canvas
# run:
#   streamlit run app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import json
import math
import os

st.set_page_config(page_title="Video Bounding Box Selector", page_icon="📦", layout="wide")

st.title("📦 Video Bounding Box Selector")
st.write("任意フレームを **画面にフィット表示** し、矩形を描いて座標を取得します。表示は縮小されますが、座標は**元解像度**で出力します。長尺動画でも1枚だけデコードするのでメモリ消費は一定です。")
st.caption("座標系: 左上が (0,0)。x→右、y→下。")

uploaded = st.file_uploader(
    "動画ファイルをアップロード",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="長い動画可。内部で一時ファイルへ保存し、選択したフレームのみデコードします。"
)

def save_uploaded_file(uploaded_file) -> str:
    """アップロードファイルを一時保存しパスを返す（大きい動画でもチャンク書き込み）。"""
    suffix = os.path.splitext(getattr(uploaded_file, "name", "video"))[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # 位置リセット（再利用時）
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        while True:
            chunk = uploaded_file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        return tmp.name

@st.cache_data(show_spinner=False)
def get_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else None
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
    }

@st.cache_data(show_spinner=False)
def load_frame(path: str, frame_index: int):
    """指定フレームをPIL Image (RGB) で返す。失敗時 None。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if uploaded is not None:
    # 新しいファイルなら一時保存
    if "_video_tmp_path" not in st.session_state or st.session_state.get("_video_name") != uploaded.name:
        st.session_state["_video_tmp_path"] = save_uploaded_file(uploaded)
        st.session_state["_video_name"] = uploaded.name
    video_path = st.session_state["_video_tmp_path"]

    meta = get_video_meta(video_path)
    if meta is None:
        st.error("動画を開けませんでした。")
        st.stop()
    frame_count = meta["frame_count"]
    fps = meta["fps"]
    duration = meta["duration"]
    orig_w = meta["width"]
    orig_h = meta["height"]

    with st.expander("動画メタデータ", expanded=False):
        show_meta = meta.copy()
        if show_meta.get("duration") is not None:
            show_meta["duration"] = round(show_meta["duration"], 2)
        show_meta["fps"] = round(show_meta["fps"], 3)
        st.json(show_meta)

    st.subheader("フレーム選択")
    method = st.radio("選択方法", ["フレーム番号", "時間 (秒)"], horizontal=True)
    if method == "フレーム番号":
        if frame_count <= 20000:
            sel_frame = st.slider("フレーム", 0, max(0, frame_count - 1), 0)
        else:
            sel_frame = st.number_input(f"フレーム (0~{frame_count-1})", min_value=0, max_value=max(0, frame_count-1), value=0, step=1)
    else:
        if duration is None:
            st.warning("FPS 未取得のため時間指定は利用できません。フレーム番号で選択してください。")
            sel_frame = 0
        else:
            sec = st.slider("時間 (秒)", 0.0, max(0.0, float(duration)), 0.0)
            sel_frame = min(frame_count - 1, int(round(sec * fps)))
    st.caption(f"選択中フレーム: {sel_frame} / {frame_count-1}")

    img = load_frame(video_path, int(sel_frame))
    if img is None:
        st.error("フレーム読み込みに失敗しました。別のフレームを選ぶか動画を確認してください。")
        st.stop()

    st.subheader("表示設定")
    # 画面に収めやすい基準幅（px）。必要ならここを変更。
    default_fit_width = 1200
    fit_width = st.number_input("フィット先の幅 (px)", min_value=400, max_value=4000, value=default_fit_width, step=50)
    # 追加で倍率を微調整
    zoom_pct = st.slider("表示倍率 (%)", 10, 200, 100, 5)

    # 表示サイズの計算（元→表示）
    base_scale = min(1.0, fit_width / orig_w)  # まずは画面幅に収める
    scale = base_scale * (zoom_pct / 100.0)
    disp_w = max(1, int(round(orig_w * scale)))
    disp_h = max(1, int(round(orig_h * scale)))

    # 表示用に縮小した背景画像を作成
    bg_img = img.resize((disp_w, disp_h), Image.LANCZOS)

    st.subheader("選択フレーム（全体が見えるように縮小表示）")
    st.caption(f"元解像度: {orig_w}×{orig_h}px / 表示: {disp_w}×{disp_h}px（縮尺 {scale:.3f}） / フレーム {sel_frame}")

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=bg_img,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        display_toolbar=True,
        key="canvas",
    )

    st.info("ドラッグで矩形を描いたら、下のボタンを押してください。複数描いた場合は最後の矩形。長尺動画では必要なフレームを選んでから描画してください。キーフレーム間隔が長いコーデックではランダムアクセスが遅い場合があります。")

    if st.button("座標を出力"):
        data = canvas_result.json_data if canvas_result is not None else None
        if not data or "objects" not in data or len(data["objects"]) == 0:
            st.warning("矩形が見つかりません。描画してからボタンを押してください。")
        else:
            rect = None
            for obj in data["objects"]:
                if obj.get("type") == "rect":
                    rect = obj  # 最後のrectが採用される
            if rect is None:
                st.warning("矩形が見つかりません。")
            else:
                # キャンバス（表示サイズ基準）上の値
                left_disp = float(rect.get("left", 0.0))
                top_disp = float(rect.get("top", 0.0))
                w_disp = float(rect.get("width", 0.0)) * float(rect.get("scaleX", 1.0))
                h_disp = float(rect.get("height", 0.0)) * float(rect.get("scaleY", 1.0))

                # 表示→元解像度へ逆変換
                inv = 1.0 / scale
                x_min = int(round(left_disp * inv))
                y_min = int(round(top_disp * inv))
                x_max = int(round((left_disp + w_disp) * inv))
                y_max = int(round((top_disp + h_disp) * inv))

                # 画像範囲でクリップ
                x_min = max(0, min(orig_w, x_min))
                y_min = max(0, min(orig_h, y_min))
                x_max = max(0, min(orig_w, x_max))
                y_max = max(0, min(orig_h, y_max))

                # 念のため整列（min<=max）
                if x_min > x_max: x_min, x_max = x_max, x_min
                if y_min > y_max: y_min, y_max = y_max, y_min

                coords = {"frame": int(sel_frame), "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

                st.success(f"frame={sel_frame} : x_min, y_min, x_max, y_max = {x_min}, {y_min}, {x_max}, {y_max}")
                st.json(coords)

                # ダウンロード
                csv_str = f"frame,x_min,y_min,x_max,y_max\n{sel_frame},{x_min},{y_min},{x_max},{y_max}\n"
                st.download_button("CSVをダウンロード", data=csv_str, file_name="bbox.csv", mime="text/csv")
                st.download_button("JSONをダウンロード", data=json.dumps(coords, ensure_ascii=False), file_name="bbox.json", mime="application/json")
else:
    st.info("上のボックスから動画ファイルをアップロードしてください。")
