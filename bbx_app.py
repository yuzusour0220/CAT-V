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

st.set_page_config(page_title="Video Bounding Box Selector", page_icon="📦", layout="wide")

st.title("📦 Video Bounding Box Selector")
st.write("動画の1フレーム目を **画面にフィット表示** し、矩形を描いて座標を取得します。表示は縮小されますが、座標は**元解像度**で出力します。")
st.caption("座標系: 左上が (0,0)。x→右、y→下。")

uploaded = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi", "mkv", "webm"])

def extract_first_frame(file_bytes: bytes):
    """Bytesから1フレーム目をPIL.Image（RGB）で返す"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

if uploaded is not None:
    img = extract_first_frame(uploaded.read())
    if img is None:
        st.error("フレーム抽出に失敗しました。別の動画でお試しください。")
        st.stop()

    orig_w, orig_h = img.width, img.height

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

    st.subheader("1フレーム目（全体が見えるように縮小表示）")
    st.caption(f"元解像度: {orig_w}×{orig_h}px / 表示: {disp_w}×{disp_h}px（縮尺 {scale:.3f}）")

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

    st.info("ドラッグで矩形を描いたら、下のボタンを押してください。複数描いた場合は最後の矩形を採用します。")

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

                coords = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

                st.success(f"x_min, y_min, x_max, y_max = {x_min}, {y_min}, {x_max}, {y_max}")
                st.json(coords)

                # ダウンロード
                csv_str = f"x_min,y_min,x_max,y_max\n{x_min},{y_min},{x_max},{y_max}\n"
                st.download_button("CSVをダウンロード", data=csv_str, file_name="bbox.csv", mime="text/csv")
                st.download_button("JSONをダウンロード", data=json.dumps(coords, ensure_ascii=False),
                                   file_name="bbox.json", mime="application/json")
else:
    st.info("上のボックスから動画ファイルをアップロードしてください。")
