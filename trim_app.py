import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st


def human_time(seconds: float) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def probe_duration(path: Path) -> float | None:
    """Return duration in seconds using ffprobe, or None if unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def run_trim(in_path: Path, out_path: Path, start_s: float, end_s: float) -> tuple[bool, str]:
    """Trim via ffmpeg with compatibility fallbacks. Returns (ok, msg)."""
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg が見つかりません。インストールしてください。"
    start_s = max(0.0, float(start_s))
    end_s = max(0.0, float(end_s))
    if end_s <= start_s:
        return False, "終了時間は開始時間より後にしてください。"
    duration = end_s - start_s

    out_path.parent.mkdir(parents=True, exist_ok=True)

    commands = []

    # 0) Fast stream copy (keyframe単位での切り出し。最も互換性が高い)
    commands.append([
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-i", str(in_path),
        "-t", f"{duration:.3f}", "-map", "0",
        "-c", "copy",
        str(out_path),
    ])

    # 1) libx264（利用可能なら高品質）
    commands.append([
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-ss", f"{start_s:.3f}", "-t", f"{duration:.3f}", "-map", "0",
        "-c:v", "libx264", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ])

    # 2) libx264（古い指定方法）
    commands.append([
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-ss", f"{start_s:.3f}", "-t", f"{duration:.3f}", "-map", "0",
        "-vcodec", "libx264", "-crf", "20",
        "-acodec", "aac", "-b:a", "192k",
        str(out_path),
    ])

    # 3) mpeg4（多くの ffmpeg で利用可）
    commands.append([
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-ss", f"{start_s:.3f}", "-t", f"{duration:.3f}", "-map", "0",
        "-vcodec", "mpeg4", "-q:v", "3",
        "-acodec", "aac", "-b:a", "192k",
        str(out_path),
    ])

    # 4) mpeg4 + mp3（AAC が無い場合のフォールバック）
    commands.append([
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-ss", f"{start_s:.3f}", "-t", f"{duration:.3f}", "-map", "0",
        "-vcodec", "mpeg4", "-q:v", "3",
        "-acodec", "libmp3lame", "-b:a", "192k",
        str(out_path),
    ])

    # 5) さらに古い ffmpeg（-ab 指定）
    commands.append([
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-ss", f"{start_s:.3f}", "-t", f"{duration:.3f}", "-map", "0",
        "-vcodec", "mpeg4", "-q:v", "3",
        "-acodec", "aac", "-ab", "192k",
        str(out_path),
    ])

    last_err = None
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, str(out_path)
        except subprocess.CalledProcessError as e:
            last_err = e

    stderr = getattr(last_err, "stderr", "").strip() if 'last_err' in locals() else ""
    return False, f"ffmpeg エラー: {stderr or str(last_err)}"


def _extract_frame_bytes(in_path: Path, t_sec: float) -> bytes | None:
    """Return JPEG bytes of a frame near t_sec using ffmpeg image2pipe.
    Uses fast seek (ss before -i) for snappy preview.
    """
    if shutil.which("ffmpeg") is None:
        return None
    t = max(0.0, float(t_sec))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(in_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "2",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True)
        return proc.stdout if proc.stdout else None
    except subprocess.CalledProcessError:
        return None


@st.cache_data(show_spinner=False)
def get_frame_cached(path_str: str, file_sig: int, t_quant: float) -> bytes | None:
    """Cache preview frames by file signature and quantized time."""
    return _extract_frame_bytes(Path(path_str), t_quant)


def main():
    st.set_page_config(page_title="動画トリミング", page_icon="✂️", layout="centered")
    st.title("✂️ 動画トリミング（シンプル）")
    st.caption("ffmpeg を使用して動画の一部分だけを書き出します。")

    # Check tools
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    if not ffmpeg_ok:
        st.error("ffmpeg が見つかりません。インストールしてからお試しください。")
    if not ffprobe_ok:
        st.warning("ffprobe が見つからないため、正確な長さ取得ができない場合があります。")

    mode = st.radio("入力方法", ["ファイルをアップロード", "ローカルのパスを指定"], horizontal=True)

    in_path: Path | None = None
    tmpfile: tempfile.NamedTemporaryFile | None = None

    if mode == "ファイルをアップロード":
        up = st.file_uploader("動画ファイル", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"])  # noqa: E231
        if up is not None:
            suffix = Path(up.name).suffix or ".mp4"
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmpfile.write(up.read())
            tmpfile.flush()
            in_path = Path(tmpfile.name)
            st.video(str(in_path))
    else:
        raw = st.text_input("動画のローカルパス", value="")
        if raw:
            cand = Path(raw).expanduser()
            if cand.exists():
                in_path = cand
                st.video(str(in_path))
            else:
                st.error("指定されたパスが見つかりません。")

    if in_path is None:
        st.stop()

    dur = probe_duration(in_path)
    if dur is None:
        st.info("長さを取得できなかったため、0秒〜任意の範囲で指定してください。")
        start_s, end_s = st.slider(
            "トリミング範囲（秒）",
            min_value=0.0,
            max_value=36000.0,
            value=(0.0, 10.0),
            step=0.1,
        )
    else:
        start_s, end_s = st.slider(
            f"トリミング範囲（0〜{human_time(dur)}）",
            min_value=0.0,
            max_value=float(max(0.1, dur)),
            value=(0.0, min(10.0, float(dur))),
            step=0.1,
        )

    # Live preview frames for the selected range
    if ffmpeg_ok and in_path.exists() and end_s > start_s:
        try:
            stt = os.stat(in_path)
            file_sig = int(stt.st_mtime_ns) ^ int(stt.st_size)
        except Exception:
            file_sig = 0
        t_vals = [
            ("開始", start_s),
            ("終了", end_s),
        ]
        c1, c2 = st.columns(2)
        cols = [c1, c2]
        for (label, t), col in zip(t_vals, cols):
            with col:
                t_q = round(float(t), 2)
                img = get_frame_cached(str(in_path), file_sig, t_q)
                if img:
                    st.image(img, caption=f"{label}: {human_time(t_q)}", use_column_width=True)
                else:
                    st.caption(f"{label}: {human_time(t_q)}（プレビュー不可）")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("開始", human_time(start_s))
    with col2:
        st.metric("終了", human_time(end_s))
    with col3:
        st.metric("長さ", human_time(max(0.0, end_s - start_s)))

    default_out = (in_path.stem + "_trimmed.mp4")

    if mode == "ファイルをアップロード":
        out_dir = Path(tempfile.mkdtemp())
    else:
        out_dir = Path("./output").resolve()

    out_name = st.text_input("出力ファイル名", value=default_out)
    out_path = out_dir / out_name

    if st.button("トリミングを実行", type="primary", disabled=not ffmpeg_ok):
        with st.spinner("トリミング中..."):
            ok, msg = run_trim(in_path, out_path, start_s, end_s)
        if ok:
            st.success("完了しました！")
            st.video(str(out_path))
            try:
                data = out_path.read_bytes()
                st.download_button(
                    "トリミング結果をダウンロード",
                    data=data,
                    file_name=out_path.name,
                    mime="video/mp4",
                )
            except Exception:
                st.write("保存先:", str(out_path))
        else:
            st.error(msg)

    # Cleanup temp file on session end
    if tmpfile is not None:
        # We don't unlink immediately because the user may download the result
        pass


if __name__ == "__main__":
    main()
