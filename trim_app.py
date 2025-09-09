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
    st.caption("ffmpeg を使用して動画の一部分だけを書き出します。単一/複数ファイル対応。")

    # Check tools
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    if not ffmpeg_ok:
        st.error("ffmpeg が見つかりません。インストールしてからお試しください。")
    if not ffprobe_ok:
        st.warning("ffprobe が見つからないため、正確な長さ取得ができない場合があります。")

    target_mode = st.radio("対象", ["単一動画", "複数動画"], horizontal=True)
    mode = st.radio("入力方法", ["ファイルをアップロード", "ローカルのパスを指定"], horizontal=True)

    # 単一動画モード
    if target_mode == "単一動画":
        in_path: Path | None = None
        tmpfile: tempfile.NamedTemporaryFile | None = None

        if mode == "ファイルをアップロード":
            up = st.file_uploader(
                "動画ファイル", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"]
            )
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

        return

    # 複数動画モード
    in_paths: list[Path] = []
    tmpfiles: list[tempfile.NamedTemporaryFile] = []

    if mode == "ファイルをアップロード":
        ups = st.file_uploader(
            "動画ファイル（複数可）",
            type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
            accept_multiple_files=True,
        )
        if ups:
            for up in ups:
                suffix = Path(up.name).suffix or ".mp4"
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tf.write(up.read())
                tf.flush()
                tmpfiles.append(tf)
                in_paths.append(Path(tf.name))
            st.success(f"{len(in_paths)} 件のファイルを読み込みました。")
            # 代表で最初の動画をプレビュー
            if in_paths:
                st.video(str(in_paths[0]))
    else:
        raw = st.text_area("動画のローカルパス（1行1ファイル）", value="", height=120)
        if raw.strip():
            for line in raw.splitlines():
                p = Path(line.strip()).expanduser()
                if p.exists():
                    in_paths.append(p)
                else:
                    st.warning(f"見つかりません: {line}")
            if in_paths:
                st.success(f"{len(in_paths)} 件のファイルを検出しました。")
                st.video(str(in_paths[0]))

    if not in_paths:
        st.stop()

    # スライダー範囲の決定：既知の長さの最小値を上限に使用
    durs = [probe_duration(p) for p in in_paths]
    known = [d for d in durs if d is not None]
    if not known:
        st.info("長さが取得できないファイルが含まれるため、0〜任意の範囲で指定してください。")
        max_v = 36000.0
        label = "トリミング範囲（秒）"
    else:
        max_v = float(max(0.1, min(known)))
        label = f"トリミング範囲（0〜{human_time(max_v)}、最短動画に合わせています）"

    start_s, end_s = st.slider(
        label,
        min_value=0.0,
        max_value=max_v,
        value=(0.0, min(10.0, max_v)),
        step=0.1,
    )

    # 代表プレビュー（最初の1本のみ）
    if ffmpeg_ok and end_s > start_s and in_paths:
        try:
            stt = os.stat(in_paths[0])
            file_sig = int(stt.st_mtime_ns) ^ int(stt.st_size)
        except Exception:
            file_sig = 0
        c1, c2 = st.columns(2)
        for (label2, t), col in zip([("開始", start_s), ("終了", end_s)], [c1, c2]):
            with col:
                t_q = round(float(t), 2)
                img = get_frame_cached(str(in_paths[0]), file_sig, t_q)
                if img:
                    st.image(img, caption=f"{label2}: {human_time(t_q)}", use_column_width=True)
                else:
                    st.caption(f"{label2}: {human_time(t_q)}（プレビュー不可）")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("開始", human_time(start_s))
    with col2:
        st.metric("終了", human_time(end_s))
    with col3:
        st.metric("長さ", human_time(max(0.0, end_s - start_s)))

    # 出力先
    if mode == "ファイルをアップロード":
        out_dir = Path(tempfile.mkdtemp())
        st.caption(f"一時出力フォルダ: {out_dir}")
    else:
        out_dir = Path("./output").resolve()
        st.caption(f"出力フォルダ: {out_dir}")

    suffix_name = st.text_input("ファイル名サフィックス", value="_trimmed")

    if st.button("一括トリミングを実行", type="primary", disabled=not ffmpeg_ok):
        results: list[tuple[Path, bool, str]] = []
        prog = st.progress(0, text="処理中...")
        for idx, src in enumerate(in_paths, start=1):
            out_path = out_dir / f"{src.stem}{suffix_name}.mp4"
            ok, msg = run_trim(src, out_path, start_s, end_s)
            results.append((out_path, ok, msg))
            prog.progress(int(idx / len(in_paths) * 100), text=f"{idx}/{len(in_paths)} 完了")
        prog.empty()

        n_ok = sum(1 for _, ok, _ in results if ok)
        n_ng = len(results) - n_ok
        if n_ok:
            st.success(f"完了: {n_ok} 件成功、{n_ng} 件失敗。")
        else:
            st.error("全て失敗しました。詳細を確認してください。")

        for out_path, ok, msg in results:
            if ok and out_path.exists():
                with st.expander(out_path.name, expanded=False):
                    st.video(str(out_path))
                    try:
                        data = out_path.read_bytes()
                        st.download_button(
                            "ダウンロード",
                            data=data,
                            file_name=out_path.name,
                            mime="video/mp4",
                        )
                    except Exception:
                        st.write("保存先:", str(out_path))
            else:
                st.warning(f"失敗: {msg}")


if __name__ == "__main__":
    main()
