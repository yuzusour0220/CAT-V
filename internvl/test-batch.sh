#!/bin/bash
export TRANSFORMERS_CACHE=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/transformers_cache
export TORCH_HOME=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/torch_home
export HF_HOME=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/hf_home
export PIP_CACHE_DIR=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/pip
export OPENAI_CACHE_DIR=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/openai

# Exit immediately if a command exits with a non-zero status.
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_FILE_PATH="${SCRIPT_DIR}/test.py"
ANSWERS_OUTPUT_FOLDER="${SCRIPT_DIR}/results"  

if [ -e "$ANSWERS_OUTPUT_FOLDER" ]; then
    echo "File $ANSWERS_OUTPUT_FOLDER already exists."
else
    echo "File $ANSWERS_OUTPUT_FOLDER does not exist. Creating it now..."
    mkdir "$ANSWERS_OUTPUT_FOLDER"
    echo "File $ANSWERS_OUTPUT_FOLDER created."
fi

# Variables - Please update these paths according to your setup
MODEL_PATH="OpenGVLab/InternVL2-8B"
VIDEO_FOLDER="/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/samurai/samed_videos"

# Define task files and frame numbers
TASK_FILES=("example.json")
FRAME_COUNTS=("8")

# Loop through each task file and each frame count
for TASK in "${TASK_FILES[@]}"; do
    for FRAME_COUNT in "${FRAME_COUNTS[@]}"; do
        QA_FILE_PATH="/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/samurai/QAs/$TASK"
        
        # Execute the Python script with the provided arguments
        python "$PYTHON_FILE_PATH" \
            --model_path "$MODEL_PATH" \
            --QA_file_path "$QA_FILE_PATH" \
            --video_folder "$VIDEO_FOLDER" \
            --answers_output_folder "$ANSWERS_OUTPUT_FOLDER" \
            --extract_frames_method "max_frames_num" \
            --max_frames_num "$FRAME_COUNT" \
            --frames_from "video" \
            # --provide_boundaries
    done
done
