#!/bin/bash

# # conda activate /home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/env/cat-2
# export TRANSFORMERS_CACHE=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/transformers_cache
# export TORCH_HOME=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/torch_home
# export HF_HOME=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/hf_home
# export PIP_CACHE_DIR=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/pip
# export OPENAI_CACHE_DIR=/home/cxu-serve/p62/ytang37/projects/Caption-Anything-2/cache/openai

set -e

GREEN="\033[32m"
RESET="\033[0m"
FRAME_COUNT=32
OUTPUT_FOLDER="./results"
mkdir -p $OUTPUT_FOLDER
MODEL_PATH="OpenGVLab/InternVL2_5-8B-MPO" # "OpenGVLab/InternVL2-8B"
GET_BOUNDARY_MODEL_PATH="Yongxin-Guo/trace-uni"
GET_MASK_MODEL_PATH="./checkpoints/sam2.1_hiera_base_plus.pt"

############################################################################################################
VIDEO_NAME="demo.mp4"
VIDEO_FOLDER="./assets/"
OBJECT_BBOX="demo.txt"
QA_FILE_PATH="$OUTPUT_FOLDER/demo_boundary.json"
FINAL_JSON_PATH="$OUTPUT_FOLDER/demo_boundary_caption.json"
FINAL_VIDEO_PATH="$OUTPUT_FOLDER/demo_boundary_caption.mp4"
MASKED_VIDEO_PATH="$OUTPUT_FOLDER/demo_mask.mp4"
############################################################################################################

VIDEO_PATH="$VIDEO_FOLDER$VIDEO_NAME"
OBJECT_BBOX_PATH="$VIDEO_FOLDER$OBJECT_BBOX"

START_TIME=$(date +%s)

echo -e "${GREEN}Step 1: Parsing...${RESET}"

python -m scripts.get_boundary \
    --video_paths $VIDEO_PATH \
    --questions "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences." \
    --model_path $GET_BOUNDARY_MODEL_PATH \
    --load_4bit \
    --device_map auto

echo -e "${GREEN}Step 2: Segmentation...${RESET}"


python scripts/get_masks.py \
    --video_path "$VIDEO_PATH" \
    --txt_path "$OBJECT_BBOX_PATH" \
    --model_path "$GET_MASK_MODEL_PATH" \
    --video_output_path "$OUTPUT_FOLDER" \
    --save_to_video True

echo -e "${GREEN}Step 3: Captioning...${RESET}"

python scripts/get_caption.py \
    --model_path "$MODEL_PATH" \
    --QA_file_path "$QA_FILE_PATH" \
    --video_folder "$OUTPUT_FOLDER" \
    --answers_output_folder "$OUTPUT_FOLDER" \
    --extract_frames_method "max_frames_num" \
    --max_frames_num "$FRAME_COUNT" \
    --frames_from "video" \
    --final_json_path "$FINAL_JSON_PATH" \
    --provide_boundaries

echo -e "${GREEN}Step 3: Generate visualizations...${RESET}"

# python scripts/get_vis.py "$MASKED_VIDEO_PATH" "$FINAL_JSON_PATH" "$FINAL_VIDEO_PATH"

echo -e "${GREEN}Completed in $(($(date +%s) - START_TIME)) seconds.${RESET}"