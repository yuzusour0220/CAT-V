#!/bin/bash

export TRANSFORMERS_CACHE=/home/cxu-serve/p62/ytang37/projects/AIGC/.cache/transformers_cache
export TORCH_HOME=/home/cxu-serve/p62/ytang37/projects/AIGC/.cache/torch_home
export HF_HOME=/home/cxu-serve/p62/ytang37/projects/AIGC/.cache/hf_home
export PIP_CACHE_DIR=/home/cxu-serve/p62/ytang37/projects/AIGC/.cache/pip
export OPENAI_CACHE_DIR=/home/cxu-serve/p62/ytang37/projects/AIGC/.cache/openai

python -m scripts.inference.inference