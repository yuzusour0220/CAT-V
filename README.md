![Cover](assets/cover.png)

Official PyTorch implementation of [Caption Anything in Video: Fine-grained Object-centric Captioning via Spatiotemporal Multimodal Prompting](https://arxiv.org/abs/2504.05541)

![cat-v-framework](assets/cat-v-framework.png)

## 🚀 Updates

## 🕹️ Demo

YouTube: [https://youtu.be/2eiPVKXEoxw](https://youtu.be/2eiPVKXEoxw)

## 🛠️ Getting Started

Follow these steps to reproduce the demo or adapt the model to your own videos.

### 1. Create an environment

```bash
conda create -n cat-v python=3.10 -y
conda activate cat-v
```

You may also build an environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate cat-v
```

### 2. Install dependencies

Install the project and all Python requirements:

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Optional cache directories can be set to speed up downloads. Add the following
to your shell profile if desired:

```bash
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export TORCH_HOME=~/.cache/torch
```

### 3. Download checkpoints

```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

### 4. Verify the installation (optional)

```bash
python -m scripts.get_boundary --help
```

## 🏃 Inference

### Quick demo

Run the full pipeline on the provided example video:

```bash
bash inference.sh
```

The script performs three stages—temporal boundary detection, object
segmentation and caption generation—and stores intermediate and final results
in the `results/` directory.

### Step-by-step usage

1. **Boundary detection**

   ```bash
   python -m scripts.get_boundary \
       --video_paths assets/demo.mp4 \
       --questions "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences." \
       --model_path Yongxin-Guo/trace-uni
   ```

   Produces `results/demo_boundary.json` containing temporal segments.

2. **Object segmentation**

   Provide a text file with bounding boxes (`x_min,y_min,x_max,y_max` per line).

   ```bash
   python scripts/get_masks.py \
       --video_path assets/demo.mp4 \
       --txt_path assets/demo.txt \
       --model_path checkpoints/sam2.1_hiera_base_plus.pt \
       --video_output_path results \
       --save_to_video True
   ```

   Generates `results/demo_mask.mp4` with the tracked object mask.

3. **Caption generation**

   ```bash
   python scripts/get_caption.py \
       --model_path OpenGVLab/InternVL2_5-8B-MPO \
       --QA_file_path results/demo_boundary.json \
       --video_folder results \
       --answers_output_folder results \
       --extract_frames_method max_frames_num \
       --max_frames_num 32 \
       --frames_from video \
       --final_json_path results/demo_boundary_caption.json \
       --provide_boundaries
   ```

   Produces captions for each segment in
   `results/demo_boundary_caption.json`.

4. **Visualization (optional)**

   ```bash
   python scripts/get_vis.py results/demo_mask.mp4 results/demo_boundary_caption.json results/demo_boundary_caption.mp4
   ```

### Gradio demo

Launch an interactive web demo for uploading custom videos and drawing the
object of interest on the first frame:

```bash
python gradio_app.py
```


## 📖 Citation
If you find this work useful for your research or applications, please cite using this BibTeX:

```bibtex
@inproceedings{tang2025cat-v,
  title={Caption Anything in Video: Fine-grained Object-centric Captioning via Spatiotemporal Multimodal Prompting},
  author={Tang, Yunlong and Bi, Jing and Hua, Hang and Xiao, Yunzhong and Song, Yizhi and Liu, Pinxin and Huang, Chao and Feng, Mingqian and Guo, Junjia and Liu, Zhuo and Song, Luchuan and Liang, Susan and Shimada, Daiki and Vosoughi, Ali and He, Jinxi and He, Liu and Zhang, Zeliang and Luo, Jiebo and Xu, Chenliang},
  journel={arXiv},
  year={2025}
}
```


## 🙏 Acknowledgements
This work was supported by Sony Group Corporation. We would like to thank Sayaka Nakamura and Jerry Jun Yokono for their insightful discussion.

We are also grateful for the following awesome projects our CAT-V arising from:

- [Caption Anything](https://github.com/ttengwang/Caption-Anything)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [SAMURAI](https://github.com/yangchris11/samurai)
- [TRACE-uni](https://github.com/gyxxyg/TRACE)
- [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [InternVL-2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)


## 👩‍💻 Contributors
Our project wouldn't be possible without the contributions of these amazing people! Thank you all for making this project better.

- [Yunlong Tang](https://yunlong10.github.io/) @ University of Rochester
- [Jing Bi](https://scholar.google.com/citations?user=ZyCYhUkAAAAJ) @ University of Rochester
- [Chao Huang](https://wikichao.github.io/) @ University of Rochester
- [Susan Liang](https://liangsusan-git.github.io/) @ University of Rochester
- [Daiki Shimada](https://scholar.google.co.jp/citations?user=1uAwouQAAAAJ) @ Sony Group Corporation
- [Hang Hua](https://hanghuacs.notion.site/Hang-Hua-151c5b68f62980e8884febf1b5c1d4a9) @ University of Rochester
- [Yunzhong Xiao](https://shawn-yzxiao.github.io/) @ Carnegie Mellon University
- [Yizhi Song](https://song630.github.io/yizhisong.github.io/) @ Purdue University
- [Pinxin Liu](https://andypinxinliu.github.io/) @ University of Rochester
- [Mingqian Feng](https://fmmarkmq.github.io/) @ University of Rochester
- [Junjia Guo](https://doujiangter.github.io/JunjiaGuo.github.io/) @ University of Rochester
- [Zhuo Liu](https://joeliuz6.github.io/) @ University of Rochester
- [Luchuan Song](https://songluchuan.github.io/) @ University of Rochester
- [Ali Vosoughi](https://alivosoughi.com/) @ University of Rochester
- [Jinxi He](https://gingin520.github.io/) @ University of Rochester
- [Liu He](https://arking1995.github.io/) @ Purdue University
- [Zeliang Zhang](https://zhangaipi.github.io/) @ University of Rochester
- [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/) @ University of Rochester
- [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/index.html) @ University of Rochester



<a href="https://github.com/yunlong10/CAT-V/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yunlong10/CAT-V" />
</a>

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yunlong10/CAT-V&type=Date)](https://star-history.com/#yunlong10/CAT-V&Date)
# CAT-V
