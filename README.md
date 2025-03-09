# CAT-2
Official PyTorch implementation of the paper "Caption Anything in Video: Object-centric Dense Video Captioning with Multimodal Controls"

## 🚀 Updates

## 🕹️ Demo

## 🛠️ Getting Started

1. Set up a conda environment (python>= 3.10) using:

```bash
conda create -n cat2 python=3.10 -y
conda activate cat2
```

2. Install the requirements:

```bash
pip install -e .
```

3. Download checkpoints

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## 🏃 RUN

```
bash inference.sh
```


## 📖 Citation
If you find this work useful for your research or applications, please cite using this BibTeX:

```bibtex
@inproceedings{tang2024caption,
  title={Caption Anything in Video: Object-centric Dense Video Captioning with Multimodal Controls},
  author={Tang, Yunlong and Bi, Jing and Hua, Hang and Xiao, Yunzhong and Song, Yizhi and Liu, Pinxin and Huang, Chao and Feng, Mingqian and Guo, Junjia and Liu, Zhuo and Song, Luchuan and Liang, Susan and Shimada, Daiki and Vosoughi, Ali and Zhang, Zeliang and Luo, Jiebo and Xu, Chenliang},
  journel={arXiv},
  year={2024}
}
```


## 🙏 Acknowledgements
We are grateful for the following awesome projects our CAT-2 arising from:

- LLaVA: Large Language and Vision Assistant
- FastChat: An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots
- Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models
- LLaMA: Open and Efficient Foundation Language Models
- Vid2seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning
- InternVid: A Large-scale Video-Text dataset


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
- [Zeliang Zhang](https://zhangaipi.github.io/) @ University of Rochester
- [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/) @ University of Rochester
- [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/index.html) @ University of Rochester



<a href="https://github.com/yunlong10/CAT-2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yunlong10/CAT-2" />
</a>
