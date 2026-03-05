# PERStance: Personality-guided enhanced multimodal stance detection
[architecture](https://github.com/jncsnlp/PERStance/blob/main/framework.png)

This repository contains the official PyTorch implementation code for Personality-guided enhanced multimodal stance detection: <a href="https://www.sciencedirect.com/science/article/pii/S0306457325005345?ref=pdf_download&fr=RR-4&rr=9d75c10a6be26b86">PERStance</a>.


## Installation
First, clone the repository locally:
```
git clone https://github.com/jncsnlp/PERStance.git
cd PERStance
```

## Requirements

Seeing in requirement.txt

You could using `pip install -r requirement.txt` to install the required packages.

When using LLM, you need to use the corresponding transformers version greater than 4.46.0.

 ## Usage

baseline:

```
sh scripts/run_baseline.sh
```

tmpt:

```
sh scripts/run_tmpt.sh
```

TMPT on zero-shot stance detection on mtse dataset:

```
>>> sh scripts/run_tmpt.sh
>>> input training dataset: [mtse, mccq, mruc, mtwq]: mtse
>>> input train dataset mode: [in_target, zero_shot]: zero_shot
>>> input model framework: [tmpt, tmpt_gpt_cot]: tmpt
>>> input model name: [bert_vit, roberta_vit, kebert_vit]: bert_vit
>>> input running mode: [sweep, wandb, normal]: normal
>>> input training cuda idx: Your Cuda index
```

The method proposed in our paper:

```
perstance_mian.py
```

## Acknowledgement
We refer to the code of TMPT. Thanks for their great contributions!

## Cite

```
@article{geng2026perstance,
  title={PERStance: Personality-guided enhanced multimodal stance detection},
  author={Geng, Guoqi and Zhan, Qianyi and Lu, Heng-Yang},
  journal={Information Processing \& Management},
  volume={63},
  number={4},
  pages={104593},
  year={2026},
  publisher={Elsevier}
}
```
Contact: gengguoqi@stu.jiangnan.edu.cn

If you encounter any difficulties or problems while using our dataset, please feel free to contact us. If you find our paper or code helpful, please give us a like. ❤️

