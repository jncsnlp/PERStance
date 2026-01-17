# PERStance

# Requirements

Seeing in requirement.txt

You could using `pip install -r requirement.txt` to install the required packages.

When using LLM, you need to use the corresponding transformers version greater than 4.46.0.

# baseline
sh scripts/run_baseline.sh

# tmpt
sh scripts/run_tmpt.sh

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
