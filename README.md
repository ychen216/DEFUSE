## Asymptotically Unbiased Estimation for Delayed Feedback Modeling via Label Correction

<!-- ### Citation
Please cite this paper if you used any content of this repo in your work:

```tex
@article{DBLP:journals/corr/abs-2202-06472,
  author    = {Yu Chen and
               Jiaqi Jin and
               Hui Zhao and
               Pengjie Wang and
               Guojun Liu and
               Jian Xu and
               Bo Zheng},
  title     = {Asymptotically Unbiased Estimation for Delayed Feedback Modeling via
               Label Correction},
  journal   = {CoRR},
  volume    = {abs/2202.06472},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.06472},
  eprinttype = {arXiv},
  eprint    = {2202.06472},
  timestamp = {Fri, 18 Feb 2022 12:23:53 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2202-06472.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
``` -->

### Environment Requirement
The code has been tested running under Python 3.8.10. The required packages are follows:
- numpy==1.18.5
- tqdm==4.61.2
- pandas==1.3.1
- scikit_learn==1.0.2
- tensorflow==2.4.1

### Example to Run the Codes
The instruction of commands has been clearly stated in the shell scripts:
We uploaded some shell scripts as a reference to run the code, however, the pathes should be modified accordingly.
- run_pretrain.sh: 
```
pretrain ckpts using the first 30 days information. we also provide the model checkpoints to reproduce the results on the public Criteo30d.
```
- run_base(_1d).sh
```
obtain baseline results under streaming setting.
```
- run_defuse.sh
```
obtain our DEFUSE results.
```

### Dataset
The criteo dataset is available at https://drive.google.com/file/d/1x4KktfZtls9QjNdFYKCjTpfjM4tG2PcK/view?usp=sharing


A preprint version of this paper is available at https://arxiv.org/abs/2202.06472.pdf
